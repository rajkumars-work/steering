"""
Stage A -- data-side pre-check (no GPU, no LLM).

Streams McAuley-Lab/Amazon-Reviews-2023 raw review jsonl for 8 categories
(the free, exact key -- category is the HF split, no classifier needed),
keeps reviews with 40-150 words (length-controlled so it doesn't confound
the readability target), and scores four candidate targets:
  - L_sentiment    : VADER compound score (nltk)                  [primary high-E candidate]
  - L_readability  : Flesch-Kincaid grade level (textstat)        [primary low-E candidate]
  - L_sentlen      : mean words/sentence                          [swap-1 alt A, rule-based]
  - L_firstperson  : first-person pronoun rate                    [swap-1 alt B, rule-based]

Also builds a second key -- review length bucket (short/medium/long by word
count) -- for swap-2 of the go/no-go gate.

Writes:
  distributions/stageA_reviews.parquet   -- raw kept reviews (text, category, scores) [not the shadow]
  distributions/stageA_perclass.npz      -- per-category w_b,g_b,v_b, ALL targets, key=category (the shadow)
  distributions/stageA_perclass_lenbucket.npz -- same, key=length-bucket (swap-2 shadow)
  distributions/stageA_summary.json      -- T,E,E/(E+T) + bootstrap CIs, every (key,target) combo tried

Usage: python collect_stagea.py [--n_per_cat 450]
"""
import argparse, json, os, re, sys, time
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "distributions")
os.makedirs(OUT, exist_ok=True)

CATEGORIES = [
    "Electronics", "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen",
    "Toys_and_Games", "Grocery_and_Gourmet_Food", "Office_Products", "Pet_Supplies",
]
MIN_WORDS, MAX_WORDS = 40, 150
BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{}.jsonl"
TARGETS = ["L_sentiment", "L_readability", "L_sentlen", "L_firstperson"]

FP_RE = re.compile(r"\b(i|i'm|i've|i'd|i'll|my|mine|myself|we|our|ours|us)\b", re.I)


def get_token():
    p = os.path.expanduser("~/.ssh/hf-read.key")
    if os.path.exists(p):
        return open(p).read().strip()
    return None


def collect_category(cat, n, token):
    from datasets import load_dataset
    url = BASE_URL.format(cat)
    ds = load_dataset("json", data_files={"full": url}, split="full", streaming=True, token=token)
    rows = []
    seen = 0
    for ex in ds:
        seen += 1
        txt = (ex.get("text") or "").strip()
        if not txt:
            continue
        nwords = len(txt.split())
        if nwords < MIN_WORDS or nwords > MAX_WORDS:
            continue
        rows.append({"category": cat, "text": txt, "n_words": nwords, "rating": ex.get("rating")})
        if len(rows) >= n:
            break
        if seen % 20000 == 0:
            print(f"    ...{cat}: scanned {seen}, kept {len(rows)}", flush=True)
    print(f"  {cat}: kept {len(rows)} / scanned {seen}", flush=True)
    return rows


def avg_sentence_len(t):
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    if not sents:
        return float(len(t.split()))
    return len(t.split()) / len(sents)


def first_person_rate(t):
    words = t.split()
    if not words:
        return 0.0
    return len(FP_RE.findall(t)) / len(words)


def score(df):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import textstat
    sia = SentimentIntensityAnalyzer()
    df["L_sentiment"] = df["text"].apply(lambda t: sia.polarity_scores(t)["compound"])
    df["L_readability"] = df["text"].apply(lambda t: textstat.flesch_kincaid_grade(t))
    df["L_sentlen"] = df["text"].apply(avg_sentence_len)
    df["L_firstperson"] = df["text"].apply(first_person_rate)
    return df


def lenbucket(n):
    if n < 70:
        return "short"
    if n < 110:
        return "medium"
    return "long"


def per_class_stats(df, key_col, targets):
    cats = sorted(df[key_col].unique())
    idx = {c: i for i, c in enumerate(cats)}
    keys = df[key_col].map(idx).to_numpy()
    B = len(cats)
    cnt = np.bincount(keys, minlength=B).astype(float)
    out = {"categories": np.array(cats, dtype=object), "count": cnt}
    for t in targets:
        v = df[t].to_numpy(float)
        s = np.bincount(keys, weights=v, minlength=B)
        s2 = np.bincount(keys, weights=v * v, minlength=B)
        nz = cnt > 0
        g = np.zeros(B); g[nz] = s[nz] / cnt[nz]
        vv = np.zeros(B); vv[nz] = s2[nz] / cnt[nz] - g[nz] ** 2
        out[f"g_{t}"] = g
        out[f"v_{t}"] = vv
    return out, keys


def TE(count, g, v):
    N = count.sum(); w = count / N
    gbar = float((w * g).sum())
    T = float((w * v).sum())
    E = float((w * (g - gbar) ** 2).sum())
    return T, E, gbar


def boot_ci_TE(df, key_col, target, cats, reps=2000, seed=0):
    """Bootstrap 95% CI for T, E, E/(E+T): resample reviews WITHIN each bin (vectorized)."""
    rng = np.random.default_rng(seed)
    B = len(cats)
    groups = [df.index[df[key_col] == c].to_numpy() for c in cats]
    vals = [df.loc[g, target].to_numpy(float) for g in groups]
    n_b = np.array([len(v) for v in vals])
    N = n_b.sum()

    # vectorized bootstrap: draw all resample indices for all bins/reps at once per bin
    g_reps = np.zeros((reps, B)); v_reps = np.zeros((reps, B))
    for b in range(B):
        idx = rng.integers(0, n_b[b], size=(reps, n_b[b]))
        samp = vals[b][idx]  # (reps, n_b)
        g_reps[:, b] = samp.mean(axis=1)
        v_reps[:, b] = samp.var(axis=1)
    w = n_b / N
    gbar_reps = (g_reps * w).sum(axis=1)
    T_reps = (v_reps * w).sum(axis=1)
    E_reps = ((g_reps - gbar_reps[:, None]) ** 2 * w).sum(axis=1)
    F_reps = np.where((E_reps + T_reps) > 0, E_reps / (E_reps + T_reps), 0.0)

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
    return {"T_CI95": ci(T_reps), "E_CI95": ci(E_reps), "EfracCI95": ci(F_reps)}


def summarize(df, key_col, targets, tag):
    pc, _ = per_class_stats(df, key_col, targets)
    cats = list(pc["categories"])
    res = {}
    for t in targets:
        T, E, gbar = TE(pc["count"], pc[f"g_{t}"], pc[f"v_{t}"])
        frac = E / (E + T) if (E + T) > 0 else 0.0
        ci = boot_ci_TE(df, key_col, t, cats)
        res[t] = {"T": T, "E": E, "gbar": gbar, "E_frac": frac, **ci,
                  "per_bin_g": {c: float(pc[f"g_{t}"][i]) for i, c in enumerate(cats)}}
    return pc, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_cat", type=int, default=450)
    args = ap.parse_args()

    token = get_token()
    t0 = time.time()
    rows = []
    for cat in CATEGORIES:
        rows.extend(collect_category(cat, args.n_per_cat, token))
    df = pd.DataFrame(rows)
    print(f"Total kept: {len(df)} reviews across {df['category'].nunique()} categories "
          f"({time.time()-t0:.1f}s to stream)")

    df = score(df)
    df["lenbucket"] = df["n_words"].apply(lenbucket)
    df.to_parquet(os.path.join(OUT, "stageA_reviews.parquet"))

    summary = {"n_total": int(len(df)),
               "n_per_category": {c: int((df["category"] == c).sum()) for c in sorted(df["category"].unique())},
               "conditions": {}}

    # Original pair: key=category, targets = sentiment (predicted high-E), readability (predicted low-E)
    pc_cat, res_cat = summarize(df, "category", TARGETS, "category")
    np.savez(os.path.join(OUT, "stageA_perclass.npz"), **pc_cat)
    summary["conditions"]["category"] = res_cat

    # Swap 2: key = length bucket, same two primary targets (+ sentiment/readability only, per spec)
    pc_lb, res_lb = summarize(df, "lenbucket", ["L_sentiment", "L_readability"], "lenbucket")
    np.savez(os.path.join(OUT, "stageA_perclass_lenbucket.npz"), **pc_lb)
    summary["conditions"]["lenbucket"] = res_lb

    with open(os.path.join(OUT, "stageA_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Stage A gate check: key=category ===")
    for t in TARGETS:
        s = res_cat[t]
        print(f"  {t:16s} T={s['T']:8.4f} E={s['E']:8.4f} E/(E+T)={s['E_frac']:.3f}  "
              f"CI95={tuple(round(x,3) for x in s['EfracCI95'])}")
    print("\n=== Stage A gate check: key=lenbucket (swap 2) ===")
    for t in ["L_sentiment", "L_readability"]:
        s = res_lb[t]
        print(f"  {t:16s} T={s['T']:8.4f} E={s['E']:8.4f} E/(E+T)={s['E_frac']:.3f}  "
              f"CI95={tuple(round(x,3) for x in s['EfracCI95'])}")
    print("\nSTAGEA-DONE ->", os.path.join(OUT, "stageA_summary.json"))


if __name__ == "__main__":
    main()
