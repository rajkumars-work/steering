import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""
Labelers for the ImageNet-domain steering-paper evidence (reconstruction).

Provides the seven target labels L and the cheap key pi used in Evidence:
  targets L:  aesthetic (CLIP ViT-L/14 -> LAION MLP), brightness (mean luminance),
              filesize_per_mp (JPEG bytes / megapixel), and 4 CLIP concept similarities.
  key pi:     ResNet-50 predicted ImageNet class (0..999), for binning generated images.

All models are loaded from the volatile cache. Batch API over PIL.Image lists.

NOTE (reconstruction): the original evidence scripts were lost; the four concept
prompts and the JPEG quality are reconstruction choices, recorded here so the
numbers are reproducible from this file.
"""
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODELS = _os.environ.get("MODELS", _os.path.join(_ROOT, "models"))
CLIP_REPO = "openai/clip-vit-large-patch14"
AESTH_WEIGHTS = f"{MODELS}/laion_aesthetic_sac_logos_ava1_l14_linearMSE.pth"
JPEG_QUALITY = 90

# Reconstruction choice — 4 broad, class-aligned concepts (animal kept: used in the joint).
CONCEPTS = {
    "animal":  "a photo of an animal",
    "vehicle": "a photo of a vehicle",
    "food":    "a photo of food",
    "nature":  "a natural outdoor landscape",
}


class _AestheticMLP(nn.Module):
    """LAION improved-aesthetic-predictor head (christophschuhmann), on 768-d CLIP L/14."""
    def __init__(self, d=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Labeler:
    def __init__(self, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        from torchvision.models import resnet50, ResNet50_Weights
        self.device = device

        # CLIP (embeddings for aesthetic + concept sims)
        self.clip = CLIPModel.from_pretrained(CLIP_REPO).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_REPO)

        # LAION aesthetic head
        self.aesth = _AestheticMLP().to(device).eval()
        self.aesth.load_state_dict(torch.load(AESTH_WEIGHTS, map_location=device))

        # Pre-encode the concept text embeddings (projected + L2-normalized)
        toks = self.clip_proc(text=list(CONCEPTS.values()), padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.clip.text_model(input_ids=toks["input_ids"], attention_mask=toks.get("attention_mask"))
            t = self.clip.text_projection(out.pooler_output)
        self.concept_emb = (t / t.norm(dim=-1, keepdim=True))  # [4, 768]
        self.concept_names = list(CONCEPTS.keys())

        # ResNet-50 key (pi)
        self.rn_weights = ResNet50_Weights.IMAGENET1K_V2
        self.resnet = resnet50(weights=self.rn_weights).to(device).eval()
        self.rn_tf = self.rn_weights.transforms()

    @torch.no_grad()
    def _clip_image_emb(self, imgs):
        px = self.clip_proc(images=imgs, return_tensors="pt").to(self.device)
        out = self.clip.vision_model(pixel_values=px["pixel_values"])
        e = self.clip.visual_projection(out.pooler_output)
        return e / e.norm(dim=-1, keepdim=True)  # [N,768] projected + L2-normalized

    @torch.no_grad()
    def labels(self, imgs):
        """imgs: list[PIL.Image] (RGB). Returns dict of np arrays, len N each."""
        if isinstance(imgs, Image.Image):
            imgs = [imgs]
        N = len(imgs)
        emb = self._clip_image_emb(imgs)                       # [N,768]
        aesthetic = self.aesth(emb).squeeze(-1).float().cpu().numpy()
        sims = (emb @ self.concept_emb.T).float().cpu().numpy()  # [N,4] cosine (unit vecs)

        brightness = np.empty(N, np.float32)
        filesize = np.empty(N, np.float32)
        for i, im in enumerate(imgs):
            a = np.asarray(im.convert("RGB"), np.float32)
            lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
            brightness[i] = lum.mean() / 255.0
            buf = io.BytesIO(); im.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
            mp = (im.size[0] * im.size[1]) / 1e6
            filesize[i] = buf.getbuffer().nbytes / mp

        out = {"aesthetic": aesthetic, "brightness": brightness, "filesize_per_mp": filesize}
        for j, name in enumerate(self.concept_names):
            out[f"sim_{name}"] = sims[:, j]
        return out

    @torch.no_grad()
    def key(self, imgs):
        """ResNet-50 predicted ImageNet class (the cheap key pi). Returns np int array."""
        if isinstance(imgs, Image.Image):
            imgs = [imgs]
        x = torch.stack([self.rn_tf(im.convert("RGB")) for im in imgs]).to(self.device)
        logits = self.resnet(x)
        return logits.argmax(-1).cpu().numpy()

    TARGETS = ["aesthetic", "brightness", "filesize_per_mp", "sim_animal", "sim_vehicle", "sim_food", "sim_nature"]


if __name__ == "__main__":
    # Smoke test: synthetic images, confirm every labeler runs and shapes are right.
    L = Labeler()
    imgs = [Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8)) for _ in range(3)]
    lab = L.labels(imgs)
    k = L.key(imgs)
    print("targets:", {t: float(np.round(lab[t][0], 4)) for t in Labeler.TARGETS})
    print("resnet key (class idx) for 3 imgs:", k.tolist())
    print("SMOKE-OK", "ntargets=", len(Labeler.TARGETS))
