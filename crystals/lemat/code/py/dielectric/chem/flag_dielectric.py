def flag_dielectric(item):
    eps = item["eps_0"]
    Eg = item["bandgap"]
    fam = item["family"]

    # Metallic instability
    if Eg < 0.3 and eps > 100:
        return "metallic screening artifact likely"

    # Wide-gap fluoride
    if fam == "f_based" and Eg > 4 and eps > 25:
        return "too large for wide-gap fluoride"

    # Halide extreme values
    if fam in ["f_based", "cl_based"] and eps > 80:
        return "very unusual dielectric for halide"

    # Oxide large soft-mode
    if fam == "oxide" and eps > 300 and Eg > 2:
        return "possible soft-mode instability"

    # Absolute ceiling
    if eps > 2000:
        return "likely numerical divergence"

    return None

