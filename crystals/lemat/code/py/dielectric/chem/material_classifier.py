from pymatgen.core import Composition


ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba"}
HALIDES = {"F", "Cl", "Br", "I"}
OXIDE = {"O"}
HEAVY_POST = {"Bi", "Pb", "Sn", "Sb", "Tl"}


def classify_material(comp_str):
    comp = Composition(comp_str)
    el_amt = comp.get_el_amt_dict()
    elements = set(el_amt.keys())

    result = {}

    # -------------------------
    # FAMILY
    # -------------------------
    if elements & OXIDE:
        result["family"] = "oxide"
    elif elements & HALIDES:
        halides_present = elements & HALIDES
        if len(halides_present) > 1:
            result["family"] = "mixed_halide"
        else:
            result["family"] = f"{list(halides_present)[0].lower()}_based"
    else:
        result["family"] = "other"

    # -------------------------
    # STRUCTURE GUESS
    # -------------------------
    n_elements = len(elements)

    # Simple binary AX
    if n_elements == 2:
        result["structure_guess"] = "simple_binary"
        return result

    # Oxide Perovskite Heuristics
    if result["family"] == "oxide":
        # A2BB'O6 pattern
        if comp.reduced_formula.count("O6"):
            result["structure_guess"] = "double_perovskite_like"

        # ABO3 pattern
        elif comp.reduced_formula.count("O3"):
            result["structure_guess"] = "perovskite_like"

        else:
            result["structure_guess"] = "complex_oxide"

    # Halide Elpasolite Heuristic (A2BB'X6)
    elif "based" in result["family"]:
        formula = comp.reduced_formula

        if "6" in formula and n_elements >= 3:
            result["structure_guess"] = "elpasolite_like"
        elif elements & HEAVY_POST:
            result["structure_guess"] = "heavy_metal_halide"
        else:
            result["structure_guess"] = "complex_halide"

    else:
        result["structure_guess"] = "complex_salt"

    return result


if __name__ == "__main__":
    points = [
        "Na2BeGaF7",
        "Li2GaPbF7",
        "RbPb2Cl5",
        "Ca2ZrTiO6",
        "AlBiF4",
        "GaBiF4",
    ]

    for p in points:
        print(p, classify_material(p))
