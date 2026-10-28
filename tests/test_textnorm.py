from qal_scribe.textnorm import edit_distance, fold_homophones, normalize


def test_fold_homophones_all_orders():
    assert fold_homophones("ሐሑሒሓሔሕሖ") == "ሀሁሂሃሄህሆ"
    assert fold_homophones("ሠላም") == "ሰላም"
    assert fold_homophones("ዐይን") == "አይን"


def test_normalize_drops_ethiopic_punctuation():
    assert normalize("ሰላም፣ እንዴት ነህ?  ደህና ነኝ።") == "ሰላም እንዴት ነህ ደህና ነኝ"


def test_edit_distance():
    assert edit_distance(list("ሰላም"), list("ሰላም")) == 0
    assert edit_distance(list("ሰላም"), list("ሰም")) == 1
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1
