from qal_scribe.textnorm import FOLD, edit_distance, fold_homophones, normalize


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


def test_every_family_folds_in_every_order():
    for order in range(7):
        for target, other in [("ሀ", "ሐ"), ("ሀ", "ኀ"), ("ሰ", "ሠ"), ("አ", "ዐ"), ("ጸ", "ፀ")]:
            expected = "አ" if order == 3 and other == "ዐ" else chr(ord(target) + order)
            assert fold_homophones(chr(ord(other) + order)) == expected


def test_labialised_forms():
    assert fold_homophones("ሧ") == "ሷ"  # ሠ series 8th order
    assert fold_homophones("ፇ") == "ጿ"  # ፀ series 8th order
    assert fold_homophones("ሗ") == "ኋ"
    assert fold_homophones("ኇ") == "ሇ"
    assert fold_homophones("ኈኊኋኌኍ") == "ኈኊኋኌኍ"  # no twin in the ሀ series: kept, not forced onto ሆ


def test_glottal_fourth_order_folds_to_alef():
    assert fold_homophones("ዓለም ኣለም") == "አለም አለም"


def test_fold_keeps_distinct_sounds_apart():
    text = "ሀሃህ ሰሳስ አኣእ ጸጻጽ ቀቐ ኸኽ ጠጨ"
    assert fold_homophones(text) == text.replace("ኣ", "አ")


def test_fold_is_length_preserving_idempotent_and_ethiopic_only():
    assert all(len(k) == 1 and len(v) == 1 and "ሀ" <= k <= "፿" for k, v in FOLD.items())
    assert not set(FOLD) & set(FOLD.values())
    text = "".join(FOLD) + "ሰላም abc 12"
    assert len(fold_homophones(text)) == len(text)
    assert fold_homophones(fold_homophones(text)) == fold_homophones(text)
