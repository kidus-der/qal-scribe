import random

import pytest

from qal_scribe.config import Config
from qal_scribe.postprocess import add_punctuation, numbers_to_digits, process, replace_words

NUMBERS = Config(numbers_to_digits=True)
PUNCT = Config(punctuation=True)


# --- pipeline -------------------------------------------------------------------------------------------------


def test_flags_off_only_normalises_whitespace():
    assert process("  ሠላም \t ሃያ\n አምስት ምን ", Config()) == "ሠላም ሃያ አምስት ምን"
    assert process("", Config()) == ""


def test_word_list_is_always_on():
    assert process("ከርሰር ሰላም", Config(), {"ከርሰር": "Cursor"}) == "Cursor ሰላም"


def test_order_word_list_then_homophones_then_numbers_then_punctuation():
    cfg = Config(merge_homophones=True, numbers_to_digits=True, punctuation=True)
    assert process("ሠላም ሁለት ሦስት", cfg, {"ሠላም": "ሠላም"}) == "ሰላም 2 3።"
    assert process("ኤክስ ዋጋው ሃያ ብር ነው", cfg, {"ኤክስ": "ሶስት"}) == "3 ዋጋው 20 ብር ነው።"


def test_features_are_independent():
    assert process("ሠላም ሁለት", Config(merge_homophones=True)) == "ሰላም ሁለት"
    assert process("ሠላም ሁለት", NUMBERS) == "ሠላም 2"
    assert process("ሠላም ሁለት", PUNCT) == "ሠላም ሁለት።"


# --- word list ------------------------------------------------------------------------------------------------


def test_words_single_word():
    assert replace_words("ከርሰር ክፈት", {"ከርሰር": "Cursor"}) == "Cursor ክፈት"


def test_words_multi_word_phrase():
    assert replace_words("ኮድ ኤዲተር ክፈት", {"ኮድ ኤዲተር": "code editor"}) == "code editor ክፈት"


def test_words_longest_phrase_wins_regardless_of_order():
    words = {"ኮድ": "code", "ኮድ ኤዲተር": "editor", "ኮድ ኤዲተር ክፈት": "open editor"}
    assert replace_words("ኮድ ኤዲተር ክፈት", words) == "open editor"
    assert replace_words("ኮድ ኤዲተር", words) == "editor"
    assert replace_words("ኮድ ጻፍ", words) == "code ጻፍ"


def test_words_whole_word_only():
    assert replace_words("ከርሰርን ከርሰርከርሰር በከርሰር", {"ከርሰር": "Cursor"}) == "ከርሰርን ከርሰርከርሰር በከርሰር"


def test_words_next_to_punctuation():
    assert replace_words("ከርሰር። (ከርሰር) ከርሰር፣", {"ከርሰር": "Cursor"}) == "Cursor። (Cursor) Cursor፣"


def test_words_match_any_homophone_spelling():
    assert replace_words("ሠላም ሰላም ዐይን", {"ሰላም": "hello", "አይን": "eye"}) == "hello hello eye"


def test_words_keep_original_spelling_around_matches():
    assert replace_words("ሠላም ከርሰር ሠላም", {"ከርሰር": "Cursor"}) == "ሠላም Cursor ሠላም"


def test_words_empty_replacement_deletes_the_word():
    assert process("እ ሰላም እ ዓለም", Config(), {"እ": ""}) == "ሰላም ዓለም"


def test_words_replacement_is_literal():
    assert replace_words("ከርሰር", {"ከርሰር": r"\1 \g<0> $1"}) == r"\1 \g<0> $1"


def test_words_regex_characters_in_entries_are_literal():
    assert replace_words("ሲ++ ጻፍ", {"ሲ++": "C++"}) == "C++ ጻፍ"


def test_words_repeated_and_adjacent_matches():
    assert replace_words("ከርሰር ከርሰር", {"ከርሰር": "Cursor"}) == "Cursor Cursor"


def test_words_none_or_empty_dict_changes_nothing():
    assert process("ከርሰር", Config(), None) == "ከርሰር"
    assert process("ከርሰር", Config(), {}) == "ከርሰር"
    assert replace_words("ከርሰር", {"": "x", "  ": "y"}) == "ከርሰር"


# --- numbers --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("ዜሮ", "0"), ("አንድ", "1"), ("ሁለት", "2"), ("ሶስት", "3"), ("አራት", "4"), ("አምስት", "5"),
        ("ስድስት", "6"), ("ሰባት", "7"), ("ስምንት", "8"), ("ዘጠኝ", "9"),
    ],
)
def test_standalone_digits(spoken, digits):
    assert numbers_to_digits(spoken) == digits


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("አስር", "10"), ("ሃያ", "20"), ("ሰላሳ", "30"), ("አርባ", "40"), ("ሃምሳ", "50"), ("ስልሳ", "60"), ("ሰባ", "70"),
        ("ሰማንያ", "80"), ("ዘጠና", "90"),
        ("አስራ አንድ", "11"), ("አስራ ሁለት", "12"), ("አስራ ዘጠኝ", "19"),
        ("ሃያ አንድ", "21"), ("ሰላሳ ሶስት", "33"), ("ዘጠና ዘጠኝ", "99"),
    ],
)
def test_tens_and_teens(spoken, digits):
    assert numbers_to_digits(spoken) == digits


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("መቶ", "100"), ("አንድ መቶ", "100"), ("ሁለት መቶ", "200"), ("ሁለት መቶ ሃምሳ", "250"), ("መቶ አምስት", "105"),
        ("ሁለት መቶ አስራ አምስት", "215"), ("ዘጠኝ መቶ ዘጠና ዘጠኝ", "999"), ("ሶስት መቶ አስር", "310"),
        ("ሺህ", "1000"), ("አንድ ሺህ", "1000"), ("ሶስት ሺህ አምስት መቶ", "3500"), ("አስር ሺህ", "10000"),
        ("ሃያ ሁለት ሺህ", "22000"), ("መቶ ሺህ", "100000"), ("ሁለት መቶ ሺህ", "200000"), ("ሺ አምስት", "1005"),
        ("አንድ ሺህ ዘጠኝ መቶ ዘጠና አምስት", "1995"), ("ሁለት ሺህ ሃያ ስድስት", "2026"),
        ("ሚሊዮን", "1000000"), ("ሁለት ሚሊዮን አምስት መቶ ሺህ", "2500000"), ("ሶስት ቢሊዮን", "3000000000"),
        ("አንድ ሚሊዮን ሁለት መቶ ሰላሳ አራት ሺህ አምስት መቶ ስልሳ ሰባት", "1234567"),
    ],
)
def test_compounds(spoken, digits):
    assert numbers_to_digits(spoken) == digits


_UNITS = ["ዜሮ", "አንድ", "ሁለት", "ሶስት", "አራት", "አምስት", "ስድስት", "ሰባት", "ስምንት", "ዘጠኝ"]
_TENS = ["-", "አስር", "ሃያ", "ሰላሳ", "አርባ", "ሃምሳ", "ስልሳ", "ሰባ", "ሰማንያ", "ዘጠና"]
_SCALES = [(10**12, "ትሪሊዮን"), (10**9, "ቢሊዮን"), (10**6, "ሚሊዮን"), (10**3, "ሺህ")]


def spell_small(n):  # 1..999
    words, h, r = [], n // 100, n % 100
    if h:
        words += ([_UNITS[h]] if h > 1 else []) + ["መቶ"]
    if 10 < r < 20:
        words += ["አስራ", _UNITS[r - 10]]
    elif r >= 10:
        words += [_TENS[r // 10]] + ([_UNITS[r % 10]] if r % 10 else [])
    elif r:
        words.append(_UNITS[r])
    return words


def spell(n):
    words = []
    for scale, word in _SCALES:
        if n >= scale:
            words += [*spell_small(n // scale), word]
            n %= scale
    return " ".join(words + (spell_small(n) if n else [])) or "ዜሮ"


def test_every_number_up_to_ten_thousand_round_trips():
    for n in range(10_001):
        assert numbers_to_digits(spell(n)) == str(n), spell(n)


def test_large_numbers_round_trip():
    rng = random.Random(7)
    for _ in range(3000):
        n = rng.randrange(10 ** rng.randrange(4, 15))
        assert numbers_to_digits(spell(n)) == str(n), spell(n)


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("አስራ ዘጠኝ መቶ ሰማንያ አምስት", "1985"), ("አስራ ዘጠኝ መቶ ሰባ ስድስት", "1976"), ("አስራ ዘጠኝ መቶ", "1900"),
        ("ሃያ አምስት መቶ", "2500"), ("አስር መቶ", "1000"), ("አስራ ሁለት መቶ ሃምሳ", "1250"), ("ዘጠና መቶ", "9000"),
        ("በአስራ ዘጠኝ መቶ ሰባ ስድስት ተወለድኩ", "በ1976 ተወለድኩ"), ("ሃያ አምስት መቶ ሺህ", "2500000"),
    ],
)
def test_two_digit_value_before_hundred_is_that_many_hundreds(spoken, digits):
    assert numbers_to_digits(spoken) == digits


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("ሁለት ሺህ አስራ ዘጠኝ መቶ", "2000 1900"), ("ሁለት ሚሊዮን ሃያ አምስት መቶ", "2000000 2500"),
        ("ሁለት መቶ ሃያ አምስት መቶ", "200 2500"), ("መቶ ሃያ መቶ", "100 2000"), ("ሃያ ሁለት መቶ ሶስት መቶ", "2200 300"),
        ("ሁለት መቶ ሃያ አምስት", "225"), ("አስራ ዘጠኝ መቶ ሰማንያ ሶስት መቶ", "1900 8300"),
    ],
)
def test_year_style_hundreds_do_not_merge_with_neighbours(spoken, digits):
    assert numbers_to_digits(spoken) == digits


@pytest.mark.parametrize(("a", "b"), [(5, 7), (20, 30), (13, 20), (7, 20), (900, 800), (3000, 4000), (10, 5)])
def test_neighbouring_numbers_stay_separate(a, b):
    assert numbers_to_digits(f"{spell(a)} {spell(b)}") == f"{a} {b}"


@pytest.mark.parametrize(("a", "b"), [(40, 6), (200, 50), (3000, 500), (1_000_000, 2)])
def test_neighbouring_numbers_that_form_one_number_merge(a, b):
    assert numbers_to_digits(f"{spell(a)} {spell(b)}") == str(a + b)


@pytest.mark.parametrize(
    ("variant", "digits"),
    [
        ("ሦስት", "3"), ("አሥር", "10"), ("ዐሥር", "10"), ("አሥራ ሁለት", "12"), ("ሀያ", "20"), ("ሐያ", "20"), ("ሓያ", "20"),
        ("ሀምሳ", "50"), ("ሰላሣ", "30"), ("ሥልሳ", "60"), ("ሰማኒያ", "80"), ("ሚሊየን", "1000000"),
        ("ቢሊየን", "1000000000"), ("ሺሕ", "1000"), ("ስድሳ", "60"), ("ስድሳ አምስት", "65"),
    ],
)
def test_spelling_variants(variant, digits):
    assert numbers_to_digits(variant) == digits


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("ሁለት ነጥብ አምስት", "2.5"), ("ዜሮ ነጥብ ሁለት አምስት", "0.25"), ("ሶስት ነጥብ አንድ አራት", "3.14"),
        ("ሃያ ነጥብ አምስት", "20.5"), ("ሁለት ነጥብ አምስት ሚሊዮን ብር", "2.5 ሚሊዮን ብር"),
    ],
)
def test_decimals(spoken, digits):
    assert numbers_to_digits(spoken) == digits


@pytest.mark.parametrize("text", ["ነጥብ", "ጥሩ ነጥብ", "ነጥብ ጥሩ"])
def test_point_the_noun_is_not_a_decimal(text):
    assert numbers_to_digits(text) == text


def test_point_needs_a_number_before_and_digit_words_after():
    assert numbers_to_digits("ሶስት ነጥብ አግኝቷል") == "3 ነጥብ አግኝቷል"
    assert numbers_to_digits("ነጥብ አምስት") == "ነጥብ 5"
    assert numbers_to_digits("ሁለት ነጥብ ሃያ") == "2 ነጥብ 20"
    assert numbers_to_digits("ሁለት። ነጥብ አምስት") == "2። ነጥብ 5"


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("አንድ ሁለት ሶስት", "1 2 3"), ("ሃያ ሃያ", "20 20"), ("አምስት ሃያ", "5 20"), ("አስር አንድ", "10 1"),
        ("ሃያ አስር", "20 10"), ("ሁለት ሺህ ሁለት ሺህ", "2000 2000"), ("ሁለት መቶ ሶስት መቶ", "200 300"),
        ("ሺህ ሺህ", "1000 1000"), ("ሺህ ሚሊዮን", "1000 1000000"), ("ሁለት ሁለት መቶ", "2 200"),
        ("ዜሮ አንድ", "0 1"), ("መቶ መቶ", "100 100"),
    ],
)
def test_separate_numbers_are_never_merged(spoken, digits):
    assert numbers_to_digits(spoken) == digits


def test_punctuation_between_number_words_separates_them():
    assert numbers_to_digits("ሁለት፣ ሶስት") == "2፣ 3"
    assert numbers_to_digits("ሃያ። አንድ") == "20። 1"
    assert numbers_to_digits("ሁለት። (አምስት)") == "2። (5)"
    assert numbers_to_digits("ሃያ, አንድ") == "20, 1"


@pytest.mark.parametrize(
    "word", ["አስራ", "አንድነት", "ሁለተኛ", "አምስተኛ", "ሁለቱም", "ሰላም", "ዓለም", "አንዱ", "መቶኛ", "ሶስተኛ"]
)
def test_non_number_words_are_untouched(word):
    assert numbers_to_digits(word) == word
    assert numbers_to_digits(f"ሰላም {word} ዓለም") == f"ሰላም {word} ዓለም"


def test_teen_prefix_needs_a_unit():
    assert numbers_to_digits("አስራ ብር") == "አስራ ብር"
    assert numbers_to_digits("አስራ አስር") == "አስራ 10"


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("በሶስት ሰዓት", "በ3 ሰዓት"), ("ከአስራ ሁለት ሰዓት", "ከ12 ሰዓት"), ("ለሁለት መቶ ብር", "ለ200 ብር"),
        ("የአንድ ሰው", "የ1 ሰው"), ("ከመቶ በላይ", "ከ100 በላይ"), ("ከሺህ በላይ", "ከ1000 በላይ"),
        ("ከአስር እስከ ሃያ", "ከ10 እስከ 20"),
    ],
)
def test_prepositions_stay_attached(spoken, written):
    assert numbers_to_digits(spoken) == written


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("በሺህ ዘጠኝ መቶ ስልሳ ሶስት", "በ1963"), ("በሺህ ዘጠኝ መቶ ሰባ ስድስት ዓ.ም", "በ1976 ዓ.ም"),  # "in 1963"
        ("አርባ አምስቱ", "45ቱ"), ("ከመቶ አስራ ሰባቱ", "ከ117ቱ"), ("ዘጠኝ መቶ ዘጠና ዘጠኙ", "999ኙ"),  # "the 45"
        ("ሁለቱ ልጆች", "ሁለቱ ልጆች"), ("ከሁለቱ", "ከሁለቱ"), ("በሚሊዮን የሚቆጠሩ", "በሚሊዮን የሚቆጠሩ"),  # words
    ],
)
def test_years_after_be_and_the_definite_ending(spoken, written):
    assert numbers_to_digits(spoken) == written


def test_a_lone_prefix_letter_is_left_alone():  # the ASR writes them: የ ሁለት
    assert numbers_to_digits("የ ሁለት በ ከ") == "የ 2 በ ከ"


def test_percent_is_not_read_as_by_hundred():
    assert numbers_to_digits("ሃያ በመቶ") == "20 በመቶ"
    assert numbers_to_digits("በመቶ") == "በመቶ"
    assert numbers_to_digits("ሃያ አምስት በመቶ ጨምሯል") == "25 በመቶ ጨምሯል"
    assert numbers_to_digits("በሺህ የሚቆጠሩ ሰዎች") == "በሺህ የሚቆጠሩ ሰዎች"


def test_numbers_inside_sentences():
    assert numbers_to_digits("ዛሬ ሃያ አምስት ብር ከፈልኩ") == "ዛሬ 25 ብር ከፈልኩ"
    assert numbers_to_digits("ሁለት ሰው ሶስት መኪና") == "2 ሰው 3 መኪና"
    assert numbers_to_digits("በሶስት ሰዓት ሁለት ሰው መጣ") == "በ3 ሰዓት 2 ሰው መጣ"
    assert numbers_to_digits("ዋጋው ሶስት ሺህ ብር ነው።") == "ዋጋው 3000 ብር ነው።"


def test_digits_and_latin_pass_through():
    assert numbers_to_digits("12 ሁለት Cursor 3.5") == "12 2 Cursor 3.5"
    assert numbers_to_digits("ሁለት2 ሁለት-ሶስት") == "ሁለት2 ሁለት-ሶስት"
    assert numbers_to_digits("") == ""


# --- punctuation ----------------------------------------------------------------------------------------------


def test_statement_gets_ethiopic_full_stop():
    assert add_punctuation("ሰላም ነው") == "ሰላም ነው።"


@pytest.mark.parametrize(
    "sentence",
    [
        "ምን ሰራህ", "ማን መጣ", "ቤትህ የት ነው", "መቼ ትመጣለህ", "እንዴት ነህ", "ለምን አልመጣህም", "ስንት ነው", "የትኛው ይሻላል",
        "ሻይ ወይስ ቡና", "ይመጣ ይሆን", "ይህ ምንድነው", "እሱ ማነው", "ከየት መጣህ", "ለማን ሰጠኸው", "የማን ነው", "በምን መጣህ",
        "ትመጣለህ ወይ", "ጥሩ ነው አይደል",
    ],
)
def test_questions(sentence):
    assert add_punctuation(sentence) == sentence + "?"


@pytest.mark.parametrize(
    "sentence",
    ["ምንም አላየሁም", "ማንም አልመጣም", "የትም አልሄድኩም", "መቼም አልረሳውም", "ሻይ ወይ ቡና እፈልጋለሁ", "ወደ ገበያ ሄድን", "ደህና ነኝ"],
)
def test_lookalikes_are_statements(sentence):
    assert add_punctuation(sentence) == sentence + "።"


def test_yes_no_suffix_is_not_guessed():
    assert add_punctuation("መጣህን") == "መጣህን።"


@pytest.mark.parametrize(
    "sentence", ["ምን እንደሆነ አላውቅም", "ማን እንደመጣ አላውቅም", "ምን ትፈልጋለህ ብዬ ጠየቅኩት", "እንዴት ነህ ብላ ጠየቀች"]
)
def test_reported_questions_are_statements(sentence):
    assert add_punctuation(sentence) == sentence + "።"


def test_only_the_last_sentence_decides():
    assert add_punctuation("ሰላም ነው። እንዴት ነህ") == "ሰላም ነው። እንዴት ነህ?"
    assert add_punctuation("እንዴት ነህ? ደህና ነኝ") == "እንዴት ነህ? ደህና ነኝ።"


@pytest.mark.parametrize(
    "text", ["ሰላም ነው።", "እንዴት ነህ?", "እንዴት ነህ፧", "ዋው!", "ሰላም…", "ሰላም፣", "(ሰላም)", "ምን ሰራህ።"]
)
def test_existing_punctuation_is_kept(text):
    assert add_punctuation(text) == text


@pytest.mark.parametrize("text", ["ሰላም ነው", "እንዴት ነህ", "ሄድኩ ነገር ግን አልተገኘም", "ዋጋው 20 ብር ነው"])
def test_punctuation_is_idempotent(text):
    once = add_punctuation(text)
    assert add_punctuation(once) == once


@pytest.mark.parametrize("text", ["", "   ", "25", "Cursor", "hello world", "።", "3.5"])
def test_text_without_ethiopic_letters_is_left_alone(text):
    assert add_punctuation(text) == text.strip()


def test_mixed_text_is_punctuated():
    assert add_punctuation("ዋጋው 25 ብር ነው") == "ዋጋው 25 ብር ነው።"
    assert add_punctuation("Cursor ክፈት") == "Cursor ክፈት።"
    assert add_punctuation("Cursor ምንድነው") == "Cursor ምንድነው?"


@pytest.mark.parametrize("opener", ["ነገር ግን", "ሆኖም", "ምክንያቱም"])
def test_comma_before_clause_openers(opener):
    assert add_punctuation(f"ወደ ገበያ ሄድኩ {opener} ምንም አላገኘሁም") == f"ወደ ገበያ ሄድኩ፣ {opener} ምንም አላገኘሁም።"


def test_comma_rule_is_conservative():
    assert add_punctuation("ነገር ግን አልመጣም") == "ነገር ግን አልመጣም።"  # sentence-initial
    assert add_punctuation("እሱ ግን አልመጣም") == "እሱ ግን አልመጣም።"  # a bare ግን follows its subject
    assert add_punctuation("ሄድኩ፣ ነገር ግን አልመጣም") == "ሄድኩ፣ ነገር ግን አልመጣም።"  # already there
    assert add_punctuation("ሄድኩ። ሆኖም ተመለሰ") == "ሄድኩ። ሆኖም ተመለሰ።"


def test_comma_rule_skips_the_noun_after_a_demonstrative():
    for demonstrative in ["ይህ", "ያ", "ይሄ", "ይኸ"]:
        text = f"{demonstrative} ነገር ግን ጥሩ ነው"
        assert add_punctuation(text) == text + "።"
    assert add_punctuation("ከሰይህ ነገር ግን ጥሩ ነው") == "ከሰይህ፣ ነገር ግን ጥሩ ነው።"  # ይህ only as a whole word
    for text in ("አንድ ነገር ግን ልንገርህ", "ሌላ ነገር ግን አለ", "1 ነገር ግን ቀረ"):
        assert add_punctuation(text) == text + "።"


def test_maybe_written_as_two_words_is_not_a_question():
    assert add_punctuation("ምን አልባት ይመጣል") == "ምን አልባት ይመጣል።"
    assert add_punctuation("ምን አልባት ማን መጣ") == "ምን አልባት ማን መጣ?"
    assert add_punctuation("ምን አልባትም ይመጣል") == "ምን አልባትም ይመጣል።"


def test_question_word_heading_a_reported_clause_is_a_statement():
    for text in ("ምን እንደምትፈልግ ንገረኝ", "ማን እንደመጣ ንገረኝ", "ምን ማድረግ እንዳለብኝ አስረዳኝ"):
        assert add_punctuation(text) == text + "።"
    assert add_punctuation("እንዴት እንደዚህ ሆነ") == "እንዴት እንደዚህ ሆነ?"  # እንደዚህ is "like this", not a verb
    assert add_punctuation("እንደ ልማዱ ማን መጣ") == "እንደ ልማዱ ማን መጣ?"  # እንደ before the question word


def test_documented_wrong_marks():  # pinned so a change to the rules is a conscious one
    assert add_punctuation("እንዴት ደስ ይላል") == "እንዴት ደስ ይላል?"
    assert add_punctuation("ዛሬ ትመጣለህ") == "ዛሬ ትመጣለህ።"


def test_documented_number_false_positives():
    assert numbers_to_digits("አርባ ምንጭ") == "40 ምንጭ"
    assert numbers_to_digits("በአንድ ላይ") == "በ1 ላይ"
    assert numbers_to_digits("ከመቶ ሃምሳ") == "ከ150"


def test_word_list_wanted_text_still_goes_through_later_steps():
    words = {"አርባ ምንጭ": "አርባ ምንጭ", "ዓ.ም": "ዓ.ም"}
    cfg = Config(merge_homophones=True, numbers_to_digits=True)
    assert process("አርባ ምንጭ በሁለት ሺህ ዓ.ም", cfg, words) == "40 ምንጭ በ2000 አ.ም"
    assert process("አርባ ምንጭ", Config(), words) == "አርባ ምንጭ"


def test_punctuation_runs_after_numbers():
    cfg = Config(numbers_to_digits=True, punctuation=True)
    assert process("ዋጋው ሃያ ብር ነው", cfg) == "ዋጋው 20 ብር ነው።"
    assert process("ስንት ሰዓት ነው", cfg) == "ስንት ሰዓት ነው?"
    assert process("   ", cfg) == ""
