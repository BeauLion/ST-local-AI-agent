"""
Tests for calendar_manager.py's _strip_unsafe_text - the emoji/symbol
filter applied to titles/locations/descriptions before staging a
calendar write. Exists because iCloud's CalDAV server has been observed
to hang (not fail cleanly) on certain 4-byte UTF-8 characters - see the
module-level comment above _UNSAFE_TEXT_PATTERN for the full story.

Pure function, no CalDAV involved - no fake_calendars fixture needed.
"""
import calendar_manager as cm


def test_removes_simple_pictograph_emoji_and_collapses_the_gap():
    assert cm._strip_unsafe_text("Hello 🎉 World") == "Hello World"


def test_removes_flag_emoji_regional_indicators():
    assert cm._strip_unsafe_text("Trip to Japan 🇯🇵 soon") == "Trip to Japan soon"


def test_removes_zwj_emoji_sequences_entirely():
    """A ZWJ (zero-width joiner) sequence like a family emoji is several
    codepoints joined together (person + ZWJ + person + ZWJ + person).
    Each individual codepoint falls in a stripped range, so the whole
    sequence should disappear, not leave stray joiner characters behind."""
    assert cm._strip_unsafe_text("family 👨‍👩‍👧 time") == "family time"


def test_removes_symbol_with_variation_selector():
    """A heart glyph followed by U+FE0F (variation selector-16, which
    requests the colorful emoji presentation) - both codepoints are in
    ranges the filter strips, individually and together."""
    assert cm._strip_unsafe_text("I love this ❤️ so much") == "I love this so much"


def test_leaves_plain_text_completely_unchanged():
    assert cm._strip_unsafe_text("Plain text, no emoji.") == "Plain text, no emoji."


def test_leaves_ordinary_punctuation_and_symbols_unchanged():
    """Only the specific emoji/symbol Unicode blocks are targeted - normal
    ASCII punctuation like &, %, #, @ must survive untouched."""
    assert cm._strip_unsafe_text("50% off @ Sam's Deli #treat") == "50% off @ Sam's Deli #treat"


def test_title_that_is_only_emoji_becomes_empty_string():
    """This is the case stage_create_event specifically checks for and
    gives its own error message about - a title of nothing but emoji
    strips down to an empty string, not None."""
    assert cm._strip_unsafe_text("🎉🎉🎉") == ""


def test_none_input_returns_none_unchanged():
    assert cm._strip_unsafe_text(None) is None


def test_empty_string_returns_empty_string():
    assert cm._strip_unsafe_text("") == ""


def test_whitespace_only_string_becomes_empty_after_strip():
    assert cm._strip_unsafe_text("   ") == ""


def test_strips_leading_and_trailing_whitespace_after_cleanup():
    assert cm._strip_unsafe_text("  leading and trailing  ") == "leading and trailing"


def test_does_not_collapse_single_intentional_spaces():
    """The double-space collapse (` {2,}` -> ` `) shouldn't touch normal
    single-space word separation - only gaps left behind by removed
    emoji."""
    assert cm._strip_unsafe_text("One Two Three") == "One Two Three"
