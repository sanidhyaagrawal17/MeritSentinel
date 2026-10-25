"""Adversarial injection detectors, unicode homoglyph normalization, and word-to-number normalizers."""

import re
import unicodedata

# Unicode homoglyphs mapping (Cyrillic, Greek, fullwidth, lookalikes to ASCII)
HOMOGLYPH_MAP = {
    # Cyrillic lowercase lookalikes
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x', '\u0455': 's',
    '\u0456': 'i', '\u0458': 'j', '\u04bb': 'h',
    # Cyrillic uppercase lookalikes
    '\u0410': 'A', '\u0412': 'B', '\u0421': 'C', '\u0415': 'E',
    '\u041d': 'H', '\u0406': 'I', '\u0408': 'J', '\u041a': 'K',
    '\u041c': 'M', '\u041e': 'O', '\u0420': 'P', '\u0422': 'T',
    '\u0425': 'X', '\u0423': 'Y',
    # Greek lookalikes
    '\u0391': 'A', '\u0392': 'B', '\u0395': 'E', '\u0396': 'Z',
    '\u0397': 'H', '\u0399': 'I', '\u039a': 'K', '\u039c': 'M',
    '\u039d': 'N', '\u039f': 'O', '\u03a1': 'P', '\u03a4': 'T',
    '\u03a5': 'Y', '\u03a7': 'X', '\u03bf': 'o', '\u03bd': 'v',
}

SMALL_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}

MAGNITUDES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
}

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|rules|prompts)",
    r"add\s+([a-zA-Z0-9_\-\s]+)\s+(experience|skill|project)",
    r"disregard\s+(the\s+)?(above|previous|system)",
    r"mark\s+(all\s+)?(claims?|statements?)\s+(as\s+)?(verified|valid|true)",
    r"system\s*:\s*you\s+are",
    r"bypass\s+(verification|security|safety)",
    r"admin\s+override",
]


def normalize_unicode_homoglyphs(text: str) -> str:
    """Normalizes Unicode homoglyphs, compatibility characters to NFKC, and strips invisible format characters."""
    # Strip invisible zero-width and format control characters (e.g. zero-width space \u200B, ZWNJ, ZWJ, soft hyphen)
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF\u00AD\u2060\u200E\u200F\u202A-\u202E\u2066-\u2069]", "", text)
    # Apply NFKC decomposition
    normalized = unicodedata.normalize("NFKC", cleaned)
    # Then replace mapped lookalikes
    chars = []
    for char in normalized:
        chars.append(HOMOGLYPH_MAP.get(char, char))
    return "".join(chars)


def parse_word_number(phrase: str) -> int:
    """Parses a written English number phrase (e.g. 'forty five thousand' or 'forty five') to an integer."""
    tokens = phrase.lower().replace("-", " ").split()
    total = 0
    current = 0
    for t in tokens:
        if t in SMALL_NUMBERS:
            current += SMALL_NUMBERS[t]
        elif t in MAGNITUDES:
            mult = MAGNITUDES[t]
            if current == 0:
                current = 1
            if mult >= 1000:
                total += current * mult
                current = 0
            else:
                current *= mult
        else:
            return 0
    return total + current


def normalize_words_to_numbers(text: str) -> str:
    """Detects and replaces written English numbers with their numeric digit representation."""
    norm = normalize_unicode_homoglyphs(text)

    # All number words pattern
    num_word_pat = r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s\-]+(?:one|two|three|four|five|six|seven|eight|nine)|(?:" + "|".join(list(SMALL_NUMBERS.keys())) + r"))"

    # Convert patterns like "forty percent" or "forty-five percent" -> "45%"
    pct_pattern = re.compile(
        rf"\b({num_word_pat})\s+percent\b",
        re.IGNORECASE
    )
    def repl_pct(match):
        val = parse_word_number(match.group(1))
        return f"{val}%"
    norm = pct_pattern.sub(repl_pct, norm)

    # Multi-word compound numbers: e.g. "forty thousand", "two million", "three hundred", "forty-five thousand"
    comp_pattern = re.compile(
        r"\b((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s\-]+)+(?:hundred|thousand|million|billion))\b",
        re.IGNORECASE
    )
    def repl_comp(match):
        val = parse_word_number(match.group(1))
        return str(val)
    norm = comp_pattern.sub(repl_comp, norm)

    # Single or two-word numbers followed by units or standalone
    unit_pattern = re.compile(
        rf"\b({num_word_pat})(?=\s+(?:nodes?|connections?|req|req/sec|reqs/sec|requests?|years?|users?)\b)",
        re.IGNORECASE
    )
    def repl_unit(match):
        val = parse_word_number(match.group(1))
        return str(val)
    norm = unit_pattern.sub(repl_unit, norm)

    return norm


class AdversarialDetector:
    """Detects prompt injections, hostile instructions, and obfuscated attacks in inputs."""

    def __init__(self):
        self.injection_regexes = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]

    def inspect(self, text: str) -> tuple[bool, list[str]]:
        """Inspects text for adversarial injections and returns (is_adversarial, detected_reasons)."""
        clean_text = normalize_unicode_homoglyphs(text)
        reasons = []

        for reg in self.injection_regexes:
            m = reg.search(clean_text)
            if m:
                reasons.append(f"Prompt injection pattern detected: '{m.group(0)}'")

        return len(reasons) > 0, reasons
