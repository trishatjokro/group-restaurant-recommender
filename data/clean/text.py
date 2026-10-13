import re
import html
import unicodedata
import pandas as pd

def text_cleaning(df):
    df = df.copy()
    cols = ["text"]

    # replace unicode characters with ASCII
    replacements = {
        "\u2018": "'", 
        "\u2019": "'", 
        "\u201c": '"', 
        "\u201d": '"',
        "\u2013": "-", 
        "\u2014": "-", 
        "\u2026": "...", 
        "\u00a0": " ",
        "\u2122": "", 
        "\u00ae": "", 
        "\u00a9": "",
    }

    def clean(text):
        if not isinstance(text, str):
            return text

        # clean <br> tags and "\n" literals from HTML
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("\\n", " ")

        # replace weird unicode forms
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = unicodedata.normalize("NFKC", text)

        # remove emojis, symbols, and extraneous characters
        text = "".join(
            ch for ch in text
            if ch in "\n\t" or unicodedata.category(ch) not in
            ("Cc", "Cf", "Co", "Cn", "So", "Sk")
        )
        text = re.sub(r"[\ufe00-\ufe0f]", "", text)

        # reduce stretched words to just double letters
        text = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)

        # turn extra line breaks and spaces into a single space
        text = re.sub(r"\s+", " ", text).strip()

        return text

    changed = pd.Series(False, index=df.index)
    # currently in a for loop in case more columns need to be cleaned
    for col in cols:
        original = df[col]
        cleaned = original.map(clean)
        is_text = original.map(lambda x: isinstance(x, str))
        changed |= is_text & (original != cleaned)
        df[col] = cleaned

    print(f"text_cleaning: removed/fixed {int(changed.sum())} rows")
    return df
