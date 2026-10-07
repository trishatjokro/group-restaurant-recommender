from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0

import re
import pandas as pd
import pyarrow as pa

df = pd.read_parquet("data/reviews_sample.parquet")


def filter_reviews(df):
    df = df.copy()

    def keep_review(text):
        if not isinstance(text, str):
            return False

        text = text.strip()

        if not text:
            return False

        # Count actual words
        words = re.findall(r"\b[\w'-]+\b", text)

        if len(words) < 15:
            return True

        # Collapse repeated characters
        cleaned = re.sub(r"(.)\1{4,}", r"\1\1", text)

        # Collapse repeated consecutive words
        cleaned = re.sub(
            r"\b(\w+)(?:\s+\1\b)+",
            r"\1",
            cleaned,
            flags=re.IGNORECASE
        )

        english_words = {
            "the", "and", "was", "were", "is", "are", "it", "this",
            "that", "food", "good", "great", "very", "but", "like",
            "had", "have", "overall", "decent", "much", "not"
        }

        text_words = set(word.lower() for word in words)
        english_matches = len(text_words & english_words)

        try:
            language = detect(cleaned)

            if language != "en" and english_matches >= 3:
                return True

            return language == "en"

        except LangDetectException:
            return True

    before = len(df)

    keep_mask = df["text"].apply(keep_review)
    df = df[keep_mask].copy()

    removed = before - len(df)

    print(f"filter_reviews: removed {removed} rows")

    return df


# TESTING CODE
cleaned = filter_reviews(df)

removed = df[~df.index.isin(cleaned.index)]

print("\nReviews that would be removed:")
print(removed[["text"]].to_string(index=False))