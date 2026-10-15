import re
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

def filter_reviews(df):
    df = df.copy()

    def keep_review(text):
        if not isinstance(text, str):
            return False

        text = text.strip()

        if not text:
            return False

        words = re.findall(r"\b[\w'-]+\b", text)

        # Keep short reviews because language detection is unreliable
        # for things like "Yum!!" and dish names
        if len(words) < 15:
            return True

        # Remove long repeated-character sequences before detection
        cleaned = re.sub(r"(.)\1{4,}", r"\1\1", text)

        # Remove repeated consecutive words
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

        text_words = {word.lower() for word in words}
        english_matches = len(text_words & english_words)

        try:
            language = detect(cleaned)

            # Protect English reviews containing foreign dish names
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