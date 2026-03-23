import re

def clean_text(text):
    """
    Cleans extracted PDF text
    """

    # Remove multiple newlines
    text = re.sub(r"\n+", "\n", text)

    # Remove references section (common pattern)
    text = re.split(r"References|REFERENCES", text)[0]

    # Remove citations like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()