import re

NUMBER_PATTERN = r"\d+"
DATE_PATTERN = r"\b\d{4}\b"


def count_facts(text: str) -> int:
    """
    Counts factual indicators:
    - numbers
    - dates
    - statistics
    """

    numbers = re.findall(
        NUMBER_PATTERN,
        text
    )

    dates = re.findall(
        DATE_PATTERN,
        text
    )

    return len(numbers) + len(dates)


def compute_fact_density(text: str):
    sentences = text.split(".")

    if not sentences:
        return 0

    factual_sentences = 0

    for s in sentences:
        if count_facts(s) > 0:
            factual_sentences += 1

    return (
        factual_sentences /
        len(sentences)
    )