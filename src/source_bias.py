import json
import os

BIAS_FILE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "source_bias.json"
)

DEFAULT_BIAS = 0.0


def load_source_bias():
    if not os.path.exists(BIAS_FILE):
        return {}

    with open(BIAS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


SOURCE_BIAS = load_source_bias()


def get_source_bias(source_name: str) -> float:
    """
    Returns bias score of a news source.
    Range:
        -1 → Supportive leaning
         0 → Neutral
        +1 → Critical leaning
    """
    if not source_name:
        return DEFAULT_BIAS

    return SOURCE_BIAS.get(
        source_name.strip(),
        DEFAULT_BIAS
    )


def compute_weighted_bias(article_bias, source_name):
    source_bias = get_source_bias(source_name)

    return (
        article_bias * 0.7 +
        source_bias * 0.3
    )