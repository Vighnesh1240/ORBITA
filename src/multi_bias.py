def emotional_bias(sentiment_score):
    return abs(sentiment_score)


def informational_bias(fact_density):
    return 1 - fact_density


def ideological_bias(bias_score):
    return abs(bias_score)


def compute_multi_bias(
    sentiment,
    fact_density,
    ideology
):
    return {
        "emotional": emotional_bias(
            sentiment
        ),
        "informational": informational_bias(
            fact_density
        ),
        "ideological": ideological_bias(
            ideology
        )
    }