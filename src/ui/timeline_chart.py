import pandas as pd
import plotly.express as px

from src.bias_timeline import load_bias_history


def build_bias_timeline(topic):

    history = load_bias_history(topic)

    if not history:
        return None

    df = pd.DataFrame(history)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    fig = px.line(
        df,
        x="timestamp",
        y="bias",
        markers=True,
        title=f"Bias Trend — {topic}"
    )

    fig.update_layout(
        height=350
    )

    return fig