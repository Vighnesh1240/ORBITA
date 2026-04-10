import json
import os
from datetime import datetime

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "bias_timeline.json"
)


def save_bias_entry(topic, bias_score):
    """
    Save bias score with timestamp.
    """

    entry = {
        "topic": topic,
        "bias": float(bias_score),
        "timestamp": datetime.now().isoformat()
    }

    try:

        if os.path.exists(DATA_FILE):

            with open(DATA_FILE, "r") as f:
                data = json.load(f)

        else:
            data = []

        data.append(entry)

        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

        print(
            f"[timeline] Saved bias {bias_score:+.3f}"
        )

    except Exception as e:

        print(
            f"[timeline] Error saving timeline: {e}"
        )


def load_bias_history(topic=None):
    """
    Load saved timeline.
    """

    if not os.path.exists(DATA_FILE):
        return []

    try:

        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        if topic:

            data = [
                d for d in data
                if d["topic"] == topic
            ]

        return data

    except:

        return []