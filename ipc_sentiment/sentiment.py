"""Step 2: score each headline and build the weekly subject x sentiment matrix."""

import numpy as np
import pandas as pd

XLMR_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # multilingual (es + en)


def score_vader(texts):
    """Fast, English-only lexicon scorer. Spanish text scores ~0, so use `xlmr` for Spanish."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])


def score_xlmr(texts, batch_size=64):
    """Multilingual transformer: score = P(positive) - P(negative), in [-1, 1]."""
    from transformers import pipeline

    clf = pipeline("sentiment-analysis", model=XLMR_MODEL, top_k=None, truncation=True)
    scores = []
    for out in clf(list(texts), batch_size=batch_size):
        probs = {d["label"].lower(): d["score"] for d in out}
        scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
    return np.array(scores)


SCORERS = {"vader": score_vader, "xlmr": score_xlmr}


def score_headlines(df, method="xlmr"):
    df = df.copy()
    if method == "vader":
        df = df[df["lang"] == "en"]  # VADER cannot read Spanish
    df["sentiment"] = SCORERS[method](df["text"].astype(str).tolist())
    return df


def weekly_matrix(scored):
    """Rows = week (Monday), columns = `<subject>_sent` (mean score) and `<subject>_n` (volume).

    Headlines from Monday to Sunday of week w are assigned to week w. The model uses
    week w's row to predict week w+1's return, so everything published before the
    Monday decision is available, and nothing after it.
    """
    df = scored.copy()
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W-SUN").dt.start_time
    g = df.groupby(["week", "subject"])["sentiment"]
    sent = g.mean().unstack()
    count = g.size().unstack().fillna(0)
    sent.columns = [f"{c}_sent" for c in sent.columns]
    count.columns = [f"{c}_n" for c in count.columns]
    matrix = pd.concat([sent, count], axis=1).sort_index()
    # Weeks with no headlines for a subject: carry the last known sentiment forward.
    sent_cols = [c for c in matrix.columns if c.endswith("_sent")]
    matrix[sent_cols] = matrix[sent_cols].ffill().fillna(0.0)
    return matrix
