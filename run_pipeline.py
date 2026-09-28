"""IPC sentiment strategy, end to end.

    python run_pipeline.py                     # download 2 years of data and run everything
    python run_pipeline.py --sentiment vader   # lighter scorer (English headlines only)
    python run_pipeline.py --model mlp         # PyTorch net instead of ridge regression
    python run_pipeline.py --headlines-csv my_tweets.csv   # use your own tweets/headlines
    python run_pipeline.py --demo              # synthetic data, no internet (tests the code)

Outputs go to `output/`: headlines, the weekly sentiment matrix, weekly predictions
and decisions, and the performance summary. Run it on Monday after the market close;
the last line printed is the decision for the week starting that Monday.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipc_sentiment import backtest, model, news, prices, sentiment
from ipc_sentiment.config import IPC_TICKER, SUBJECTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=float, default=2.0, help="history to download")
    p.add_argument("--end", default=None, help="end date (default: today)")
    p.add_argument("--langs", default="es,en", help="headline languages to download")
    p.add_argument("--headlines-csv", default=None, help="your own data: columns date,subject,text[,lang]")
    p.add_argument("--sentiment", choices=["xlmr", "vader"], default="xlmr")
    p.add_argument("--model", choices=["ridge", "mlp"], default="ridge")
    p.add_argument("--lags", type=int, default=1, help="weeks of sentiment history used as input")
    p.add_argument("--volume", action="store_true", help="also use news volume per subject as input")
    p.add_argument("--changes", action="store_true", help="also use week-over-week sentiment change")
    p.add_argument("--min-train", type=int, default=52, help="weeks used before the first prediction")
    p.add_argument("--threshold", type=float, default=0.002, help="min |predicted weekly return| to act")
    p.add_argument("--neutral", choices=["keep", "flat"], default="keep", help="meaning of HOLD")
    p.add_argument("--cost-bps", type=float, default=10, help="trading cost per unit turnover")
    p.add_argument("--cash-rate", type=float, default=0.0, help="annual rate on cash, e.g. 0.09 (CETES)")
    p.add_argument("--out", default="output")
    p.add_argument("--demo", action="store_true", help="run on synthetic data")
    return p.parse_args()


def synthetic_data(start, end, seed=0, strength=0.01):
    """Fake headlines + prices, with a weak planted link from sentiment to next-week return."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    weeks = pd.date_range(days[0].to_period("W-SUN").start_time, days[-1], freq="W-MON")
    tone = pd.DataFrame(rng.normal(0, 0.3, (len(weeks), len(SUBJECTS))), index=weeks, columns=list(SUBJECTS))
    rows = []
    for w in weeks:
        for s in SUBJECTS:
            for _ in range(rng.integers(5, 40)):
                rows.append({"date": w + pd.Timedelta(days=int(rng.integers(0, 7))), "subject": s,
                             "lang": "es", "text": "", "sentiment": np.clip(tone.at[w, s] + rng.normal(0, 0.4), -1, 1)})
    signal = strength * tone[["banxico_rates", "peso_fx", "us_mx_trade"]].mean(axis=1).shift(1).fillna(0)
    daily_signal = signal.reindex(days, method="ffill") / 5
    daily_ret = daily_signal + rng.normal(0.0003, 0.009, len(days))
    close = pd.Series(50000 * np.cumprod(1 + daily_ret), index=days, name="close")
    return pd.DataFrame(rows), close


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(days=int(args.years * 365))

    # 1) headlines + prices
    if args.demo:
        print("DEMO MODE: synthetic data, results say nothing about the real IPC.")
        scored, close = synthetic_data(start, end)
    else:
        if args.headlines_csv:
            headlines = news.load_headlines_csv(args.headlines_csv)
        else:
            print(f"Downloading headlines {start:%Y-%m-%d} -> {end:%Y-%m-%d} ...")
            headlines = news.download_headlines(SUBJECTS, start, end, langs=args.langs.split(","))
        headlines.to_csv(out / "headlines.csv", index=False)
        print(f"{len(headlines)} headlines. Scoring sentiment with {args.sentiment} ...")
        # 2) sentiment per headline
        scored = sentiment.score_headlines(headlines, args.sentiment)
        scored.to_csv(out / "headlines_scored.csv", index=False)
        close = prices.download_ipc(IPC_TICKER, start - pd.Timedelta(days=7), end + pd.Timedelta(days=1))

    # 2) weekly subject x sentiment matrix
    matrix = sentiment.weekly_matrix(scored)
    matrix.to_csv(out / "sentiment_matrix.csv")
    returns = prices.weekly_returns(close)

    # 3) walk-forward model: sentiment up to t-1 -> return of week t
    X, y, X_next = model.build_features(matrix, returns, lags=args.lags, use_volume=args.volume, use_changes=args.changes)
    print(f"{len(y)} weeks x {X.shape[1]} features; first {args.min_train} weeks used only for training.")
    pred = model.walk_forward(X, y, model=args.model, min_train=args.min_train)

    # 4) Monday decisions + backtest
    bt, summary, extra = backtest.evaluate(pred, returns, args.threshold, args.neutral, args.cost_bps, args.cash_rate)
    bt.to_csv(out / "weekly_decisions.csv")
    summary.to_csv(out / "summary.csv")

    final = model.MODELS[args.model]().fit(X.values, y.values)
    next_pred = pd.Series(final.predict(X_next.values), index=X_next.index)
    last_pos = bt["position"].iloc[-1] if len(bt) else 0.0
    next_raw = backtest.decisions(next_pred, args.threshold, "flat")[1].iloc[0]
    if next_raw == "HOLD" and args.neutral == "keep":
        next_action = f"HOLD (keep {'LONG' if last_pos > 0 else 'SHORT' if last_pos < 0 else 'no position'})"
    else:
        next_action = next_raw

    pd.set_option("display.float_format", "{:.4f}".format)
    print("\nOut-of-sample performance")
    print(summary)
    print(json.dumps(extra, indent=2))
    print("\nLast 8 weeks:")
    print(bt[["pred", "action", "position", "ret", "pnl"]].tail(8))
    print(f"\nDecision for week of {X_next.index[0]:%Y-%m-%d}: {next_action} "
          f"(predicted return {next_pred.iloc[0]:+.2%})")
    with open(out / "next_decision.json", "w") as f:
        json.dump({"week": f"{X_next.index[0]:%Y-%m-%d}", "predicted_return": float(next_pred.iloc[0]),
                   "action": next_action, **extra}, f, indent=2)


if __name__ == "__main__":
    main()
