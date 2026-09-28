# IPC sentiment strategy

Weekly long / hold / short decisions on the Mexican S&P/BMV IPC index (`^MXX`), driven by
news sentiment on the subjects that move it.

| Step | What it does | Code |
|---|---|---|
| 1 | Download two years of headlines per subject, week by week (Google News RSS, Spanish + English; no API key) | `ipc_sentiment/news.py` |
| 2 | Score each headline's sentiment and build the **week × subject** matrix | `ipc_sentiment/sentiment.py` |
| 3 | Predict week *t*'s IPC return from sentiment up to week *t−1* (ridge regression or a PyTorch MLP), trained walk-forward | `ipc_sentiment/model.py` |
| 4 | Every Monday: LONG if predicted return > threshold, SHORT if < −threshold, else HOLD; backtest with costs vs buy-and-hold | `ipc_sentiment/backtest.py` |

Subjects (edit them in `ipc_sentiment/config.py`): Banxico rates, Fed rates, peso/dollar,
Mexican inflation, US–Mexico trade/tariffs, Mexican politics/reforms, Pemex/oil,
nearshoring, Mexican economy/GDP, the BMV itself.

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py --demo              # synthetic data, checks the code works
python run_pipeline.py                     # real data: ~2,000 requests, cached in data/
python run_pipeline.py --sentiment vader   # lighter: English headlines only, no transformer download
python run_pipeline.py --model mlp         # PyTorch net instead of ridge
python run_pipeline.py --cash-rate 0.09    # earn CETES-like interest when flat
```

Run it on Monday after the close. The last printed line is that week's decision. Results
are written to `output/` (`sentiment_matrix.csv`, `weekly_decisions.csv`, `summary.csv`,
`next_decision.json`).

**Tweets:** historical search on X/Twitter needs a paid API tier. If you have tweets from
any source, save them as a CSV with columns `date,subject,text` (optionally `lang`) and run
`python run_pipeline.py --headlines-csv tweets.csv`.

**Timing:** headlines from Monday to Sunday of week *t−1* set the position taken at the
close of week *t*'s first trading day, held until the next one. Nothing published after the
decision is used.

## Reading the results honestly

- Two years is ~104 weekly returns. After 52 training weeks, only ~52 weeks are tested
  out of sample. That is too few to separate skill from luck: in the synthetic demo with
  **no** signal at all, backtest Sharpe ratios ranged from −1.9 to +1.5 across random seeds.
- Check `permutation_p_value`: it measures how often shuffling the same positions does as
  well. Above ~0.05, the result is consistent with luck.
- Keep the feature count small (default: one per subject). More lags, volume or change
  features (`--lags`, `--volume`, `--changes`) overfit fast on this little data.
- Tuning the threshold, subjects or model on the same two years, then quoting the best
  backtest, overstates the expected return.
