"""Step 4: turn predictions into Monday decisions (LONG / HOLD / SHORT) and backtest them."""

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


def decisions(pred, threshold=0.002, neutral="keep"):
    """LONG if pred > threshold, SHORT if pred < -threshold, otherwise HOLD.

    neutral="keep": HOLD means keep last week's position (fewer trades).
    neutral="flat": HOLD means stay out of the market (cash).
    """
    raw = np.where(pred > threshold, 1.0, np.where(pred < -threshold, -1.0, np.nan))
    pos = pd.Series(raw, index=pred.index)
    pos = pos.ffill().fillna(0.0) if neutral == "keep" else pos.fillna(0.0)
    action = pd.Series(np.select([raw == 1, raw == -1], ["LONG", "SHORT"], "HOLD"), index=pred.index)
    return pos.rename("position"), action.rename("action")


def backtest(position, returns, cost_bps=10, cash_rate=0.0):
    """Weekly P&L. `cost_bps` is charged per unit of position change (1 -> -1 costs twice).

    `cash_rate` is the annual rate earned on cash when flat (e.g. ~0.09 for CETES).
    """
    ret = returns.reindex(position.index)
    turnover = position.diff().abs().fillna(position.abs())
    cash = (1 + cash_rate) ** (1 / WEEKS_PER_YEAR) - 1
    pnl = position * ret + (1 - position.abs()) * cash - turnover * cost_bps / 1e4
    return pd.DataFrame({"ret": ret, "position": position, "turnover": turnover, "pnl": pnl})


def stats(pnl):
    pnl = pnl.dropna()
    equity = (1 + pnl).cumprod()
    years = len(pnl) / WEEKS_PER_YEAR
    vol = pnl.std() * np.sqrt(WEEKS_PER_YEAR)
    return {
        "weeks": len(pnl),
        "total_return": equity.iloc[-1] - 1,
        "annual_return": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "annual_vol": vol,
        "sharpe": pnl.mean() * WEEKS_PER_YEAR / vol if vol > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "hit_rate": (pnl[pnl != 0] > 0).mean(),
    }


def permutation_pvalue(position, returns, n=2000, seed=0):
    """How often does a random reshuffle of the same positions do as well?

    A p-value above ~0.05 means the result is indistinguishable from luck. Returns NaN
    when the position never changed (the strategy did no timing, so there is nothing to test).
    """
    if position.nunique() < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    ret = returns.reindex(position.index).values
    pos = position.values
    actual = np.mean(pos * ret)
    sims = np.array([np.mean(rng.permutation(pos) * ret) for _ in range(n)])
    return float((sims >= actual).mean())


def evaluate(pred, returns, threshold=0.002, neutral="keep", cost_bps=10, cash_rate=0.0):
    position, action = decisions(pred, threshold, neutral)
    bt = backtest(position, returns, cost_bps, cash_rate)
    bt["pred"], bt["action"] = pred, action
    buy_hold = backtest(pd.Series(1.0, index=position.index), returns, cost_bps=0, cash_rate=cash_rate)
    summary = pd.DataFrame({"strategy": stats(bt["pnl"]), "buy_and_hold": stats(buy_hold["pnl"])})
    extra = {
        "direction_accuracy": float((np.sign(pred) == np.sign(bt["ret"])).mean()),
        "pred_vs_actual_corr": float(np.corrcoef(pred, bt["ret"])[0, 1]),
        "permutation_p_value": permutation_pvalue(position, returns),
    }
    return bt, summary, extra
