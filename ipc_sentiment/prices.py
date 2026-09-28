"""IPC prices and weekly returns aligned to Monday decisions."""

import pandas as pd


def download_ipc(ticker, start, end):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a column per ticker
        close = close.iloc[:, 0]
    return close.dropna().rename("close")


def weekly_returns(close):
    """Return of holding the IPC from one week's decision day to the next.

    The decision day is the first trading day of each week (Monday, or Tuesday after
    a holiday). We trade at that day's close, so the return for week `t` is
    close(first day of week t+1) / close(first day of week t) - 1.
    The index is the Monday of week t.
    """
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    week = close.index.to_period("W-SUN").start_time  # Monday of each date's week
    first = close.groupby(week).first()
    ret = first.shift(-1) / first - 1
    return ret.rename("ret").dropna()
