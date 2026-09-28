"""Step 1: download headlines per subject, one query per subject per week.

Source: Google News RSS search, which is free, needs no API key, and accepts
`after:`/`before:` date operators, so we can walk back week by week over two years.
Each query returns up to ~100 headlines.

Tweets: X/Twitter's historical search requires a paid API tier. If you have tweets
(or headlines from any other source), put them in a CSV with the columns
`date, subject, text` and pass it with `--headlines-csv` instead of downloading.
"""

import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

GOOGLE_NEWS = "https://news.google.com/rss/search"
LOCALES = {
    "es": {"hl": "es-419", "gl": "MX", "ceid": "MX:es-419"},
    "en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
}


def fetch_week(query, start, end, lang, session, retries=3):
    """Headlines for `query` published in [start, end). Returns a list of dicts."""
    params = {
        "q": f"{query} after:{start:%Y-%m-%d} before:{end:%Y-%m-%d}",
        **LOCALES[lang],
    }
    for attempt in range(retries):
        try:
            resp = session.get(GOOGLE_NEWS, params=params, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))

    rows = []
    for item in ET.fromstring(resp.content).iter("item"):
        title = item.findtext("title") or ""
        source = item.findtext("source") or ""
        # Google appends " - Source" to titles; drop it so it doesn't bias sentiment.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        pub = item.findtext("pubDate")
        rows.append(
            {
                "date": parsedate_to_datetime(pub).replace(tzinfo=None) if pub else start,
                "text": title.strip(),
                "source": source,
            }
        )
    return rows


def download_headlines(subjects, start, end, langs=("es", "en"), cache_dir="data/headlines", pause=1.0):
    """Download weekly headlines for every subject between `start` and `end`.

    Results are cached per subject/language/week, so the download can be interrupted
    and resumed.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research script)"

    week_starts = pd.date_range(pd.Timestamp(start).to_period("W-SUN").start_time, end, freq="W-MON")
    frames = []
    for subject, queries in subjects.items():
        for lang in langs:
            query = queries.get(lang)
            if not query:
                continue
            for ws in week_starts:
                path = cache / f"{subject}_{lang}_{ws:%Y%m%d}.csv"
                if path.exists():
                    frames.append(pd.read_csv(path, parse_dates=["date"]))
                    continue
                we = ws + pd.Timedelta(days=7)
                rows = fetch_week(query, ws, we, lang, session)
                df = pd.DataFrame(rows, columns=["date", "text", "source"])
                df["subject"], df["lang"] = subject, lang
                df.to_csv(path, index=False)
                frames.append(df)
                print(f"  {subject:14s} {lang} {ws:%Y-%m-%d}: {len(df):3d} headlines")
                time.sleep(pause)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["text"]).drop_duplicates(subset=["subject", "text"])
    return out[["date", "subject", "lang", "source", "text"]]


def load_headlines_csv(path):
    """Load your own headlines/tweets. Required columns: date, subject, text."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = {"date", "subject", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "lang" not in df.columns:
        df["lang"] = "es"
    return df
