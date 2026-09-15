"""One-time script: fetch posters + descriptions from TMDB and save them
to model/checkpoints/posters.parquet.

Run this ONCE on your own computer (not inside Streamlit, not on deploy):

    pip install requests pandas pyarrow python-dotenv --break-system-packages
    python model/enrich_posters.py

Requirements before running:
1. A file named `.env` in the SAME FOLDER as this script (model/), containing:
       TMDB_API_KEY=your_key_here
   (never commit this .env file — it should be in .gitignore)
2. `links.csv` from MovieLens (movieId, imdbId, tmdbId columns) — this script
   looks for it at datasets/raw/links.csv by default; pass a different path
   with --links if yours lives somewhere else.

What it does:
- Reads links.csv to get the tmdbId for every movieId in your dataset.
- For each tmdbId, calls TMDB's /movie/{id} endpoint once to get the poster
  path and the description (overview).
- Saves everything into model/checkpoints/posters.parquet with columns:
    movieId, poster_url, overview
- Is resumable: if the script is interrupted (or you re-run it), it skips
  movies it already fetched successfully, so you never lose progress or
  waste API calls re-fetching what you already have.
- Respects TMDB's rate limit with a small delay between requests. For
  ~7,000 movies this takes roughly 30-40 minutes — that's expected, just
  let it run in the background.

This script never touches datasets/ or model/checkpoints/model.npz —
it only reads links.csv and writes a new, separate posters.parquet file.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
REQUEST_DELAY_SECONDS = 0.28  # ~3-4 requests/sec, comfortably under TMDB's limit

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LINKS_PATH = SCRIPT_DIR.parent / "datasets" / "raw" / "links.csv"
OUTPUT_PATH = SCRIPT_DIR / "checkpoints" / "posters.parquet"


def load_api_key() -> str:
    load_dotenv(SCRIPT_DIR / ".env")
    key = os.getenv("TMDB_API_KEY")
    if not key:
        raise SystemExit(
            "TMDB_API_KEY not found. Create a file named '.env' next to this "
            "script (model/.env) containing:\n\n    TMDB_API_KEY=your_key_here\n"
        )
    return key


def fetch_one(tmdb_id: int, api_key: str) -> dict | None:
    """One API call for one movie. Returns None on any failure (missing
    movie, network hiccup, etc.) so the caller can just skip it."""
    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{int(tmdb_id)}",
            params={"api_key": api_key},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        poster_path = data.get("poster_path")
        return {
            "poster_url": f"{POSTER_BASE}{poster_path}" if poster_path else None,
            "overview": data.get("overview") or None,
        }
    except requests.RequestException:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", default=str(DEFAULT_LINKS_PATH),
                         help="Path to MovieLens links.csv")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N movies (for a quick test run)")
    args = parser.parse_args()

    api_key = load_api_key()

    links_path = Path(args.links)
    if not links_path.exists():
        raise SystemExit(f"Can't find links.csv at {links_path}. "
                          f"Pass the correct path with --links /path/to/links.csv")

    links = pd.read_csv(links_path)
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    if args.limit:
        links = links.head(args.limit)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load whatever we already fetched in a previous run.
    if OUTPUT_PATH.exists():
        done = pd.read_parquet(OUTPUT_PATH)
        done_ids = set(done["movieId"])
        print(f"Found existing posters.parquet with {len(done)} movies already done — resuming.")
    else:
        done = pd.DataFrame(columns=["movieId", "poster_url", "overview"])
        done_ids = set()

    todo = links[~links["movieId"].isin(done_ids)]
    total = len(todo)
    print(f"{len(done_ids)} already fetched, {total} left to fetch.")

    new_rows = []
    save_every = 100  # write progress to disk periodically, not just at the very end

    for i, row in enumerate(todo.itertuples(index=False), start=1):
        result = fetch_one(row.tmdbId, api_key)
        new_rows.append({
            "movieId": row.movieId,
            "poster_url": result["poster_url"] if result else None,
            "overview": result["overview"] if result else None,
        })

        if i % 20 == 0 or i == total:
            print(f"  {i}/{total} done…")

        if i % save_every == 0:
            combined = pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_parquet(OUTPUT_PATH, index=False)
            done = combined
            new_rows = []
            done_ids = set(done["movieId"])

        time.sleep(REQUEST_DELAY_SECONDS)

    if new_rows:
        combined = pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_parquet(OUTPUT_PATH, index=False)
        done = combined

    n_with_poster = done["poster_url"].notna().sum()
    print(f"\nDone. {len(done)} movies total, {n_with_poster} have a poster "
          f"({n_with_poster / len(done):.0%}).")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
