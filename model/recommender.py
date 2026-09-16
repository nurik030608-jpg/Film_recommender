"""Serving-side recommender: loads the exported checkpoint. No training dependencies.

Checkpoint layout (created by train_and_export.py):
    checkpoint/model.npz        FunkSVD parameters + genre matrix + user profiles
    checkpoint/movies.parquet   catalog (movieId, title, genres) aligned to model columns
    checkpoint/ratings.parquet  subsample ratings (userId, movieId, rating)
    checkpoint/meta.json        hyperparameters + evaluation numbers
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class Recommender:
    def __init__(self, checkpoint_dir=None):
        d = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).parent / "checkpoints"
        z = np.load(d / "model.npz")
        self.P, self.Q = z["P"], z["Q"]
        self.bu, self.bi = z["bu"], z["bi"]
        self.mu = float(z["mu"])
        self.user_ids = z["user_ids"]              # row order of P / bu / profiles
        self.item_ids = z["item_ids"]              # column order of Q / bi / Gd / movies
        self.Gd = z["Gd_normed"]                   # (n_items, n_genres), L2-normalized rows
        self.profiles = z["profiles"]              # (n_users, n_genres), L2-normalized rows
        self.meta = json.loads((d / "meta.json").read_text())
        self.movies = pd.read_parquet(d / "movies.parquet")
        self.ratings = pd.read_parquet(d / "ratings.parquet")

        # Optional: TMDB poster/overview enrichment (model/enrich_posters.py).
        # Missing file is fine — poster_url/overview just stay all-null and
        # the frontend falls back to its gradient tiles.
        posters_path = d / "posters.parquet"
        if posters_path.exists():
            posters = pd.read_parquet(posters_path)[["movieId", "poster_url", "overview"]]
            self.movies = self.movies.merge(posters, on="movieId", how="left")
        else:
            self.movies["poster_url"] = None
            self.movies["overview"] = None

        self.user_pos = {int(u): k for k, u in enumerate(self.user_ids)}
        self.item_pos = {int(m): k for k, m in enumerate(self.item_ids)}
        self.title_to_pos = {t: k for k, t in enumerate(self.movies["title"])}

        self.num_ratings = np.zeros(len(self.item_ids), dtype=np.int64)
        vc = self.ratings["movieId"].value_counts()
        self.num_ratings[[self.item_pos[m] for m in vc.index]] = vc.to_numpy()
        self.seen = self.ratings.groupby("userId")["movieId"].apply(set).to_dict()

    # ---------- internal scoring ----------
    def _cf_row(self, row):
        """Predicted ratings for one user over the whole catalog."""
        return np.clip(self.mu + self.bu[row] + self.bi + self.P[row] @ self.Q.T, 0.5, 5.0)

    @staticmethod
    def _minmax(x):
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x, dtype=float)

    def _frame(self, order, cols):
        out = self.movies.iloc[order][["movieId", "title", "genres", "poster_url", "overview"]].reset_index(drop=True)
        for name, values in cols.items():
            out[name] = np.round(np.asarray(values, dtype=float)[order], 3)
        out["num_ratings"] = self.num_ratings[order]
        return out

    # ---------- public API ----------
    def users_by_activity(self):
        """User ids sorted by number of ratings (descending)."""
        counts = self.ratings["userId"].value_counts()
        return counts.index.to_numpy(), counts.to_numpy()

    def user_history(self, user_id, n=10):
        h = self.ratings.loc[self.ratings["userId"] == user_id].merge(self.movies, on="movieId")
        return (h.sort_values("rating", ascending=False)
                 .head(n)[["title", "genres", "rating", "poster_url", "overview"]].reset_index(drop=True))

    def recommend(self, user_id, alpha=None, n=10):
        """Hybrid top-N for a known user. alpha: 1 = pure CF, 0 = pure content."""
        if user_id not in self.user_pos:
            raise KeyError(f"user {user_id} not in the trained model (cold start) — "
                           f"use recommend_for_new_user() or popular().")
        if alpha is None:
            alpha = self.meta["best_alpha"]
        row = self.user_pos[user_id]
        cf = self._minmax(self._cf_row(row))
        ct = self._minmax(self.Gd @ self.profiles[row])
        score = alpha * cf + (1 - alpha) * ct
        s = score.copy()
        s[[self.item_pos[m] for m in self.seen.get(user_id, ()) if m in self.item_pos]] = -np.inf
        order = np.argsort(-s)[:n]
        return self._frame(order, {"cf": cf, "content": ct, "hybrid": score})

    def similar_movies(self, title, n=10):
        """Genre-similar movies; ties broken by popularity; the query itself excluded."""
        if title not in self.title_to_pos:
            raise KeyError(f"title not found: {title!r}")
        pos = self.title_to_pos[title]
        sims = (self.Gd @ self.Gd[pos]).astype(float)
        s = sims.copy()
        s[pos] = -np.inf
        order = np.lexsort((-self.num_ratings, -s))[:n]     # sort by sim, then popularity
        return self._frame(order, {"similarity": sims})

    def recommend_for_new_user(self, liked_movie_ids, n=10):
        """Cold start: build a genre profile from a few liked movies (content-only)."""
        pos = [self.item_pos[m] for m in liked_movie_ids if m in self.item_pos]
        if not pos:
            return self.popular(n)
        profile = self.Gd[pos].sum(axis=0)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        ct = (self.Gd @ profile).astype(float)
        s = ct.copy()
        s[pos] = -np.inf
        order = np.lexsort((-self.num_ratings, -s))[:n]
        return self._frame(order, {"content": self._minmax(ct)})

    def popular(self, n=10):
        order = np.argsort(-self.num_ratings)[:n]
        return self._frame(order, {})
