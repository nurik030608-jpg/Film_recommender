"""Glue between the trained Recommender and the CineMatch frontend (src/lib/api.ts).

Every method returns plain dicts shaped exactly like the frontend's `Movie` type:
    movieId, title, year, genres[], posterUrl, overview, avgRating, numRatings, matchScore?
Genre + decade filters are applied BEFORE ranking, so every row reacts to them.
Nothing here is random: same input -> same output.
"""
import re

import numpy as np

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
DECADES = {  # frontend Decade value -> (first year, last year); None = open-ended
    "pre1980": (None, 1979), "1980s": (1980, 1989), "1990s": (1990, 1999),
    "2000s": (2000, 2009), "2010s": (2010, 2019), "2020s": (2020, None),
}


def _text(v):
    return v.strip() if isinstance(v, str) and v.strip() else None


class Engine:
    def __init__(self, rec):
        self.rec = rec
        m = rec.movies.reset_index(drop=True)
        if len(m) != len(rec.item_ids) or not (m["movieId"].to_numpy() == rec.item_ids).all():
            raise RuntimeError(
                "movies.parquet / posters.parquet are not aligned with model.npz item order "
                "(duplicate movieIds in posters.parquet?). Fix the data before serving.")
        self.n_items = len(m)
        self.ids = rec.item_ids
        self.titles = m["title"].tolist()
        self.titles_lower = [t.lower() for t in self.titles]
        self.genres = [
            [g for g in x.split("|") if g and g != "(no genres listed)"] if isinstance(x, str) else []
            for x in m["genres"]
        ]
        self.years = np.array([int(y.group(1)) if (y := YEAR_RE.search(t)) else 0 for t in self.titles])
        self.poster = [_text(v) for v in m["poster_url"]]
        self.overview = [_text(v) for v in m["overview"]]
        self.num = rec.num_ratings
        avg = rec.ratings.groupby("movieId")["rating"].mean().reindex(self.ids).fillna(0.0)
        self.avg = avg.to_numpy(dtype=float)
        all_genres = sorted({g for gl in self.genres for g in gl})
        self.genre_mask = {g: np.array([g in gl for gl in self.genres]) for g in all_genres}

    # ---------------- helpers ----------------
    def mask(self, genres, decade):
        ok = np.ones(self.n_items, dtype=bool)
        if genres:
            any_genre = np.zeros(self.n_items, dtype=bool)
            for g in genres:
                if g in self.genre_mask:
                    any_genre |= self.genre_mask[g]
            ok &= any_genre
        if decade in DECADES:
            lo, hi = DECADES[decade]
            ok &= self.years > 0
            if lo is not None:
                ok &= self.years >= lo
            if hi is not None:
                ok &= self.years <= hi
        return ok

    def _top(self, score, valid, n):
        """Indices of the n best valid items: by score, ties broken by popularity."""
        s = np.where(valid, score, -np.inf)
        order = np.lexsort((-self.num, -s))[:n]
        return order[np.isfinite(s[order])]

    def _pack(self, order, match=None):
        out = []
        for i in order:
            d = {
                "movieId": int(self.ids[i]), "title": self.titles[i], "year": int(self.years[i]),
                "genres": self.genres[i], "posterUrl": self.poster[i], "overview": self.overview[i],
                "avgRating": round(float(self.avg[i]), 2), "numRatings": int(self.num[i]),
            }
            if match is not None:
                d["matchScore"] = round(float(np.nan_to_num(match[i])), 2)
            out.append(d)
        return out

    def _positions(self, movie_ids):
        return [self.rec.item_pos[m] for m in movie_ids if m in self.rec.item_pos]

    # ---------------- API ----------------
    def profiles(self, limit=8):
        """A spread of real users, from most active down to the 90th percentile of activity."""
        users, counts = self.rec.users_by_activity()
        ranks = np.unique((np.linspace(0, 0.9, limit) * (len(users) - 1)).astype(int))
        return [{"userId": int(users[r]), "numRatings": int(counts[r])} for r in ranks]

    def history(self, user_id, n, genres, decade):
        r = self.rec.ratings
        h = r.loc[r["userId"] == user_id]
        sc = np.full(self.n_items, -np.inf)
        for mid, rating in zip(h["movieId"], h["rating"]):
            if mid in self.rec.item_pos:
                sc[self.rec.item_pos[mid]] = rating
        valid = self.mask(genres, decade) & np.isfinite(sc)
        return self._pack(self._top(sc, valid, n))

    def recommend(self, user_id, alpha, n, genres, decade):
        rec = self.rec
        if user_id not in rec.user_pos:
            raise KeyError(user_id)
        row = rec.user_pos[user_id]
        cf = rec._minmax(rec._cf_row(row))
        ct = rec._minmax(rec.Gd @ rec.profiles[row])
        score = alpha * cf + (1 - alpha) * ct          # same formula as Recommender.recommend
        valid = self.mask(genres, decade)
        valid[self._positions(rec.seen.get(user_id, ()))] = False
        return self._pack(self._top(score, valid, n), score)

    def recommend_new(self, liked_ids, n, genres, decade):
        pos = self._positions(liked_ids)
        if not pos:
            return self.popular(n, genres, decade, [])
        profile = self.rec.Gd[pos].sum(axis=0)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        ct = (self.rec.Gd @ profile).astype(float)
        valid = self.mask(genres, decade)
        valid[pos] = False
        return self._pack(self._top(ct, valid, n), self.rec._minmax(ct))

    def popular(self, n, genres, decade, exclude_ids):
        valid = self.mask(genres, decade)
        valid[self._positions(exclude_ids)] = False
        return self._pack(self._top(self.num.astype(float), valid, n))

    def similar(self, title, n, genres, decade):
        pos = self.rec.title_to_pos.get(title)
        if pos is None:
            return []
        sims = (self.rec.Gd @ self.rec.Gd[pos]).astype(float)
        valid = self.mask(genres, decade)
        valid[pos] = False
        return self._pack(self._top(sims, valid, n), sims)

    def search(self, q, n):
        q = q.strip().lower()
        if not q:
            return []
        hits = [i for i, t in enumerate(self.titles_lower) if q in t]
        hits.sort(key=lambda i: -self.num[i])
        return self._pack(hits[:n])

    def movie(self, movie_id):
        pos = self.rec.item_pos.get(movie_id)
        return self._pack([pos])[0] if pos is not None else None
