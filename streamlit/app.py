"""CineMatch — Netflix-style Streamlit frontend.

Calls the trained checkpoint in model/checkpoints/ via model/recommender.py.
This file never trains anything and never touches datasets/ — it only reads
the exported model artifacts through the Recommender class.

Run:  streamlit run streamlit/app.py   (from the repo root)

Clickable-card technique: each movie tile is a real st.button (not an
overlay), styled via the CSS class Streamlit auto-generates from a widget's
`key` (`.st-key-<key>`). This needs a reasonably modern Streamlit version
(the `st-key-*` class hook). requirements.txt pins an unpinned "streamlit"
so Streamlit Cloud installs the latest release, which has it.
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# --- make `model/` importable regardless of where streamlit is launched from ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.recommender import Recommender  # noqa: E402

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------------
# Netflix-style theme (static, page-wide rules only — per-card rules are
# injected next to each card since they depend on that card's unique key)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #141414; }
    #MainMenu, footer { visibility: hidden; }

    .cm-hero {
        padding: 2.2rem 1rem 1.2rem 1rem;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 1rem;
    }
    .cm-logo {
        color: #E50914; font-size: 2.4rem; font-weight: 800;
        letter-spacing: -1px; margin: 0;
    }
    .cm-sub { color: #b3b3b3; font-size: 0.95rem; margin-top: 0.2rem; }

    .cm-row-title {
        color: #ffffff; font-size: 1.3rem; font-weight: 700;
        margin: 1.6rem 0 0.6rem 0.2rem;
    }
    .cm-empty { color: #808080; font-style: italic; padding: 1rem 0.2rem; }

    .cm-decade-title {
        color: #fff; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px;
    }

    /* ---- Detail panel: slides in from the right, doesn't cover the page ---- */
    .st-key-detail_panel {
        position: fixed !important;
        top: 0; right: 0;
        width: 46%;
        min-width: 380px;
        max-width: 640px;
        height: 100vh;
        background: #181818 !important;
        box-shadow: -10px 0 34px rgba(0,0,0,0.65);
        z-index: 9999;
        overflow-y: auto;
        padding: 1.6rem 1.4rem 2rem 1.4rem !important;
        animation: cm-slide-in 0.3s ease-out;
        border-left: 1px solid #2a2a2a;
    }
    @keyframes cm-slide-in {
        from { transform: translateX(100%); opacity: 0.5; }
        to   { transform: translateX(0);    opacity: 1;   }
    }
    div[class*="st-key-panel_movies_"] {
        animation: cm-fade-in 0.22s ease;
    }
    @keyframes cm-fade-in {
        from { opacity: 0; transform: translateY(5px); }
        to   { opacity: 1; transform: translateY(0);   }
    }
    .st-key-panel_close button {
        background: transparent !important; border: none !important;
        color: #b3b3b3 !important; font-size: 1.1rem !important;
        padding: 0 !important; width: auto !important;
    }
    .st-key-panel_close button:hover { color: #fff !important; }
    .st-key-panel_prev button, .st-key-panel_next button {
        background: #262626 !important; color: #fff !important;
        border: 1px solid #404040 !important; border-radius: 4px !important;
    }
    .st-key-panel_prev button:hover, .st-key-panel_next button:hover {
        background: #E50914 !important; border-color: #E50914 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model loading (cached — checkpoint is read from disk once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CineMatch model…")
def load_model():
    return Recommender()


rec = load_model()
meta = rec.meta

st.session_state.setdefault("panel_movie", None)
st.session_state.setdefault("panel_offset", 0)

# a fixed palette so the same movie always gets the same gradient
PALETTE = [
    ("#E50914", "#831010"), ("#0071EB", "#003a75"), ("#7B2FF7", "#3d0f8f"),
    ("#00A896", "#02565a"), ("#F5A623", "#8a4e00"), ("#D91E5C", "#5c0c26"),
    ("#2E8B57", "#0f3d24"), ("#8E44AD", "#3b1c4a"),
]


def card_gradient(title: str) -> tuple[str, str]:
    h = int(hashlib.md5(title.encode()).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]


def make_key(*parts) -> str:
    """CSS-class-safe, deterministic key for a widget (raw titles contain
    spaces/quotes/unicode, which would break a `.st-key-<key>` CSS selector)."""
    raw = "|".join(str(p) for p in parts)
    return "c" + hashlib.md5(raw.encode()).hexdigest()[:14]


@st.cache_data(show_spinner=False)
def all_genres(_rec) -> list[str]:
    genre_set = set()
    for g in _rec.movies["genres"].dropna():
        genre_set.update(g.split("|"))
    return sorted(x for x in genre_set if x and x != "(no genres listed)")


def browse_by_genre(rec, selected_genres: list[str], n: int = 12):
    """Movies matching ANY of the selected genres, ranked by popularity."""
    if not selected_genres:
        return None
    mask = rec.movies["genres"].apply(
        lambda g: isinstance(g, str) and any(genre in g.split("|") for genre in selected_genres)
    )
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        return rec.movies.iloc[[]]
    order = idx[np.argsort(-rec.num_ratings[idx])][:n]
    out = rec.movies.iloc[order][["movieId", "title", "genres"]].reset_index(drop=True)
    out["num_ratings"] = rec.num_ratings[order]
    return out


YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

DECADE_BUCKETS = [
    ("Before 1980", None, 1979),
    ("1980s", 1980, 1989),
    ("1990s", 1990, 1999),
    ("2000s", 2000, 2009),
    ("2010s", 2010, 2019),
    ("2020s+", 2020, None),
]


def parse_year(title: str):
    m = YEAR_RE.search(title)
    return int(m.group(1)) if m else None


@st.cache_data(show_spinner=False)
def browse_by_decade(_rec, n_per_decade: int = 6) -> dict:
    years = _rec.movies["title"].apply(parse_year)
    out = {}
    for label, lo, hi in DECADE_BUCKETS:
        mask = years.notna()
        if lo is not None:
            mask &= years >= lo
        if hi is not None:
            mask &= years <= hi
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            out[label] = None
            continue
        order = idx[np.argsort(-_rec.num_ratings[idx])][:n_per_decade]
        df = _rec.movies.iloc[order][["movieId", "title", "genres"]].reset_index(drop=True)
        df["num_ratings"] = _rec.num_ratings[order]
        out[label] = df
    return out


# ----------------------------------------------------------------------------
# Clickable card: a real st.button, skinned via its own `.st-key-<key>` CSS
# rule to look like a gradient tile. Clicking it opens/updates the side panel.
# ----------------------------------------------------------------------------
def render_card(movie, key: str, score_col: str | None = None,
                 height: int = 168, compact: bool = False):
    c1, c2 = card_gradient(movie["title"])
    genres = movie.get("genres", "")
    genres_display = genres.replace("|", " · ") if isinstance(genres, str) else ""

    subtitle = genres_display
    if score_col and score_col in movie:
        pct = int(round(movie[score_col] * 100))
        subtitle = f"{pct}% match · {genres_display}" if genres_display else f"{pct}% match"

    if compact:
        label = movie["title"]
    else:
        label = f"{movie['title']}  \n*{subtitle}*" if subtitle else movie["title"]

    title_size = "0.68rem" if compact else "0.92rem"
    padding = "6px 8px" if compact else "14px 12px"
    radius = 4 if compact else 6
    shadow = "0 2px 8px rgba(0,0,0,0.4)" if compact else "0 4px 14px rgba(0,0,0,0.5)"

    st.markdown(f"""
<style>
.st-key-{key} button {{
    background: linear-gradient(135deg, {c1}, {c2}) !important;
    height: {height}px !important; width: 100% !important;
    border: none !important; border-radius: {radius}px !important;
    color: white !important; text-align: left !important;
    display: flex !important; flex-direction: column !important;
    align-items: flex-start !important; justify-content: flex-end !important;
    padding: {padding} !important;
    box-shadow: {shadow} !important;
    transition: transform 0.15s ease !important;
    white-space: normal !important; cursor: pointer !important;
}}
.st-key-{key} button:hover {{ transform: scale(1.035) !important; }}
.st-key-{key} button p {{
    margin: 0 !important; font-weight: 700 !important;
    font-size: {title_size} !important; line-height: 1.22 !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.7) !important;
}}
.st-key-{key} button p em {{
    display: block !important; font-style: normal !important;
    font-size: 0.72rem !important; font-weight: 400 !important;
    opacity: 0.9 !important; margin-top: 3px !important;
}}
</style>
""", unsafe_allow_html=True)

    if st.button(label, key=key, use_container_width=True):
        st.session_state["panel_movie"] = movie["title"]
        st.session_state["panel_offset"] = 0


def render_row(row_title: str, df, score_col: str | None = None):
    st.markdown(f'<div class="cm-row-title">{row_title}</div>', unsafe_allow_html=True)
    if df is None or len(df) == 0:
        st.markdown('<div class="cm-empty">Nothing here yet — try a different selection.</div>',
                    unsafe_allow_html=True)
        return
    cols = st.columns(min(len(df), 6))
    for i, (_, movie) in enumerate(df.iterrows()):
        with cols[i % len(cols)]:
            render_card(movie, key=make_key("row", row_title, i, movie["title"]),
                        score_col=score_col)


# ----------------------------------------------------------------------------
# Detail panel — slides in from the right (1/4 info + 3/4 similar titles)
# when a card is clicked. Clicking a movie inside the panel swaps the panel
# to that movie (reuses render_card's own click handler — same mechanism,
# so there's no separate/inconsistent click path to go stale or misbehave).
# ----------------------------------------------------------------------------
def render_panel():
    movie_title = st.session_state.get("panel_movie")
    if not movie_title or movie_title not in rec.title_to_pos:
        return

    with st.container(key="detail_panel"):
        close_col, _spacer = st.columns([1, 9])
        with close_col:
            if st.button("✕", key="panel_close"):
                st.session_state["panel_movie"] = None

        info_col, movies_col = st.columns([1, 3])

        row = rec.movies.iloc[rec.title_to_pos[movie_title]]
        with info_col:
            st.markdown(f"#### {row['title']}")
            genres_list = row["genres"].split("|") if isinstance(row["genres"], str) else []
            for g in genres_list:
                st.markdown(f"- {g}")

        with movies_col:
            st.markdown("**Similar titles**")
            sims = rec.similar_movies(movie_title, n=30)
            total = len(sims)
            offset = min(st.session_state.get("panel_offset", 0), max(0, total - 3))
            page = sims.iloc[offset: offset + 3]

            with st.container(key=f"panel_movies_{offset}"):
                mcols = st.columns(3)
                for i, (_, m) in enumerate(page.iterrows()):
                    with mcols[i]:
                        render_card(m, key=make_key("panel", movie_title, offset, m["title"]),
                                    score_col="similarity", height=130, compact=False)

            nav_l, nav_r = st.columns(2)
            with nav_l:
                if offset > 0:
                    if st.button("◀ Back", key="panel_prev", use_container_width=True):
                        st.session_state["panel_offset"] = max(0, offset - 3)
            with nav_r:
                if offset + 3 < total:
                    if st.button("Next ▶", key="panel_next", use_container_width=True):
                        st.session_state["panel_offset"] = offset + 3


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="cm-hero">
    <p class="cm-logo">CINEMATCH</p>
    <p class="cm-sub">
        Hybrid recommender · FunkSVD (k={meta['k']}) + genre content · α={meta['best_alpha']} ·
        {meta['n_users_sampled']:,} users · {meta['n_items']:,} titles · test RMSE {meta['test_rmse']}
    </p>
</div>
""", unsafe_allow_html=True)

render_panel()

# ----------------------------------------------------------------------------
# "Who's watching" — profile picker (existing user vs. new user)
# ----------------------------------------------------------------------------
mode = st.sidebar.radio("Who's watching?", ["Existing user", "New user"])
st.sidebar.divider()

if mode == "Existing user":
    users, counts = rec.users_by_activity()
    idx = st.sidebar.selectbox(
        "Profile",
        range(len(users)),
        index=len(users) // 2,
        format_func=lambda i: f"User {users[i]}  ·  {counts[i]} ratings",
    )
    user_id = int(users[idx])
    alpha = st.sidebar.slider(
        "Balance: genres ↔ similar tastes", 0.0, 1.0, float(meta["best_alpha"]), 0.05,
        help="0 = genres only, 1 = collaborative filtering only",
    )
    top_n = st.sidebar.slider("How many to show", 6, 24, 12, 6)

    history = rec.user_history(user_id, 6)
    render_row("Your History (Top Rated)", history)
    render_row("Top Picks For You", rec.recommend(user_id, alpha=alpha, n=top_n), score_col="hybrid")
    render_row("Trending Now", rec.popular(top_n))

    if len(history) > 0:
        seed_title = history.iloc[0]["title"]
        render_row(f"Because you watched: {seed_title}", rec.similar_movies(seed_title, top_n))

else:
    st.sidebar.markdown("Pick a few movies you like — we'll build your taste profile on the fly.")
    all_titles = rec.movies["title"].tolist()
    liked_titles = st.sidebar.multiselect("Movies you liked", all_titles)
    top_n = st.sidebar.slider("How many to show", 6, 24, 12, 6)

    render_row("Trending Now", rec.popular(top_n))

    if liked_titles:
        liked_ids = [int(rec.movies["movieId"].iloc[rec.title_to_pos[t]]) for t in liked_titles]
        render_row("Recommended For You (by genre)",
                   rec.recommend_for_new_user(liked_ids, top_n), score_col="content")
    else:
        st.markdown('<div class="cm-empty">Pick at least one movie on the left '
                    'to get personalized recommendations.</div>', unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("**Browse by Genre**")
selected_genres = st.sidebar.multiselect("Genres", all_genres(rec))
genre_n = st.sidebar.slider("How many to show", 6, 24, 12, 6, key="genre_n")

st.divider()

# ----------------------------------------------------------------------------
# Browse by Genre (sidebar multiselect feeds this row)
# ----------------------------------------------------------------------------
if selected_genres:
    render_row(f"Genres: {', '.join(selected_genres)}",
               browse_by_genre(rec, selected_genres, genre_n))
else:
    st.markdown('<div class="cm-row-title">🎭 Browse by Genre</div>', unsafe_allow_html=True)
    st.markdown('<div class="cm-empty">Pick one or more genres in the sidebar to browse titles.'
                '</div>', unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------------------------------
# Browse by Decade — 6 columns, one per decade bucket
# ----------------------------------------------------------------------------
st.markdown('<div class="cm-row-title">📅 Browse by Decade</div>', unsafe_allow_html=True)
decade_data = browse_by_decade(rec, 6)
decade_cols = st.columns(6)
for col, (label, df) in zip(decade_cols, decade_data.items()):
    with col:
        st.markdown(f'<div class="cm-decade-title">{label}</div>', unsafe_allow_html=True)
        if df is None or len(df) == 0:
            st.markdown('<div class="cm-empty">—</div>', unsafe_allow_html=True)
            continue
        for j, (_, movie) in enumerate(df.iterrows()):
            render_card(movie, key=make_key("decade", label, j, movie["title"]),
                        height=78, compact=True)

st.divider()
st.caption(
    "Dataset: F. M. Harper & J. A. Konstan (2015), The MovieLens Datasets, ACM TiiS 5(4). "
    "Model checkpoint loaded read-only from model/checkpoints/ — no training happens here."
)
