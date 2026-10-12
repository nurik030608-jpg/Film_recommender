"""CineMatch API — serves the trained model to the Lovable frontend.

Run from the repo root:   uvicorn server.main:app --port 8000
Routes match src/lib/api.ts (`${API_BASE}/api/...`).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model.recommender import Recommender
from server.engine import Engine

engine = Engine(Recommender())
app = FastAPI(title="CineMatch API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Opts(BaseModel):
    genres: list[str] = []
    decade: str = "all"
    n: int = Field(12, ge=1, le=100)


class HistoryReq(Opts):
    userId: int


class RecommendReq(Opts):
    userId: int
    alpha: float = Field(0.8, ge=0, le=1)


class NewUserReq(Opts):
    likedMovieIds: list[int] = []


class PopularReq(Opts):
    excludeIds: list[int] = []


class SimilarReq(Opts):
    title: str
    n: int = Field(30, ge=1, le=100)


@app.get("/api/profiles")
def profiles():
    return engine.profiles()


@app.post("/api/history")
def history(r: HistoryReq):
    return engine.history(r.userId, r.n, r.genres, r.decade)


@app.post("/api/recommend")
def recommend(r: RecommendReq):
    try:
        return engine.recommend(r.userId, r.alpha, r.n, r.genres, r.decade)
    except KeyError:
        raise HTTPException(404, f"user {r.userId} is not in the trained model")


@app.post("/api/recommend_new")
def recommend_new(r: NewUserReq):
    return engine.recommend_new(r.likedMovieIds, r.n, r.genres, r.decade)


@app.post("/api/popular")
def popular(r: PopularReq):
    return engine.popular(r.n, r.genres, r.decade, r.excludeIds)


@app.post("/api/similar")
def similar(r: SimilarReq):
    return engine.similar(r.title, r.n, r.genres, r.decade)


@app.get("/api/search")
def search(q: str = "", n: int = 8):
    return engine.search(q, n)


@app.get("/api/movie/{movie_id}")
def movie(movie_id: int):
    return engine.movie(movie_id)  # null when unknown (frontend expects Movie | null)
