import os
import re
import time
import base64
import math
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


def norm_name(s: str) -> str:
    """Normalize artist name for robust join (case/space/punct-insensitive)."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    return s


def get_spotify_token(client_id: str, client_secret: str) -> str:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    headers = {"Authorization": f"Basic {auth}"}
    data = {"grant_type": "client_credentials"}
    resp = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def spotify_get(url: str, token: str, params=None, timeout=30):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)

    # retry for rate limit
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "2"))
        time.sleep(retry_after + 1)
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)

    resp.raise_for_status()
    return resp.json()


def search_artist_best_match(query_name: str, token: str) -> dict:
    """Return best match artist object from Spotify search."""
    params = {"q": query_name, "type": "artist", "limit": 5}
    data = spotify_get(f"{SPOTIFY_API_BASE}/search", token, params=params)
    items = data.get("artists", {}).get("items", [])
    if not items:
        return {}

    target = norm_name(query_name)
    exacts = [a for a in items if norm_name(a.get("name", "")) == target]
    if exacts:
        return sorted(exacts, key=lambda x: x.get("popularity", 0), reverse=True)[0]

    return sorted(items, key=lambda x: x.get("popularity", 0), reverse=True)[0]


def get_artist(artist_id: str, token: str) -> dict:
    return spotify_get(f"{SPOTIFY_API_BASE}/artists/{artist_id}", token)


def get_artist_top_tracks(artist_id: str, token: str, market: str = "KR") -> list:
    data = spotify_get(
        f"{SPOTIFY_API_BASE}/artists/{artist_id}/top-tracks",
        token,
        params={"market": market},
    )
    return data.get("tracks", [])


def get_track_popularity(track_obj: dict) -> int:
    return int(track_obj.get("popularity", 0) or 0)


def safe_join_list(xs, max_items=5):
    xs = [x for x in xs if x]
    if not xs:
        return ""
    return " | ".join(xs[:max_items])


def get_artist_albums_recent_count(artist_id: str, token: str, years: int = 3) -> tuple[int, str]:
    """
    Count albums/singles released in last `years`.
    Returns (count, latest_release_date_iso)
    """
    cutoff = datetime.now(timezone.utc) - relativedelta(years=years)

    items = []
    url = f"{SPOTIFY_API_BASE}/artists/{artist_id}/albums"
    params = {"include_groups": "album,single", "limit": 50, "offset": 0}

    latest_date = None
    for _ in range(4):  # up to 200 items
        data = spotify_get(url, token, params=params)
        page = data.get("items", [])
        if not page:
            break
        items.extend(page)
        if data.get("next") is None:
            break
        params["offset"] += params["limit"]
        time.sleep(0.15)

    cnt = 0
    for a in items:
        rd = a.get("release_date")
        prec = a.get("release_date_precision")
        if not rd:
            continue

        try:
            if prec == "year":
                dt = datetime(int(rd), 1, 1, tzinfo=timezone.utc)
            elif prec == "month":
                y, m = rd.split("-")
                dt = datetime(int(y), int(m), 1, tzinfo=timezone.utc)
            else:
                y, m, d = rd.split("-")
                dt = datetime(int(y), int(m), int(d), tzinfo=timezone.utc)
        except Exception:
            continue

        if latest_date is None or dt > latest_date:
            latest_date = dt

        if dt >= cutoff:
            cnt += 1

    latest_iso = latest_date.date().isoformat() if latest_date else ""
    return cnt, latest_iso


def score_artist(popularity: int, followers: int, kr_top_tracks_count: int,
                 top_track_concentration: float, releases_3y: int) -> dict:
    """
    Spotify-only scoring (0-100):
    - Scale: 40 (popularity + followers)
    - KR presence: 30 (KR top tracks count)
    - Catalog stability: 30 (대표곡 쏠림 완화 + 최근 3년 릴리즈)
    """
    # Scale (0-40)
    pop_score = max(0, min(25, (popularity / 100) * 25))

    followers_score = 0.0
    if followers > 0:
        lo, hi = 4.699, 5.477  # log10(50k), log10(300k)
        val = math.log10(followers)
        followers_score = (val - lo) / (hi - lo) * 15
        followers_score = max(0, min(15, followers_score))

    scale_score = pop_score + followers_score  # 0-40

    # KR presence (0-30)
    kr_presence_score = max(0, min(30, (kr_top_tracks_count / 10) * 30))

    # Catalog stability (0-30)
    conc_score = (1 - max(0.0, min(1.0, top_track_concentration))) * 15  # 0-15
    rel_score = max(0, min(15, (releases_3y / 10) * 15))
    catalog_score = conc_score + rel_score

    total = scale_score + kr_presence_score + catalog_score

    return {
        "scale_score": round(scale_score, 2),
        "kr_presence_score": round(kr_presence_score, 2),
        "catalog_score": round(catalog_score, 2),
        "total_score": round(total, 2),
    }


def main():
    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET environment variables.")

    token = get_spotify_token(client_id, client_secret)

    if not os.path.exists("artists.txt"):
        raise RuntimeError("artists.txt not found in repo root.")

    with open("artists.txt", "r", encoding="utf-8") as f:
        input_artists = [line.strip() for line in f if line.strip()]

    if not input_artists:
        raise RuntimeError("artists.txt is empty.")

    # Load country map
    country_map = {}
    if os.path.exists("artist_country_map.csv"):
        df_map = pd.read_csv("artist_country_map.csv")
        if "artist_name" in df_map.columns and "country" in df_map.columns:
            for _, r in df_map.iterrows():
                country_map[norm_name(str(r["artist_name"]))] = str(r["country"]).strip()

    rows = []
    for name in tqdm(input_artists, desc="Fetching Spotify data"):
        found = search_artist_best_match(name, token)
        if not found:
            rows.append({
                "input_name": name,
                "artist_name": "",
                "artist_id": "",
                "country": country_map.get(norm_name(name), ""),
                "match_status": "NOT_FOUND",
            })
            continue

        artist_id = found["id"]
        artist = get_artist(artist_id, token)

        popularity = int(artist.get("popularity", 0) or 0)
        followers = int(artist.get("followers", {}).get("total", 0) or 0)
        genres = artist.get("genres", []) or []

        kr_tracks = get_artist_top_tracks(artist_id, token, market="KR")
        kr_count = len(kr_tracks)

        kr_track_names = [t.get("name", "") for t in kr_tracks[:5]]
        kr_track_pops = [str(get_track_popularity(t)) for t in kr_tracks[:5]]

        pops = [get_track_popularity(t) for t in kr_tracks[:5]]
        pop_sum = sum(pops) if pops else 0
        top1 = max(pops) if pops else 0
        concentration = (top1 / pop_sum) if pop_sum > 0 else 0.0

        rel_3y, latest_release_date = get_artist_albums_recent_count(artist_id, token, years=3)

        # Join country map: prefer exact Spotify name, else input name
        country = country_map.get(norm_name(artist.get("name", "")), "") or country_map.get(norm_name(name), "")

        scores = score_artist(
            popularity=popularity,
            followers=followers,
            kr_top_tracks_count=kr_count,
            top_track_concentration=concentration,
            releases_3y=rel_3y
        )

        rows.append({
            "input_name": name,
            "artist_name": artist.get("name", ""),
            "artist_id": artist_id,
            "country": country,
            "match_status": "OK",
            "popularity": popularity,
            "followers_total": followers,
            "genres": ", ".join(genres),
            "kr_top_tracks_count": kr_count,
            "kr_top_track_names_top5": safe_join_list(kr_track_names, max_items=5),
            "kr_top_track_popularity_top5": safe_join_list(kr_track_pops, max_items=5),
            "top_track_concentration_top5": round(concentration, 4),
            "releases_last_3y_count": rel_3y,
            "latest_release_date": latest_release_date,
            **scores
        })

        time.sleep(0.12)

    df = pd.DataFrame(rows)

    # Raw metrics
    df.to_csv("artist_metrics.csv", index=False, encoding="utf-8-sig")

    # Ranked
    ok = df[df["match_status"] == "OK"].copy()
    ok = ok.sort_values(
        ["total_score", "kr_top_tracks_count", "popularity", "followers_total"],
        ascending=False
    )
    ok.insert(0, "rank", range(1, len(ok) + 1))
    ok.to_csv("artist_ranked.csv", index=False, encoding="utf-8-sig")

    print("Done. Outputs: artist_metrics.csv, artist_ranked.csv")


if __name__ == "__main__":
    main()
