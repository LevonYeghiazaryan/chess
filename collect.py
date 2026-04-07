"""
collect.py — Data collection pipeline for Chess.com Titled Tuesday prediction.

Fetches tournament games, player profiles, and player stats from Chess.com PubAPI.
Saves 3 CSV files to the data/ directory.

Usage:
    python collect.py              # Full run: fetch from API + save CSVs
    python collect.py --skip-fetch  # Skip API calls, just rebuild from cached CSVs

API endpoints used:
    /pub/tournament/{id}/{round}/{group}  — game data
    /pub/player/{username}                — player profile (title, country)
    /pub/player/{username}/stats          — blitz rating, win/loss/draw record
"""

import argparse
import os
import time

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "chess-ml-project"}
BASE_URL = "https://api.chess.com/pub"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TOURNAMENTS = {
    "feb_2026": "titled-tuesday-blitz-february-10-2026-6221327",
    "mar_2026": "titled-tuesday-blitz-march-10-2026-6277141",
}
NUM_ROUNDS = 11


# ── API helpers ───────────────────────────────────────────────────────────

def fetch_json(url: str) -> dict | None:
    """GET JSON from Chess.com API with retry on rate limit."""
    for attempt in range(3):
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            print(f"    Rate limited, retrying in 5s (attempt {attempt + 1})")
            time.sleep(5)
        else:
            return None
    return None


# ── Phase 1: Tournament games ────────────────────────────────────────────

def fetch_games() -> pd.DataFrame:
    """Fetch all games from both tournaments."""
    all_games = []

    for label, tid in TOURNAMENTS.items():
        print(f"\n  [{label}] Fetching {tid}...")

        for round_num in range(1, NUM_ROUNDS + 1):
            round_data = fetch_json(f"{BASE_URL}/tournament/{tid}/{round_num}")
            if not round_data:
                continue

            for group_url in round_data.get("groups", []):
                group_data = fetch_json(group_url)
                if not group_data:
                    continue

                for game in group_data.get("games", []):
                    game["tournament"] = label
                    game["round"] = round_num
                all_games.extend(group_data.get("games", []))

            print(f"    Round {round_num}: done")
            time.sleep(0.5)

    df = pd.json_normalize(all_games)
    print(f"\n  Total games: {len(df)}")
    return df


# ── Phase 2: Player profiles & stats ─────────────────────────────────────

def fetch_players(df_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch profile and blitz stats for every unique player."""
    players = sorted(
        set(df_games["white.username"].str.lower()) |
        set(df_games["black.username"].str.lower())
    )
    print(f"\n  Fetching data for {len(players)} players...")

    profiles, stats = [], []

    for i, username in enumerate(players):
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{len(players)}")

        # Profile
        p = fetch_json(f"{BASE_URL}/player/{username}")
        if p:
            profiles.append({
                "username": username,
                "title": p.get("title"),
                "country": p.get("country", "").split("/")[-1] if p.get("country") else None,
                "joined": p.get("joined"),
                "status": p.get("status"),
                "is_streamer": p.get("is_streamer", False),
            })

        # Stats
        s = fetch_json(f"{BASE_URL}/player/{username}/stats")
        if s:
            blitz = s.get("chess_blitz", {})
            rec = blitz.get("record", {})
            total = rec.get("win", 0) + rec.get("loss", 0) + rec.get("draw", 0)
            stats.append({
                "username": username,
                "blitz_last_rating": blitz.get("last", {}).get("rating"),
                "blitz_last_rd": blitz.get("last", {}).get("rd"),
                "blitz_best_rating": blitz.get("best", {}).get("rating"),
                "blitz_wins": rec.get("win", 0),
                "blitz_losses": rec.get("loss", 0),
                "blitz_draws": rec.get("draw", 0),
                "blitz_total_games": total,
                "blitz_win_rate": rec.get("win", 0) / total if total else None,
                "blitz_draw_rate": rec.get("draw", 0) / total if total else None,
            })

        time.sleep(0.3)

    return pd.DataFrame(profiles), pd.DataFrame(stats)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chess.com data collection pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip API calls, use existing CSVs in data/")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    games_path = f"{DATA_DIR}/raw_games.csv"
    profiles_path = f"{DATA_DIR}/player_profiles.csv"
    stats_path = f"{DATA_DIR}/player_stats.csv"

    if args.skip_fetch:
        print("Skipping API fetch, loading cached CSVs...")
        df_games = pd.read_csv(games_path)
        df_profiles = pd.read_csv(profiles_path)
        df_stats = pd.read_csv(stats_path)
    else:
        # Phase 1: Games
        print("=" * 60)
        print("PHASE 1: Fetching tournament games")
        print("=" * 60)
        df_games = fetch_games()
        df_games.to_csv(games_path, index=False)
        print(f"  Saved: {games_path} ({df_games.shape})")

        # Phase 2: Players
        print("\n" + "=" * 60)
        print("PHASE 2: Fetching player profiles & stats")
        print("=" * 60)
        df_profiles, df_stats = fetch_players(df_games)
        df_profiles.to_csv(profiles_path, index=False)
        df_stats.to_csv(stats_path, index=False)
        print(f"  Saved: {profiles_path} ({df_profiles.shape})")
        print(f"  Saved: {stats_path} ({df_stats.shape})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Games:    {len(df_games)} ({df_games['tournament'].value_counts().to_dict()})")
    print(f"  Profiles: {len(df_profiles)}")
    print(f"  Stats:    {len(df_stats)}")
    print(f"  Titles:   {df_profiles['title'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
