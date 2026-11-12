"""Roblox Up-and-Coming top-10 forecaster for make.fun (program/v1).

Reads the chart history in $MAKEFUN_INPUT_DIR (else /input) and writes
$output/forecast.json: one probability distribution over the top 10
places, averaged over the 72 scored hours, plus a top10 guess.

Model: every game gets a latent Plackett-Luce strength from its
recency-weighted ranks. A seeded Monte Carlo then walks each strength
forward hour by hour (mean reversion to the game's own level plus
noise), ages games out at 28 days when their creation date is known,
samples each hour's chart by strength plus Gumbel noise, and averages
the top-10 occupancy over the whole window. A slice of every place is
left unnamed on purpose: it is the mass for games that join the chart
during the window, which the scorer spreads over the real chart.

Deterministic: fixed seed, sorted iteration, no clock, no network.
Python 3.13 standard library only.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

SEED = 20261113
SIMS = 500
HALF_LIFE_H = 30.0        # recency weight half-life for rank evidence
STEP_NOISE = 0.22
REVERT = 0.035            # hourly mean reversion toward the game's level
GUMBEL = 0.70
UNNAMED = 0.12
MIN_P = 0.0009            # rows below this are dropped (scorer floor 0.0001)
MAX_ROWS_PER_PLACE = 100


def gumbel(rng, beta):
    u = rng.random()
    if u <= 0.0:
        u = 1e-12
    if u >= 1.0:
        u = 1.0 - 1e-12
    return -beta * math.log(-math.log(u))


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(indir):
    task = json.load(open(os.path.join(indir, "task.json")))
    until = parse_ts(task["history_until"])
    reads = []
    created = {}
    with open(os.path.join(indir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chart = []
            for g in r.get("up_and_coming") or []:
                chart.append((int(g["universe_id"]), int(g["rank"])))
            det = r.get("details") or {}
            for uid, d in det.items():
                if d and d.get("created"):
                    try:
                        created[int(uid)] = parse_ts(d["created"])
                    except Exception:
                        pass
            reads.append((parse_ts(r["at"]), chart))
    reads.sort(key=lambda x: x[0])
    return task, until, reads, created


def game_stats(reads, until):
    """Recency-weighted evidence per game."""
    stats = {}
    for at, chart in reads:
        age_h = max(0.0, (until - at).total_seconds() / 3600.0)
        w = 0.5 ** (age_h / HALF_LIFE_H)
        for uid, rank in chart:
            s = stats.setdefault(uid, {"w": 0.0, "wr": 0.0, "last_rank": rank,
                                       "last_w": -1.0, "seen": 0})
            s["w"] += w
            s["wr"] += w * rank
            s["seen"] += 1
            if w >= s["last_w"]:
                s["last_w"] = w
                s["last_rank"] = rank
    total_w = sum(0.5 ** (max(0.0, (until - at).total_seconds() / 3600.0) / HALF_LIFE_H)
                  for at, _ in reads)
    return stats, total_w


def main():
    indir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, until, reads, created = load_history(indir)
    stats, total_w = game_stats(reads, until)
    window_from = parse_ts(task["window"]["from"])

    # Latent strength per game: better (lower) weighted rank -> higher
    # strength; presence in recent reads raises it; a vanished game fades.
    games = []
    for uid, s in stats.items():
        mean_rank = s["wr"] / s["w"]
        presence = s["w"] / total_w if total_w else 0.0
        strength = -math.log(mean_rank) + 0.9 * presence + 0.35 * (1.0 / max(1, s["last_rank"]))
        games.append({"uid": uid, "theta": strength, "level": strength,
                      "created": created.get(uid)})
    games.sort(key=lambda g: g["uid"])

    rng = random.Random(SEED)
    places = 10
    hours = int(task["window"]["slots"])
    counts = {g["uid"]: [0] * places for g in games}
    top10_hits = {g["uid"]: 0 for g in games}
    place_sum = {g["uid"]: 0.0 for g in games}

    for _ in range(SIMS):
        theta = {g["uid"]: g["theta"] for g in games}
        alive = list(games)
        for h in range(hours):
            now = window_from.timestamp() + (h + 1) * 3600
            scored = []
            for g in alive:
                uid = g["uid"]
                theta[uid] += REVERT * (g["level"] - theta[uid]) \
                    + rng.gauss(0.0, STEP_NOISE)
                if g["created"] is not None:
                    age_days = (now - g["created"].timestamp()) / 86400.0
                    if age_days > 28.0:
                        continue
                scored.append((theta[uid] + gumbel(rng, GUMBEL), uid))
            scored.sort(key=lambda x: -x[0])
            for place, (_, uid) in enumerate(scored[:places]):
                counts[uid][place] += 1
                place_sum[uid] += place + 1
                top10_hits[uid] += 1

    total = SIMS * hours
    probabilities = []
    for place in range(places):
        rows = []
        for g in games:
            uid = g["uid"]
            p = counts[uid][place] / total
            if p >= MIN_P:
                rows.append((uid, p))
        rows.sort(key=lambda x: -x[1])
        rows = rows[:MAX_ROWS_PER_PLACE]
        s = sum(p for _, p in rows)
        cap = 1.0 - UNNAMED
        if s > cap:
            rows = [(uid, p * cap / s) for uid, p in rows]
        for uid, p in rows:
            probabilities.append({"universe_id": uid, "place": place + 1,
                                  "p": round(p, 6)})

    ranked = sorted(games, key=lambda g: (-top10_hits[g["uid"]],
                                          place_sum[g["uid"]] / max(1, top10_hits[g["uid"]])))
    top10 = [g["uid"] for g in ranked[:places]]

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "forecast.json"), "w") as f:
        json.dump({"probabilities": probabilities, "top10": top10}, f,
                  separators=(",", ":"))


if __name__ == "__main__":
    main()
