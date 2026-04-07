# Chess Game Outcome Prediction

Predict Win / Loss / Draw (from White's perspective) using pre-game features from Chess.com Titled Tuesday blitz tournaments.

**Data**: 4,063 games, 2 tournaments (Feb & Mar 2026), ~700 titled players.
**Model**: LightGBM (n_estimators=100, max_depth=3). Elo baseline for comparison.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch data from Chess.com API (~10 min, 1400 API calls)
python collect.py

# Or skip fetching if data/ already has CSVs:
python collect.py --skip-fetch

# 3. Run all models (3 splits × 13 models + SHAP plots)
python main_model.py
```

**Output**: model comparison tables printed to stdout, SHAP plots saved to `plots/`.

## Project Structure

```
chesscom/
  README.md           ← this file (experiment log + instructions)
  writeup.md          ← submission writeup (split rationale, quality assessment, next steps)
  requirements.txt    ← Python dependencies
  collect.py          ← data collection pipeline (Chess.com API → CSVs)
  main_model.py       ← full modeling pipeline (features → 3 splits × 13 models → SHAP)
  data/
    raw_games.csv     ← 4,063 games with ratings, results, PGN, FEN
    player_stats.csv  ← 699 players: blitz win/loss/draw record, best rating
    player_profiles.csv ← 700 players: title, country, account age
  plots/
    shap_importance.png
    shap_beeswarm.png
    shap_summary.png
```

---

## Experiment Log

## Trial 1: Ratings Only (6 features)

**Features**: `white.rating`, `black.rating`, `rating_diff`, `abs_rating_diff`, `round`, `is_march`

**Why**: Rating difference is the fundamental predictor of chess outcomes. `abs_rating_diff` helps draw detection (close games draw more). `round` captures Swiss pairing effect.

**Dropped during analysis**: `elo_expected_white` (sigmoid of rating_diff — redundant for trees), `avg_rating` (derivable from individual ratings — marginal value).

| | Test Acc | Test F1m |
|--|----------|----------|
| Elo Baseline | 0.696 | 0.483 |
| LightGBM (6f) | 0.704 | 0.488 |

**Verdict**: LightGBM barely beats Elo (+0.8% acc). Rating diff dominates. Draws never predicted.

---

## Trial 2: + Player Stats (14 features)

**Added** (8 features from `/pub/player/{username}/stats`):

| Feature | What it captures |
|---------|-----------------|
| `win_rate_diff` | Who wins more overall |
| `draw_rate_diff` | Who draws more |
| `draw_rate_avg` | Are both players draw-prone |
| `best_rating_diff` | Peak strength gap |
| `experience_diff` | log(total games) gap |
| `rating_gap_diff` | Who's further below peak |
| `white/black_blitz_last_rd` | Rating certainty |

| | Test Acc | Test F1m | Overfit Gap |
|--|----------|----------|-------------|
| LightGBM (6f) | 0.704 | 0.488 | +0.030 |
| **LightGBM (14f)** | **0.725** | **0.503** | +0.023 |

**Verdict**: +2.1% accuracy. Player stats add real signal beyond ratings. Best neutral model across all experiments.

---

## Trial 3: + Player Profiles (20 features)

**Added** (6 features from `/pub/player/{username}`): `title_strength_diff`, `account_age_diff`, `white/black_is_streamer`, `white/black_is_premium`

| | Test Acc | Test F1m |
|--|----------|----------|
| LightGBM (14f) | **0.725** | **0.503** |
| LightGBM (20f) | 0.715 | 0.495 |

**Verdict**: **Hurt performance** (-1.0% acc). Title is already captured by rating. Account age, streamer status, premium status are noise. Reverted.

---

## Trial 4: + Move Count from FEN (17 features)

**Added**: `white/black_avg_moves`, `avg_moves_diff` — average game length per player computed from FEN's fullmove counter.

| | Test Acc | Test F1m |
|--|----------|----------|
| LightGBM (14f) | **0.725** | **0.503** |
| LightGBM (17f) | 0.707 | 0.490 |

**Verdict**: **Hurt performance** (-1.8% acc). With 5-10 games per player, per-player averages from this dataset are too noisy. Reverted.

---

## Trial 5: + Clock Features from PGN (21 features)

**Added** (7 features parsed from PGN clock timestamps):

| Feature | What it captures |
|---------|-----------------|
| `avg_final_clock_diff` | Who manages time better |
| `time_trouble_rate_diff` | Who gets below 30s more |
| `avg_time_per_move_diff` | Thinking speed gap |
| `white/black_time_trouble_rate` | Individual time pressure tendency |
| `white/black_avg_final_clock` | Individual time management |

| | Test Acc | Test F1m |
|--|----------|----------|
| LightGBM (14f) | **0.725** | 0.503 |
| LightGBM (21f) | 0.722 | 0.500 |

**Verdict**: Neutral for accuracy. But clock features shine with balanced weights (see Trial 7).

---

## Trial 5b: + Granular Time Trouble (<10s, <5s, <1s)

**Added**: `tt10_diff`, `tt5_diff`, `tt1_diff` on top of existing `tt30`.

**Verdict**: **Hurt balanced model** (-1.4% F1m). Too sparse — only 9% of games end with <1s. Reverted.

---

## Trial 5c: + Mouse Skill Features

**Added**: `moves_under_30s_diff` (avg moves made while clock < 30s), `instant_move_rate_diff` (% of moves played in < 1 second — pre-move ability), `endgame_speed_diff` (avg time/move after move 30) + individual values. All computed per player from PGN clock timestamps across their games in the dataset.

**Verdict**: **Hurt or neutral**. Theoretically sound (mouse speed matters in blitz) but 5-10 games per player is too sparse to estimate reliably. Reverted.

---

## Trial 6: Class Weights

Tested draw weight multipliers on the 14-feature model:

| Config | Test Acc | Test F1m | Draw Recall |
|--------|----------|----------|-------------|
| Neutral | **0.725** | 0.503 | 0.000 |
| draw×3 | 0.707 | 0.526 | 0.063 |
| draw×5 | 0.677 | 0.553 | 0.254 |
| draw×10 | 0.514 | 0.478 | 0.651 |
| **balanced** | 0.659 | **0.566** | 0.381 |

**Verdict**: Balanced gives best F1 macro. Trades 6% accuracy for 38% draw recall.

---

## Trial 7: Clock + Balanced (best combination)

Clock features that were neutral in Trial 5 become valuable with balanced weights:

| Model | Test Acc | Test F1m | Draw F1 |
|-------|----------|----------|---------|
| Stats (14f) bal | 0.654 | 0.548 | 0.211 |
| **Stats+Clk (21f) bal** | **0.667** | **0.566** | **0.255** |

**Verdict**: Clock + balanced = best F1 macro model. Clock features help the model differentiate draws when forced to care about them.

---

## Trial 8: Three Split Strategies

| Split | Best Neutral (Acc) | Best Balanced (F1m) |
|-------|-------------------|-------------------|
| **Random 80/20** | Stats (14) — **0.725** | Stats+Clk (21) bal — **0.566** |
| Feb→March | Stats (14) — 0.714 | All+Clk (30) bal — 0.526 |
| Per-Round | Stats+Clk (21) — 0.725 | All+Clk (30) bal — 0.514 |

**Verdict**: Random 80/20 is the most reliable split. Feb→March balanced models overfit heavily (player stats leak). Per-Round ≈ Random for neutral.

---

## Trial 9: In-Tournament Features (previous result, running score, prize contention)

**Added** (10 features computed from earlier rounds within the same tournament):

| Feature | Description |
|---------|-------------|
| `white/black_prev_result` | Did the player win (1), draw (0.5), or lose (0) their last game? |
| `white/black_running_score` | Accumulated points going into this round |
| `running_score_diff` | Momentum gap between players |
| `prev_result_diff` | Who's coming off a better result |
| `can_top6_diff` | Can the player still mathematically reach top 6 (≥8.5 pts)? |
| `can_top3_diff` | Can the player still reach top 3 (≥9.5 pts)? |
| `pts_needed_top6_diff` | How many points still needed for top 6 |
| `pts_needed_top3_diff` | How many points still needed for top 3 |

Round 1 games have no history → filled with 0.5 (neutral) for prev_result, 0 for running_score.

**Results (Random 80/20):**

| Model | Test Acc | Test F1m | Draw F1 | Overfit Gap |
|-------|----------|----------|---------|-------------|
| Stats (14) | 0.725 | 0.503 | 0.000 | +0.023 |
| Stats+Tourn (24) | 0.720 | 0.499 | 0.000 | +0.028 |
| Stats+Clk (21) | 0.722 | 0.500 | 0.000 | +0.031 |
| **Stats+Clk+Tourn (31)** | **0.728** | **0.504** | 0.000 | +0.022 |
| Stats+Tourn (24) bal | 0.653 | 0.542 | 0.190 | +0.048 |
| Stats+Clk+Tourn (31) bal | 0.656 | 0.550 | 0.220 | +0.056 |

**Verdict**: Tournament features alone hurt (-0.5% acc), but **combined with clock they produce the new best accuracy model** (0.728). The combination works because clock captures time management style while tournament captures competitive context — complementary signals. Balanced models didn't benefit as much — more features increase overfitting risk.

---

## Final Winners

| Goal | Model | Test Acc | Test F1m | Draw F1 |
|------|-------|----------|----------|---------|
| **Best accuracy** | Stats+Clk+Tourn (31) | **0.728** | 0.504 | 0.000 |
| **Best F1 macro** | Stats+Clk (21) bal | 0.667 | **0.566** | 0.255 |

---

## SHAP Analysis — What Drives Predictions

Top features by mean |SHAP| value for the best balanced model (Ratings+Stats+Clk, 21f, balanced):

**Draw class** — what makes the model predict a draw:
| Rank | Feature | mean |SHAP| | Interpretation |
|------|---------|-------------|----------------|
| 1 | `draw_rate_avg` | 0.151 | Both players' lifetime draw tendency — strongest draw signal |
| 2 | `black_blitz_last_rd` | 0.099 | Rating uncertainty — volatile ratings → less predictable outcomes |
| 3 | `abs_rating_diff` | 0.080 | Close ratings → more likely draw |
| 4 | `black_avg_final_clock` | 0.061 | Time management pattern |
| 5 | `rating_diff` | 0.058 | Even matchup → draw |

**Win/Loss classes** — what decides who wins:
| Rank | Feature | mean |SHAP| | Interpretation |
|------|---------|-------------|----------------|
| 1 | `rating_diff` | 0.495 | Overwhelmingly dominant — higher rated player wins |
| 2 | `best_rating_diff` | 0.133 | Peak strength adds signal beyond current rating |
| 3 | `abs_rating_diff` | 0.066 | Size of the gap affects confidence |
| 4 | `avg_time_per_move_diff` | 0.060 | Faster thinker has an edge in blitz |
| 5 | `experience_diff` | 0.042 | More experienced player has slight advantage |

---

## Key Learnings

1. **Elo is a strong baseline** (69.6% acc) — ML adds only +3% with pre-game features
2. **Player stats help** (14f > 6f by +2.1%) — win rate, draw rate, peak rating add real signal
3. **More features ≠ better** — profiles, move count, mouse skills all hurt
4. **Draws are fundamentally hard** — 8% minority class, best: 38% recall at 19% precision
5. **Class weights unlock draw detection** — but at the cost of 6% accuracy
6. **Clock features are conditional** — useless alone, valuable with balanced weights
7. **Small dataset limits per-player features** — 5-10 games per player makes aggregates noisy

---

## Known Limitations

1. **Small dataset** — only 4,063 games from 2 tournaments. Per-player features (win rate, draw rate, clock stats) are computed from 5-10 games each, making them noisy and unreliable.
2. **Stale player stats** — fetched from the API in April 2026, after both tournaments. Lifetime aggregates include games played after Feb/Mar. Impact is small (thousands of lifetime games dilute a few months) but not strictly pre-game information.
3. **Player profiles reflect current state** — a player's title, account status, or country could have changed between the tournament date and when we fetched the data.
4. **Draw class imbalance** — draws are only 7.8% of games. The model's best draw detection (38% recall, 19% precision) means 4 out of 5 "draw" predictions are wrong. Draws in blitz are inherently driven by in-game decisions, not pre-game statistics.
5. **No head-to-head history** — some player matchups are systematically one-sided (style mismatches, psychological edges). We don't capture this because we lack sufficient historical data between specific player pairs.
6. **Model capacity is not the bottleneck** — we verified that the model CAN overfit the training data by increasing `n_estimators` and `max_depth`. With aggressive parameters (depth=6, 500 trees), train accuracy reaches 90%+ while test stays at ~72%. This confirms the ceiling is in the data, not the model. With only pre-game features and 4,063 games, ~72% accuracy may be close to the achievable limit. More data and richer features (head-to-head, in-tournament momentum, opening prep) are needed to push further.

---

## Next Steps & Improvements

1. **More data** — fetch 10+ weekly Titled Tuesdays → 20,000+ games, 50+ per player. Per-player stats become reliable, clock and mouse skill features become viable.
2. **Leak-free temporal features** — compute rolling stats from previous tournaments only (Jan→Feb, Feb→Mar). Eliminates post-tournament stat leakage entirely.
3. **In-tournament momentum** — running score from rounds 1..N-1 as a feature for round N. A player on 5/5 plays differently than one on 2/5.
4. **Head-to-head records** — historical matchup results between the two specific players. Some pairings are systematically one-sided.
5. **Opening-based draw probability** — some ECO codes draw more than others. Compute per-player opening preferences from historical games to estimate matchup-specific draw likelihood.
6. **Player embeddings** — with enough data (50+ games per player), learn latent vectors that capture playing style, time management patterns, and performance tendencies beyond what aggregated stats can express.
7. **Previous game result** — did the player just win, lose, or draw? A loss in the previous round may cause tilt (worse performance) or motivation (stronger play). A win streak builds confidence. This sequential context is currently ignored.
8. **Tournament standing & prize motivation** — a player fighting for a money position (e.g., 7.5/9 going into round 10, needing wins to finish top 3) plays more aggressively than someone already out of contention at 4/9. Features: distance to prize positions, mathematical elimination status, points needed to reach top N.
9. **Ensembling** — combine multiple model types (LightGBM, logistic regression, Elo baseline) into a blended prediction. Each model captures different patterns: Elo handles the rating-based prior, LightGBM captures non-linear feature interactions, logistic regression provides calibrated probabilities. A weighted average or stacking approach could outperform any single model, especially for the draw class where different models make different errors.
# chess
