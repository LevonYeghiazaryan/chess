"""
main_model.py — Chess game outcome prediction pipeline.

Trains Elo baseline + LightGBM models across 3 split strategies,
with neutral and balanced class weights. Generates SHAP plots.

Usage: python main_model.py
"""

import os
import re
import time
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

LGB_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    num_leaves=8, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, random_state=42, verbose=-1,
)


# ═══════════════════════════════════════════════════════════════════════════
#  1. LOAD & TARGET
# ═══════════════════════════════════════════════════════════════════════════

df = pd.read_csv(os.path.join(DATA_DIR, 'raw_games.csv'))

df['outcome'] = np.where(
    df['white.result'] == 'win', 'white_win',
    np.where(df['black.result'] == 'win', 'white_loss', 'draw')
)

print(f'Loaded: {df.shape[0]} games')
print(f'Target: {df["outcome"].value_counts().to_dict()}')


# ═══════════════════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

# --- Rating features ---
df['rating_diff'] = df['white.rating'] - df['black.rating']
df['abs_rating_diff'] = df['rating_diff'].abs()
df['is_march'] = (df['tournament'] == 'mar_2026').astype(int)

# --- Player stats ---
stats = pd.read_csv(os.path.join(DATA_DIR, 'player_stats.csv'))
stats['username'] = stats['username'].str.lower()

for side in ['white', 'black']:
    df[f'{side}_username_lower'] = df[f'{side}.username'].str.lower()
    merged = df[[f'{side}_username_lower']].merge(
        stats.rename(columns={'username': f'{side}_username_lower'}),
        on=f'{side}_username_lower', how='left')
    for col in ['blitz_best_rating', 'blitz_win_rate', 'blitz_draw_rate',
                'blitz_last_rd', 'blitz_total_games']:
        df[f'{side}_{col}'] = merged[col].values

df['best_rating_diff'] = df['white_blitz_best_rating'] - df['black_blitz_best_rating']
df['win_rate_diff'] = df['white_blitz_win_rate'] - df['black_blitz_win_rate']
df['draw_rate_diff'] = df['white_blitz_draw_rate'] - df['black_blitz_draw_rate']
df['draw_rate_avg'] = (df['white_blitz_draw_rate'] + df['black_blitz_draw_rate']) / 2
df['experience_diff'] = np.log1p(df['white_blitz_total_games']) - np.log1p(df['black_blitz_total_games'])
df['rating_gap_diff'] = (df['white_blitz_best_rating'] - df['white.rating']) - \
                        (df['black_blitz_best_rating'] - df['black.rating'])

# --- Player profiles ---
profiles = pd.read_csv(os.path.join(DATA_DIR, 'player_profiles.csv'))
profiles['username'] = profiles['username'].str.lower()
title_map = {'GM': 6, 'WGM': 5, 'IM': 4, 'WIM': 3, 'FM': 2, 'WFM': 2, 'CM': 1, 'WCM': 1, 'NM': 1}
profiles['title_strength'] = profiles['title'].map(title_map).fillna(0)
profiles['account_age_years'] = (time.time() - profiles['joined']) / (365.25 * 24 * 3600)

for side in ['white', 'black']:
    merged = df[[f'{side}_username_lower']].merge(
        profiles[['username', 'title_strength', 'account_age_years', 'is_streamer', 'status']].rename(
            columns={'username': f'{side}_username_lower',
                     'title_strength': f'{side}_title_strength',
                     'account_age_years': f'{side}_account_age',
                     'is_streamer': f'{side}_is_streamer',
                     'status': f'{side}_status'}),
        on=f'{side}_username_lower', how='left')
    df[f'{side}_title_strength'] = merged[f'{side}_title_strength'].values
    df[f'{side}_account_age'] = merged[f'{side}_account_age'].values
    df[f'{side}_is_streamer'] = merged[f'{side}_is_streamer'].astype(int).values
    df[f'{side}_is_premium'] = (merged[f'{side}_status'] == 'premium').astype(int).values

df['title_strength_diff'] = df['white_title_strength'] - df['black_title_strength']
df['account_age_diff'] = df['white_account_age'] - df['black_account_age']

# --- Move count from FEN ---
df['move_count'] = df['fen'].str.split().str[-1].astype(int)
move_agg = pd.concat([
    df[['white_username_lower', 'move_count']].rename(columns={'white_username_lower': 'u'}),
    df[['black_username_lower', 'move_count']].rename(columns={'black_username_lower': 'u'}),
]).groupby('u')['move_count'].mean().reset_index()
move_agg.columns = ['u', 'avg_game_length']

for side in ['white', 'black']:
    merged = df[[f'{side}_username_lower']].merge(
        move_agg.rename(columns={'u': f'{side}_username_lower', 'avg_game_length': f'{side}_avg_moves'}),
        on=f'{side}_username_lower', how='left')
    df[f'{side}_avg_moves'] = merged[f'{side}_avg_moves'].values
df['avg_moves_diff'] = df['white_avg_moves'] - df['black_avg_moves']

# --- Clock features from PGN ---
def parse_clocks(pgn):
    clks = re.findall(r'\[%clk (\d+):(\d+):(\d+\.?\d*)\]', pgn)
    secs = [int(h)*3600 + int(m)*60 + float(s) for h, m, s in clks]
    return secs[0::2], secs[1::2]

clock_rows = []
for _, row in df.iterrows():
    wc, bc = parse_clocks(row['pgn'])
    for prefix, clks in [('w', wc), ('b', bc)]:
        if len(clks) >= 2:
            clock_rows.append({
                'u': row[f'{"white" if prefix=="w" else "black"}_username_lower'],
                'final': clks[-1],
                'avg_tpm': (300 - clks[-1]) / len(clks),
                'low_time': int(clks[-1] < 30),
            })

clock_agg = pd.DataFrame(clock_rows).groupby('u').mean().reset_index()
for side in ['white', 'black']:
    merged = df[[f'{side}_username_lower']].merge(
        clock_agg.rename(columns={'u': f'{side}_username_lower',
                                   'final': f'{side}_avg_final_clock',
                                   'low_time': f'{side}_time_trouble_rate',
                                   'avg_tpm': f'{side}_avg_time_per_move'}),
        on=f'{side}_username_lower', how='left')
    for col in ['avg_final_clock', 'time_trouble_rate', 'avg_time_per_move']:
        df[f'{side}_{col}'] = merged[f'{side}_{col}'].values

df['avg_final_clock_diff'] = df['white_avg_final_clock'] - df['black_avg_final_clock']
df['time_trouble_rate_diff'] = df['white_time_trouble_rate'] - df['black_time_trouble_rate']
df['avg_time_per_move_diff'] = df['white_avg_time_per_move'] - df['black_avg_time_per_move']

print(f'Features engineered. Nulls in stats: {df["white_blitz_win_rate"].isna().sum()}')


# ═══════════════════════════════════════════════════════════════════════════
#  3. FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════

CLOCK = ['avg_final_clock_diff', 'time_trouble_rate_diff', 'avg_time_per_move_diff',
         'white_time_trouble_rate', 'black_time_trouble_rate',
         'white_avg_final_clock', 'black_avg_final_clock']

F_BASE = ['white.rating', 'black.rating', 'rating_diff', 'abs_rating_diff', 'round', 'is_march']

F_STATS = F_BASE + ['win_rate_diff', 'draw_rate_diff', 'draw_rate_avg',
                     'best_rating_diff', 'experience_diff', 'rating_gap_diff',
                     'white_blitz_last_rd', 'black_blitz_last_rd']

F_FULL = F_STATS + ['white_avg_moves', 'black_avg_moves', 'avg_moves_diff',
                     'title_strength_diff', 'account_age_diff',
                     'white_is_streamer', 'black_is_streamer',
                     'white_is_premium', 'black_is_premium']

F_BASE_CLK = F_BASE + CLOCK
F_STATS_CLK = F_STATS + CLOCK
F_FULL_CLK = F_FULL + CLOCK

FEATURE_SETS = [
    ('Ratings (6)',          F_BASE),
    ('Ratings+Stats (14)',   F_STATS),
    ('All Features (23)',    F_FULL),
    ('Ratings+Clk (13)',     F_BASE_CLK),
    ('Ratings+Stats+Clk (21)', F_STATS_CLK),
    ('All+Clk (30)',         F_FULL_CLK),
]

print(f'Feature sets: {", ".join(f"{n}" for n, _ in FEATURE_SETS)}')


# ═══════════════════════════════════════════════════════════════════════════
#  4. MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def elo_predict(X, draw_rate):
    elo = 1 / (1 + 10 ** ((X['black.rating'].values - X['white.rating'].values) / 400))
    p = np.column_stack([np.full_like(elo, draw_rate),
                         (1 - elo) * (1 - draw_rate),
                         elo * (1 - draw_rate)])
    p /= p.sum(axis=1, keepdims=True)
    classes = np.array(['draw', 'white_loss', 'white_win'])
    return classes[np.argmax(p, axis=1)], p, classes


def train_lgb(Xtr, ytr, Xte, yte, feats, label, cw=None):
    params = {**LGB_PARAMS, **({"class_weight": cw} if cw else {})}
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr[feats], ytr)
    return {'model': m, 'label': label, 'features': feats,
            'train_preds': m.predict(Xtr[feats]), 'train_proba': m.predict_proba(Xtr[feats]),
            'test_preds': m.predict(Xte[feats]), 'test_proba': m.predict_proba(Xte[feats]),
            'classes': m.classes_}


def build_all(Xtr, ytr, Xte, yte):
    dr = (ytr == 'draw').mean()
    ep_tr, epb_tr, ecls = elo_predict(Xtr, dr)
    ep_te, epb_te, _ = elo_predict(Xte, dr)
    models = {'Elo Baseline': {'train_preds': ep_tr, 'test_preds': ep_te,
                                'train_proba': epb_tr, 'test_proba': epb_te, 'classes': ecls}}
    for name, feats in FEATURE_SETS:
        models[name] = train_lgb(Xtr, ytr, Xte, yte, feats, name)
        models[f'{name} bal'] = train_lgb(Xtr, ytr, Xte, yte, feats, f'{name} bal', cw='balanced')
    return models


# ═══════════════════════════════════════════════════════════════════════════
#  5. PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════

CLS = ['draw', 'white_loss', 'white_win']

def print_results(split_data, split_name):
    models, yt_train, yt_test = split_data
    print(f'\n{"#" * 100}')
    print(f'  SPLIT: {split_name}')
    print(f'{"#" * 100}')

    # Overfit table
    print(f'\n{"Model":<28} {"Tr Acc":>7} {"Te Acc":>7} {"Gap":>6} {"Tr LL":>7} {"Te LL":>7} {"Gap":>6} {"Tr F1m":>7} {"Te F1m":>7} {"Gap":>6}')
    print('-' * 100)
    for n, m in models.items():
        ta = accuracy_score(yt_train, m['train_preds'])
        ea = accuracy_score(yt_test, m['test_preds'])
        tl = log_loss(yt_train, m['train_proba'], labels=m['classes'])
        el = log_loss(yt_test, m['test_proba'], labels=m['classes'])
        tf = f1_score(yt_train, m['train_preds'], average='macro')
        ef = f1_score(yt_test, m['test_preds'], average='macro')
        print(f'{n:<28} {ta:>7.4f} {ea:>7.4f} {ta-ea:>+6.3f} {tl:>7.4f} {el:>7.4f} {el-tl:>+6.3f} {tf:>7.4f} {ef:>7.4f} {tf-ef:>+6.3f}')

    # Per-class metrics
    print(f'\n{"Model":<28} {"Class":<13} {"Prec":>6} {"Rec":>6} {"F1":>6} {"N":>5}')
    print('-' * 68)
    for n, m in models.items():
        p, r, f, s = precision_recall_fscore_support(yt_test, m['test_preds'], labels=CLS)
        for i, c in enumerate(CLS):
            print(f'{n if i==0 else "":<28} {c:<13} {p[i]:>6.3f} {r[i]:>6.3f} {f[i]:>6.3f} {int(s[i]):>5}')
        acc = accuracy_score(yt_test, m['test_preds'])
        f1m = f1_score(yt_test, m['test_preds'], average='macro')
        print(f'{"":<28} {"accuracy":<13} {"":>6} {"":>6} {acc:>6.3f} {len(yt_test):>5}')
        print(f'{"":<28} {"macro F1":<13} {"":>6} {"":>6} {f1m:>6.3f} {len(yt_test):>5}')
        print('-' * 68)


# ═══════════════════════════════════════════════════════════════════════════
#  6. RUN ALL SPLITS
# ═══════════════════════════════════════════════════════════════════════════

df_model = df.dropna(subset=F_FULL_CLK + ['outcome']).copy()
all_splits = {}

# Split A: Random 80/20
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
    df_model, df_model['outcome'], test_size=0.2, random_state=42, stratify=df_model['outcome'])
print(f'\nSplit A: Random 80/20 — Train: {len(Xtr_r)}, Test: {len(Xte_r)}')
all_splits['Random 80/20'] = (build_all(Xtr_r, ytr_r, Xte_r, yte_r), ytr_r, yte_r)
print_results(all_splits['Random 80/20'], 'Random 80/20')

# Split B: Feb → March
df_feb = df_model[df_model['tournament'] == 'feb_2026']
df_mar = df_model[df_model['tournament'] == 'mar_2026']
print(f'\nSplit B: Feb→March — Train: {len(df_feb)}, Test: {len(df_mar)}')
all_splits['Feb→March'] = (build_all(df_feb, df_feb['outcome'], df_mar, df_mar['outcome']),
                            df_feb['outcome'], df_mar['outcome'])
print_results(all_splits['Feb→March'], 'Feb→March')

# Split C: Per-round 80/20
tr_parts, te_parts = [], []
for rnd in sorted(df_model['round'].unique()):
    rd = df_model[df_model['round'] == rnd]
    rt, re = train_test_split(rd, test_size=0.2, random_state=42, stratify=rd['outcome'])
    tr_parts.append(rt); te_parts.append(re)
Xtr_rnd, Xte_rnd = pd.concat(tr_parts), pd.concat(te_parts)
print(f'\nSplit C: Per-Round 80/20 — Train: {len(Xtr_rnd)}, Test: {len(Xte_rnd)}')
all_splits['Per-Round'] = (build_all(Xtr_rnd, Xtr_rnd['outcome'], Xte_rnd, Xte_rnd['outcome']),
                            Xtr_rnd['outcome'], Xte_rnd['outcome'])
print_results(all_splits['Per-Round'], 'Per-Round 80/20')


# ═══════════════════════════════════════════════════════════════════════════
#  7. BEST MODELS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print(f'\n{"#" * 100}')
print(f'  BEST MODELS PER SPLIT')
print(f'{"#" * 100}')
print(f'\n{"Split":<16} {"Best Neutral (Acc)":<30} {"Acc":>7} {"Best Balanced (F1m)":<30} {"F1m":>7}')
print('-' * 95)

for sname, (models, _, yt) in all_splits.items():
    best_a, best_av, best_f, best_fv = '', 0, '', 0
    for n, m in models.items():
        a = accuracy_score(yt, m['test_preds'])
        f = f1_score(yt, m['test_preds'], average='macro')
        if a > best_av: best_av, best_a = a, n
        if f > best_fv: best_fv, best_f = f, n
    print(f'{sname:<16} {best_a:<30} {best_av:>7.4f} {best_f:<30} {best_fv:>7.4f}')

# Detailed best models with per-class metrics
print(f'\n{"#" * 100}')
print(f'  BEST MODELS — DETAILED')
print(f'{"#" * 100}')

for sname, (models, _, yt) in all_splits.items():
    # Find best neutral and best balanced
    best_acc_name, best_acc_val = '', 0
    best_f1m_name, best_f1m_val = '', 0
    for n, m in models.items():
        a = accuracy_score(yt, m['test_preds'])
        f = f1_score(yt, m['test_preds'], average='macro')
        if a > best_acc_val: best_acc_val, best_acc_name = a, n
        if f > best_f1m_val: best_f1m_val, best_f1m_name = f, n

    # Also include Elo baseline
    for label in ['Elo Baseline', best_acc_name, best_f1m_name]:
        if label not in models:
            continue
        m = models[label]
        p, r, f, s = precision_recall_fscore_support(yt, m['test_preds'], labels=CLS)
        acc = accuracy_score(yt, m['test_preds'])
        f1m = f1_score(yt, m['test_preds'], average='macro')

        print(f'\n  [{sname}] {label}  —  Acc: {acc:.4f}  F1m: {f1m:.4f}')
        print(f'  {"Class":<13} {"Prec":>8} {"Recall":>8} {"F1":>8}')
        print(f'  {"-"*40}')
        for i, c in enumerate(CLS):
            print(f'  {c:<13} {p[i]:>8.3f} {r[i]:>8.3f} {f[i]:>8.3f}')
        print(f'  {"macro avg":<13} {"":>8} {"":>8} {f1m:>8.3f}')
        print(f'  {"accuracy":<13} {"":>8} {"":>8} {acc:>8.3f}')


# ═══════════════════════════════════════════════════════════════════════════
#  8. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

print(f'\n{"=" * 70}')
print('FEATURE IMPORTANCE (Random split)')
print('=' * 70)

r_models = all_splits['Random 80/20'][0]
for label in ['Ratings+Stats (14)', 'Ratings+Stats+Clk (21) bal']:
    m = r_models[label]
    imp = pd.Series(m['model'].feature_importances_, index=m['features']).sort_values(ascending=False)
    print(f'\n  {label} (top 5):')
    for feat, val in imp.head(5).items():
        print(f'    {feat:25s}  {val}')


# ═══════════════════════════════════════════════════════════════════════════
#  9. SHAP PLOTS
# ═══════════════════════════════════════════════════════════════════════════

print(f'\n{"=" * 70}')
print('SHAP PLOTS')
print('=' * 70)

shap_model = r_models['Ratings+Stats+Clk (21) bal']['model']
shap_feats = F_STATS_CLK
shap_raw = shap_model.predict_proba(Xte_r[shap_feats], pred_contrib=True)
n_cls = len(shap_model.classes_)
n_f = len(shap_feats)
shap_all = np.array(shap_raw).reshape(len(Xte_r), n_cls, n_f + 1)
cls_ordered = list(shap_model.classes_)

# Plot 1: Mean |SHAP| per class
fig, axes = plt.subplots(1, n_cls, figsize=(5*n_cls, 6), sharey=True)
fig.suptitle('Mean |SHAP| Value per Feature (by class)', fontsize=14, fontweight='bold')
for ci, (ax, c) in enumerate(zip(axes, cls_ordered)):
    sv = np.abs(shap_all[:, ci, :-1]).mean(axis=0)
    o = np.argsort(sv)
    ax.barh(np.array(shap_feats)[o], sv[o], color='#2196F3')
    ax.set_xlabel('Mean |SHAP|'); ax.set_title(c)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Beeswarm
fig, axes = plt.subplots(1, n_cls, figsize=(5*n_cls, 6), sharey=True)
fig.suptitle('SHAP Values vs Feature Value', fontsize=14, fontweight='bold')
for ci, (ax, c) in enumerate(zip(axes, cls_ordered)):
    sv = shap_all[:, ci, :-1]
    ms = np.abs(sv).mean(axis=0)
    o = np.argsort(ms)
    for rank, fi in enumerate(o):
        fv = Xte_r[shap_feats].iloc[:, fi].values
        fmin, fmax = fv.min(), fv.max()
        nv = (fv - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(fv)
        yp = np.full(len(Xte_r), rank) + np.random.uniform(-0.3, 0.3, len(Xte_r))
        ax.scatter(sv[:, fi], yp, c=nv, cmap='coolwarm', s=3, alpha=0.5, rasterized=True)
    ax.set_yticks(range(n_f)); ax.set_yticklabels(np.array(shap_feats)[o])
    ax.axvline(0, color='gray', lw=0.5); ax.set_xlabel('SHAP value'); ax.set_title(c)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Summary (stacked)
ms_all = np.abs(shap_all[:, :, :-1]).mean(axis=1).mean(axis=0)
o = np.argsort(ms_all)
fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(n_f)
cc = {'draw': '#9E9E9E', 'white_loss': '#F44336', 'white_win': '#2196F3'}
for ci, c in enumerate(cls_ordered):
    sm = np.abs(shap_all[:, ci, :-1]).mean(axis=0)
    ax.barh(np.array(shap_feats)[o], sm[o], left=bottom[o], label=c, color=cc.get(c, '#666'))
    bottom += sm
ax.set_xlabel('Mean |SHAP|'); ax.set_title('SHAP Summary — All Classes')
ax.legend(title='Class', loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()

print('Saved: shap_importance.png, shap_beeswarm.png, shap_summary.png')
