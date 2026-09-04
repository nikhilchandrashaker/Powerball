import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score

df = pd.read_csv('powerball_with_features.csv', parse_dates=['date'])

# Focus on current era (largest, most relevant sample): G, 5/69+1/26
CAP = 69
sub = df[df['era']=='G: 2015-present (5/69+1/26)'].sort_values('date').reset_index(drop=True)
whitecols = ['num1','num2','num3','num4','num5']
n_draws = len(sub)
print(f"Building long-format panel: {n_draws} draws x {CAP} numbers = {n_draws*CAP} rows")

# Build long panel: one row per (draw_index, number), features computed ONLY from history
# strictly before that draw (no leakage).
last_seen = {k: -1 for k in range(1, CAP+1)}
appear_count = {k: 0 for k in range(1, CAP+1)}
last3_appear = {k: [] for k in range(1, CAP+1)}  # track recent appearance flags for streak feature

rows = []
for i in range(n_draws):
    drawn = set(int(sub.loc[i,c]) for c in whitecols)
    prev_drawn = set(int(sub.loc[i-1,c]) for c in whitecols) if i > 0 else set()
    for k in range(1, CAP+1):
        gap = (i - last_seen[k]) if last_seen[k] >= 0 else i
        rolling_freq = appear_count[k] / i if i > 0 else 0.0
        recent_streak = sum(last3_appear[k][-3:])  # appearances in last up to 3 draws
        prev_appeared = 1 if k in prev_drawn else 0
        rows.append((i, k, gap, rolling_freq, recent_streak, prev_appeared, 1 if k in drawn else 0))
    # update state AFTER building features for this draw
    for k in range(1, CAP+1):
        last3_appear[k].append(1 if k in drawn else 0)
    for k in drawn:
        last_seen[k] = i
        appear_count[k] += 1

panel = pd.DataFrame(rows, columns=['draw_idx','number','gap','rolling_freq','recent_streak','prev_appeared','target'])
panel.to_csv('ml_panel.csv', index=False)
print(panel['target'].mean(), "= empirical P(appear) per row, theoretical =", 5/CAP)

# Time-series CV over draw_idx (never train on future draws)
draw_ids = panel['draw_idx'].unique()
draw_ids.sort()
tscv = TimeSeriesSplit(n_splits=5)
feat_cols = ['gap','rolling_freq','recent_streak','prev_appeared']

results = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(draw_ids)):
    train_draws = set(draw_ids[train_idx])
    test_draws = set(draw_ids[test_idx])
    train = panel[panel['draw_idx'].isin(train_draws)]
    test = panel[panel['draw_idx'].isin(test_draws)]
    if train['target'].sum() == 0 or test['target'].sum() == 0:
        continue

    model = LogisticRegression(max_iter=1000)
    model.fit(train[feat_cols], train['target'])
    proba = model.predict_proba(test[feat_cols])[:,1]

    baseline_p = 5/CAP
    baseline_proba = np.full(len(test), baseline_p)

    ll_model = log_loss(test['target'], proba, labels=[0,1])
    ll_base = log_loss(test['target'], baseline_proba, labels=[0,1])
    auc_model = roc_auc_score(test['target'], proba)
    auc_base = 0.5  # constant prediction => AUC undefined/0.5 by definition of no discrimination

    results.append(dict(fold=fold, n_train_draws=len(train_draws), n_test_draws=len(test_draws),
                         logloss_model=ll_model, logloss_baseline=ll_base,
                         auc_model=auc_model, coef=dict(zip(feat_cols, model.coef_[0]))))

for r in results:
    print(f"Fold {r['fold']}: train_draws={r['n_train_draws']:4d} test_draws={r['n_test_draws']:3d} "
          f"| logloss model={r['logloss_model']:.5f} vs baseline={r['logloss_baseline']:.5f} "
          f"| AUC model={r['auc_model']:.4f} (baseline=0.5000)")
    print(f"    coefficients: {r['coef']}")

import json
with open('ml_results.json','w') as f:
    json.dump([{k:v for k,v in r.items() if k!='coef'} | {'coef':r['coef']} for r in results], f, indent=2)
