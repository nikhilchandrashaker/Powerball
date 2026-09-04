import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('powerball_with_era.csv', parse_dates=['date'])

# Focus gap/overdue analysis on the current era (most data, most relevant to "is it random NOW")
# but also run it globally within-era for robustness.
era_caps = {
    'A: 1992-1997 (5/45+1/45)': (45,45),
    'B: 1997-2002 (5/49+1/42)': (49,42),
    'C: 2002-2005 (5/53+1/42)': (53,42),
    'D: 2005-2009 (5/55+1/42)': (55,42),
    'E: 2009-2012 (5/59+1/39)': (59,39),
    'F: 2012-2015 (5/59+1/35)': (59,35),
    'G: 2015-present (5/69+1/26)': (69,26),
}

def gap_overdue_test(sub, cap, col_is_powerball=False):
    """For each draw i (after a warmup), compute each number's current gap
    (draws since last seen), then check: did numbers with longer gaps appear
    in draw i at a higher rate than expected? Logistic regression: P(appear) ~ gap."""
    sub = sub.reset_index(drop=True)
    n = len(sub)
    last_seen = {k: -1 for k in range(1, cap+1)}
    rows = []
    for i in range(n):
        if col_is_powerball:
            drawn = {int(sub.loc[i,'powerball'])}
        else:
            drawn = set(int(sub.loc[i,c]) for c in ['num1','num2','num3','num4','num5'])
        for k in range(1, cap+1):
            gap = i - last_seen[k] if last_seen[k] >= 0 else i  # gap since last seen (or since start)
            rows.append((gap, 1 if k in drawn else 0))
        for k in drawn:
            last_seen[k] = i
    g = pd.DataFrame(rows, columns=['gap','appeared'])
    # drop the very first ~cap draws where "gap" is just confounded with warmup
    warmup = cap * 2
    g2 = g[g['gap'] >= 0]
    # logistic regression coefficient sign/significance on gap -> appearance
    import statsmodels.api as sm
    X = sm.add_constant(g2['gap'])
    model = sm.Logit(g2['appeared'], X).fit(disp=0)
    coef = model.params['gap']
    pval = model.pvalues['gap']
    # also simple correlation check: mean gap for appeared vs not
    mean_gap_appeared = g2[g2['appeared']==1]['gap'].mean()
    mean_gap_not = g2[g2['appeared']==0]['gap'].mean()
    return dict(n_obs=len(g2), logit_coef=coef, logit_p=pval,
                mean_gap_appeared=mean_gap_appeared, mean_gap_not_appeared=mean_gap_not)

results = []
for era, (w_cap, p_cap) in era_caps.items():
    sub = df[df['era']==era]
    if len(sub) < 60: continue
    r_white = gap_overdue_test(sub, w_cap, col_is_powerball=False)
    r_pb = gap_overdue_test(sub, p_cap, col_is_powerball=True)
    results.append(dict(era=era, ball='white', **r_white))
    results.append(dict(era=era, ball='powerball', **r_pb))

res = pd.DataFrame(results)
pd.set_option('display.width',150)
print(res.to_string(index=False))
res.to_csv('gap_results.csv', index=False)
