import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('powerball_with_features.csv', parse_dates=['date'])
freq = pd.read_csv('frequency_results.csv')
gaps = pd.read_csv('gap_results.csv')

era_caps = {
    'A: 1992-1997 (5/45+1/45)': 45,'B: 1997-2002 (5/49+1/42)': 49,'C: 2002-2005 (5/53+1/42)': 53,
    'D: 2005-2009 (5/55+1/42)': 55,'E: 2009-2012 (5/59+1/39)': 59,'F: 2012-2015 (5/59+1/35)': 59,
    'G: 2015-present (5/69+1/26)': 69,
}
rng = np.random.default_rng(7)

rows = []
for era, cap in era_caps.items():
    sub = df[df['era']==era].sort_values('date').reset_index(drop=True)
    n = len(sub)
    if n < 60: continue

    # F: frequency uniformity p-value (white balls, from chi-square goodness of fit)
    p_F = freq.loc[freq['era']==era, 'white_p'].values[0]

    # G: gap independence p-value (white balls, logistic regression coef p-value)
    p_G = gaps.loc[(gaps['era']==era) & (gaps['ball']=='white'), 'logit_p'].values[0]

    # A: autocorrelation - lag-1 Pearson correlation of the draw-sum series, tested for significance
    sums = sub['sum'].values
    r, p_A = stats.pearsonr(sums[:-1], sums[1:])

    # C: combination distribution - KS test comparing observed sum distribution to simulated random draws
    sims = np.array([rng.choice(np.arange(1, cap+1), size=5, replace=False).sum() for _ in range(20000)])
    ks_stat, p_C = stats.ks_2samp(sums, sims)

    # T: temporal stability within era - chi-square comparing first-half vs second-half white-ball frequency
    half = n // 2
    first = sub.iloc[:half][['num1','num2','num3','num4','num5']].values.flatten()
    second = sub.iloc[half:][['num1','num2','num3','num4','num5']].values.flatten()
    obs_first = pd.Series(first).value_counts().reindex(range(1,cap+1), fill_value=0)
    obs_second = pd.Series(second).value_counts().reindex(range(1,cap+1), fill_value=0)
    # scale second to same total as first for a 2-sample chi-square (contingency)
    table = np.array([obs_first.values, obs_second.values])
    # remove all-zero columns to avoid errors, add small constant
    chi2_T, p_T, dof_T, _ = stats.chi2_contingency(table + 1)

    RANDOM = 100 * np.mean([p_F, p_G, p_A, p_C, p_T])
    rows.append(dict(era=era, n=n, F_p=p_F, G_p=p_G, A_p=p_A, C_p=p_C, T_p=p_T, RANDOM_score=RANDOM))

res = pd.DataFrame(rows).sort_values('RANDOM_score', ascending=False)
pd.set_option('display.width',150)
print(res.round(4).to_string(index=False))
res.to_csv('random_scores.csv', index=False)
