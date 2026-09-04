import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

df = pd.read_csv('powerball_with_era.csv', parse_dates=['date'])
era_caps = {
    'A: 1992-1997 (5/45+1/45)': 45,'B: 1997-2002 (5/49+1/42)': 49,'C: 2002-2005 (5/53+1/42)': 53,
    'D: 2005-2009 (5/55+1/42)': 55,'E: 2009-2012 (5/59+1/39)': 59,'F: 2012-2015 (5/59+1/35)': 59,
    'G: 2015-present (5/69+1/26)': 69,
}
whitecols = ['num1','num2','num3','num4','num5']

def combo_features(row):
    nums = sorted(row[whitecols].astype(int).tolist())
    odd = sum(1 for x in nums if x % 2 == 1)
    total = sum(nums)
    rng = nums[-1] - nums[0]
    consec = sum(1 for a,b in zip(nums, nums[1:]) if b - a == 1)
    decades = len(set((x-1)//10 for x in nums))
    return pd.Series({'odd_count': odd, 'sum': total, 'range': rng,
                       'consec_pairs': consec, 'n_decades': decades})

feat = df.apply(combo_features, axis=1)
df2 = pd.concat([df, feat], axis=1)

# repeats from previous draw (within same era only, to avoid crossing rule changes)
df2 = df2.sort_values('date').reset_index(drop=True)
repeats = [np.nan]
for i in range(1, len(df2)):
    if df2.loc[i,'era'] != df2.loc[i-1,'era']:
        repeats.append(np.nan)
        continue
    prev = set(df2.loc[i-1, whitecols].astype(int))
    cur = set(df2.loc[i, whitecols].astype(int))
    repeats.append(len(prev & cur))
df2['repeats_from_prev'] = repeats

print("=== Odd/even split distribution (all eras combined, count of odd among 5) ===")
print(df2['odd_count'].value_counts().sort_index())
print()

print("=== Repeats from previous draw: observed vs random-baseline expectation ===")
for era, cap in era_caps.items():
    sub = df2[df2['era']==era].dropna(subset=['repeats_from_prev'])
    if len(sub) < 20: continue
    obs_mean = sub['repeats_from_prev'].mean()
    # theoretical expected number of matches between two independent 5-of-cap draws (hypergeometric overlap)
    # E[overlap] = 5 * 5 / cap
    exp_mean = 5*5/cap
    print(f"{era}: n={len(sub)}, observed mean repeats={obs_mean:.3f}, theoretical expected={exp_mean:.3f}")
print()

print("=== Sum of five white balls: observed vs simulated-random mean/std ===")
rng = np.random.default_rng(42)
for era, cap in era_caps.items():
    sub = df2[df2['era']==era]
    if len(sub) < 20: continue
    obs_mean, obs_std = sub['sum'].mean(), sub['sum'].std()
    sims = np.array([rng.choice(np.arange(1,cap+1), size=5, replace=False).sum() for _ in range(20000)])
    print(f"{era}: n={len(sub)}, obs sum mean={obs_mean:.1f} (sd {obs_std:.1f}), "
          f"sim mean={sims.mean():.1f} (sd {sims.std():.1f})")

print()
print("=== Odd count: observed vs theoretical binomial-ish (hypergeometric) distribution, era G ===")
subG = df2[df2['era']=='G: 2015-present (5/69+1/26)']
obs = subG['odd_count'].value_counts().sort_index()
# simulate
capG=69
sims = []
for _ in range(50000):
    draw = rng.choice(np.arange(1,capG+1), size=5, replace=False)
    sims.append(sum(1 for x in draw if x%2==1))
sim_counts = pd.Series(sims).value_counts(normalize=True).sort_index()
obs_norm = obs/obs.sum()
comp = pd.DataFrame({'observed_frac': obs_norm, 'simulated_frac': sim_counts}).fillna(0)
print(comp)

df2.to_csv('powerball_with_features.csv', index=False)
