import pandas as pd
import numpy as np

df = pd.read_csv('powerball_all.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)

# Verified rule-change dates (official MUSL changes), refining the user's 4-bucket table into
# the actual 7 distinct rulesets that existed historically.
era_bounds = [
    ('1992-04-22', '1997-11-04', 'A: 1992-1997 (5/45+1/45)', 45, 45),
    ('1997-11-05', '2002-10-08', 'B: 1997-2002 (5/49+1/42)', 49, 42),
    ('2002-10-09', '2005-08-30', 'C: 2002-2005 (5/53+1/42)', 53, 42),
    ('2005-08-31', '2009-01-06', 'D: 2005-2009 (5/55+1/42)', 55, 42),
    ('2009-01-07', '2012-01-17', 'E: 2009-2012 (5/59+1/39)', 59, 39),
    ('2012-01-18', '2015-10-06', 'F: 2012-2015 (5/59+1/35)', 59, 35),
    ('2015-10-07', '2026-09-02', 'G: 2015-present (5/69+1/26)', 69, 26),
]

def assign_era(d):
    for start, end, label, w, p in era_bounds:
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            return label
    return 'UNKNOWN'

df['era'] = df['date'].apply(assign_era)
print(df['era'].value_counts().sort_index())
print()
# sanity check: max white/pb value per era should not exceed bound
for start, end, label, w, p in era_bounds:
    sub = df[df['era']==label]
    if len(sub)==0: continue
    wmax = sub[['num1','num2','num3','num4','num5']].max().max()
    pmax = sub['powerball'].max()
    print(f"{label}: n={len(sub)}, white_max_seen={wmax} (cap {w}), pb_max_seen={pmax} (cap {p})")

df.to_csv('powerball_with_era.csv', index=False)
