import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('powerball_with_era.csv', parse_dates=['date'])
era_caps = {
    'A: 1992-1997 (5/45+1/45)': (45,45),
    'B: 1997-2002 (5/49+1/42)': (49,42),
    'C: 2002-2005 (5/53+1/42)': (53,42),
    'D: 2005-2009 (5/55+1/42)': (55,42),
    'E: 2009-2012 (5/59+1/39)': (59,39),
    'F: 2012-2015 (5/59+1/35)': (59,35),
    'G: 2015-present (5/69+1/26)': (69,26),
}

results = []
for era, (w_cap, p_cap) in era_caps.items():
    sub = df[df['era']==era]
    n = len(sub)
    if n < 10: continue

    # --- White ball frequency uniformity (chi-square goodness of fit) ---
    white_draws = sub[['num1','num2','num3','num4','num5']].values.flatten()
    obs_white = pd.Series(white_draws).value_counts().reindex(range(1, w_cap+1), fill_value=0)
    exp_white = np.full(w_cap, len(white_draws)/w_cap)
    chi2_w, p_w = stats.chisquare(obs_white, exp_white)

    # --- Powerball frequency uniformity ---
    obs_pb = sub['powerball'].value_counts().reindex(range(1, p_cap+1), fill_value=0)
    exp_pb = np.full(p_cap, n/p_cap)
    chi2_p, p_p = stats.chisquare(obs_pb, exp_pb)

    results.append(dict(era=era, n=n, w_cap=w_cap, p_cap=p_cap,
                         white_chi2=chi2_w, white_p=p_w,
                         pb_chi2=chi2_p, pb_p=p_p,
                         white_hottest=int(obs_white.idxmax()), white_hottest_n=int(obs_white.max()),
                         white_coldest=int(obs_white.idxmin()), white_coldest_n=int(obs_white.min()),
                         white_expected_each=round(len(white_draws)/w_cap,1)))

res = pd.DataFrame(results)
pd.set_option('display.width', 150)
print(res[['era','n','white_chi2','white_p','pb_chi2','pb_p']].to_string(index=False))
print()
print(res[['era','white_expected_each','white_hottest','white_hottest_n','white_coldest','white_coldest_n']].to_string(index=False))
res.to_csv('frequency_results.csv', index=False)
