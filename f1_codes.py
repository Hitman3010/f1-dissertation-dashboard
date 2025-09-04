import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pymc as pm
import arviz as az

DATA_DIR = "F1 data/"

# loading dataset
results = pd.read_csv(DATA_DIR + "results.csv")
drivers = pd.read_csv(DATA_DIR + "drivers.csv")
constructors = pd.read_csv(DATA_DIR + "constructors.csv")
races = pd.read_csv(DATA_DIR + "races.csv")

try:
    status = pd.read_csv(DATA_DIR + "status.csv")
    have_status = True
except Exception:
    have_status = False

merged = (results
          .merge(drivers, on='driverId', how='left')
          .merge(constructors, on='constructorId', how='left')
          .merge(races, on='raceId', how='left'))

# Team and race names
team_name_col = "name_x" if "name_x" in merged.columns else ("name" if "name" in constructors.columns else "constructorRef")
race_name_col = "name_y" if "name_y" in merged.columns else ("name" if "name" in races.columns else "raceId")

# Driver full name
merged['driver_name'] = merged['forename'].astype(str).str.strip() + " " + merged['surname'].astype(str).str.strip()
merged['team'] = merged[team_name_col] if team_name_col in merged.columns else merged['constructorRef']
merged['race_name'] = merged[race_name_col] if race_name_col in merged.columns else merged['raceId'].astype(str)

f1_recent = merged[merged['year'] >= 2000].copy().reset_index(drop=True)

# DNF adjustment
if have_status and {'statusId', 'status'}.issubset(set(status.columns)):
    f1_recent = f1_recent.merge(status[['statusId', 'status']], on='statusId', how='left')
    st = f1_recent['status'].astype(str).str.lower()
    finished_like = st.str.contains('finished') | st.str.contains(r'^\+\d+\s*lap')
    f1_recent['dnf'] = (~finished_like).astype(int)
else:
    finished_ids = {1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} 
    f1_recent['dnf'] = (~f1_recent['statusId'].isin(finished_ids)).astype(int)

# Race metrics
f1_recent = f1_recent.dropna(subset=['grid', 'positionOrder']).copy()
f1_recent['positions_gained'] = f1_recent['grid'].astype(float) - f1_recent['positionOrder'].astype(float)
f1_recent['win'] = (f1_recent['positionOrder'] == 1).astype(int)
f1_recent['podium'] = (f1_recent['positionOrder'] <= 3).astype(int)

# Driver and season grouping
driver_season_summary = f1_recent.groupby(['driver_name', 'year']).agg(
    races_entered=('raceId', 'nunique'),
    points_total=('points', 'sum'),
    wins=('win', 'sum'),
    podiums=('podium', 'sum'),
    avg_grid=('grid', 'mean'),
    avg_finish=('positionOrder', 'mean'),
    total_positions_gained=('positions_gained', 'sum'),
    dnfs=('dnf', 'sum')
).reset_index()

# Teammate comparision
race_data = f1_recent[['year', 'raceId', 'driver_name', 'team', 'positionOrder']].dropna()
pairs = []
for (_, grp) in race_data.groupby(['year', 'raceId', 'team']):
    if len(grp) == 2:  
        gsorted = grp.sort_values('positionOrder')
        winner = gsorted.iloc[0]['driver_name']
        for d in grp['driver_name']:
            pairs.append({
                'year': grp['year'].iloc[0],
                'raceId': grp['raceId'].iloc[0],
                'team': grp['team'].iloc[0],
                'driver': d,
                'beat_teammate': d == winner
            })
teammate_df = pd.DataFrame(pairs)
tt = teammate_df.groupby(['driver', 'year']).agg(
    races_with_teammate=('raceId', 'count'),
    teammate_wins=('beat_teammate', 'sum')
).reset_index()
tt['teammate_win_pct'] = np.where(tt['races_with_teammate'] > 0,
                                  tt['teammate_wins'] / tt['races_with_teammate'], 0.0)

# DPI normalising
dpi_df = driver_season_summary.merge(tt, left_on=['driver_name', 'year'],
                                     right_on=['driver', 'year'], how='left').drop(columns=['driver'])

dpi_df[['races_with_teammate', 'teammate_wins', 'teammate_win_pct']] = \
    dpi_df[['races_with_teammate', 'teammate_wins', 'teammate_win_pct']].fillna(0)

dpi_df['inv_avg_grid'] = -dpi_df['avg_grid']
dpi_df['inv_avg_finish'] = -dpi_df['avg_finish']
dpi_df['inv_dnfs'] = -dpi_df['dnfs']

to_scale = ['points_total', 'wins', 'podiums', 'races_entered',
            'inv_avg_grid', 'inv_avg_finish', 'total_positions_gained', 'inv_dnfs', 'teammate_win_pct']

dpi_df[to_scale] = dpi_df[to_scale].fillna(0)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(dpi_df[to_scale])
for col, arr in zip(to_scale, scaled.T):
    dpi_df[f'norm_{col}'] = arr

weights = {
    'norm_points_total': 0.22,
    'norm_wins': 0.18,
    'norm_podiums': 0.13,
    'norm_races_entered': 0.04,
    'norm_inv_avg_grid': 0.09,
    'norm_inv_avg_finish': 0.09,
    'norm_total_positions_gained': 0.09,
    'norm_inv_dnfs': 0.04,
    'norm_teammate_win_pct': 0.12
}
dpi_df['DPI'] = sum(dpi_df[c] * w for c, w in weights.items())

# For streamlit app
dpi_df.to_csv(DATA_DIR + "dpi_dataset.csv", index=False)

# Bayesian modelling
model_data = f1_recent[['raceId', 'driver_name', 'team', 'points']].dropna().copy()
model_data['driver_id'] = model_data['driver_name'].astype('category').cat.codes
model_data['team_id'] = model_data['team'].astype('category').cat.codes

points = model_data['points'].to_numpy()
driver_idx = model_data['driver_id'].to_numpy()
team_idx = model_data['team_id'].to_numpy()
n_drivers = model_data['driver_id'].nunique()
n_teams = model_data['team_id'].nunique()

with pm.Model() as f1_model:
    sigma = pm.HalfNormal('sigma', sigma=5.0)
    sigma_driver = pm.HalfNormal('sigma_driver', sigma=5.0)
    sigma_team = pm.HalfNormal('sigma_team', sigma=5.0)

    alpha_driver = pm.Normal('alpha_driver', mu=0.0, sigma=sigma_driver, shape=n_drivers)
    beta_team = pm.Normal('beta_team', mu=0.0, sigma=sigma_team, shape=n_teams)

    mu = alpha_driver[driver_idx] + beta_team[team_idx]
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=points)

    trace = pm.sample(1000, tune=1000, target_accept=0.95, random_seed=42, return_inferencedata=True)

# Posterior means for driver effects
summ = az.summary(trace, var_names=['alpha_driver'])
driver_labels = model_data['driver_name'].astype('category').cat.categories
driver_skill = (pd.DataFrame({'driver_name': driver_labels,
                              'bayes_skill': summ['mean'].to_numpy()})
                .sort_values('bayes_skill', ascending=False))

# Merging average DPI
dpi_mean = dpi_df.groupby('driver_name', as_index=False)['DPI'].mean().rename(columns={'DPI': 'dpi_mean'})
driver_skill = driver_skill.merge(dpi_mean, on='driver_name', how='left')
driver_skill.to_csv(DATA_DIR + "bayesian_driver_rankings.csv", index=False)

print("Saved:", DATA_DIR + "dpi_dataset.csv", "and", DATA_DIR + "bayesian_driver_rankings.csv")
