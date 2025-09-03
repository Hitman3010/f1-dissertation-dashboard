import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading datasets
dpi_df = pd.read_csv("F1 data/dpi_dataset.csv")
summary_df = pd.read_csv("F1 data/bayesian_driver_team_summary.csv")

# Preprocessing
summary_df.rename(columns={'mean': 'skill', 'hdi_3%': 'low', 'hdi_97%': 'high'}, inplace=True)
summary_df['type'] = summary_df['type'].str.lower()
summary_df.dropna(subset=['skill'], inplace=True)

# Seperating driver and team skill
driver_skills = summary_df[summary_df['type'] == 'driver'][['name', 'skill']].rename(
    columns={'name': 'driver_name', 'skill': 'driver_skill'})
team_skills = summary_df[summary_df['type'] == 'team'][['name', 'skill']].rename(
    columns={'name': 'team_name', 'skill': 'constructor_effect'})

# Pairing driver with the latest constructor
latest_teams = dpi_df.sort_values(['driver_name', 'year'], ascending=[True, False]) \
    .drop_duplicates('driver_name')[['driver_name', 'team_name']]

# Merging
skill_vs_car = driver_skills.merge(latest_teams, on='driver_name', how='left') \
                            .merge(team_skills, on='team_name', how='left')
skill_vs_car.dropna(inplace=True)

# Streamlit dashboard
st.set_page_config(page_title="F1 Driver Skill vs Constructor Dashboard", layout="wide")
st.title("F1 Driver Skill vs Constructor Dashboard")

st.sidebar.header("Selections for DPI and Bayesian skill estimates")
drivers = sorted(dpi_df['driver_name'].unique())
constructors = sorted(team_skills['team_name'].unique())

selected_driver = st.sidebar.selectbox("Select Driver", drivers)
selected_constructor = st.sidebar.selectbox("Select Constructor", constructors)

# DPI Trend Plot
st.subheader(f"DPI Trend: {selected_driver}")
driver_trend = dpi_df[dpi_df['driver_name'] == selected_driver]

fig1, ax1 = plt.subplots()
sns.lineplot(data=driver_trend, x='year', y='DPI', marker='o', ax=ax1)
ax1.set_title(f"{selected_driver}'s Driver Performance Index Over Time")
ax1.set_xlabel("Year")
ax1.set_ylabel("DPI")
st.pyplot(fig1)

# Bayesian Skill Table - Driver
st.subheader(f"Bayesian Skill Estimate: {selected_driver}")
driver_post = summary_df[(summary_df['type'] == 'driver') & (summary_df['name'] == selected_driver)]
if not driver_post.empty:
    st.dataframe(driver_post[['name', 'skill', 'low', 'high']])
else:
    st.warning("No Bayesian skill estimate found for this driver.")

# Bayesian Skill Table - Team
st.subheader(f"Constructor Estimate: {selected_constructor}")
team_post = summary_df[(summary_df['type'] == 'team') & (summary_df['name'] == selected_constructor)]
if not team_post.empty:
    st.dataframe(team_post[['name', 'skill', 'low', 'high']])
else:
    st.warning("No constructor effect found for this team.")

# Scatter plot for driver skill and constructor effect
st.subheader("Driver Skill vs Constructor Effect")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=skill_vs_car, x='constructor_effect', y='driver_skill', alpha=0.4, ax=ax2)
# Highlight selected driver
highlight = skill_vs_car[skill_vs_car['driver_name'] == selected_driver]
if not highlight.empty:
    ax2.scatter(highlight['constructor_effect'], highlight['driver_skill'], color='red', s=100, label='Selected Driver')
    ax2.text(highlight['constructor_effect'].values[0], highlight['driver_skill'].values[0],
             selected_driver, fontsize=9)

constructor_row = team_skills[team_skills['team_name'] == selected_constructor]
if not constructor_row.empty:
    constructor_effect_value = constructor_row['constructor_effect'].values[0]
    ax2.axvline(x=constructor_effect_value, color='darkgreen', linestyle='--', label=f"{selected_constructor} Constructor")

ax2.set_title("Driver Skill vs Constructor Effect")
ax2.set_xlabel("Constructor Effect (Bayesian)")
ax2.set_ylabel("Driver Skill (Bayesian)")
ax2.legend()
st.pyplot(fig2)

# Simulation
st.subheader("Driver and Constructor Simulator")

sim_driver = st.selectbox("Select Driver for Simulation", drivers, key="sim_driver")
sim_team = st.selectbox("Select Hypothetical Constructor", constructors, key="sim_team")

# Get skill values
driver_row = driver_skills[driver_skills['driver_name'] == sim_driver]
team_row = team_skills[team_skills['team_name'] == sim_team]

if not driver_row.empty and not team_row.empty:
    simulated_score = driver_row['driver_skill'].values[0] + team_row['constructor_effect'].values[0]
    st.success(f"Simulated Performance Score for {sim_driver} driving for {sim_team}: **{simulated_score:.2f}**")

    # Compare with actual (if available)
    actual_team_row = latest_teams[latest_teams['driver_name'] == sim_driver]
    if not actual_team_row.empty:
        actual_team_name = actual_team_row['team_name'].values[0]
        actual_team_skill = team_skills[team_skills['team_name'] == actual_team_name]['constructor_effect'].values[0]
        actual_combo_score = driver_row['driver_skill'].values[0] + actual_team_skill

        st.info(f"Actual Score for {sim_driver} at {actual_team_name}: **{actual_combo_score:.2f}**")

        delta = simulated_score - actual_combo_score
        comment = "That would be an upgrade!" if delta > 0 else "Not an upgrade"
        st.markdown(f"**Difference: {delta:+.2f}** — {comment}")
else:
    st.warning("Could not find Bayesian estimates for simulation.")
