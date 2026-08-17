import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TorForensics MC-Est", page_icon="🌪️", layout="wide"
)

st.title("🌪️ TorForensics MC-Est: Instant Wind Speed Estimator")

# Sidebar controls
st.sidebar.header("Parameters")
mass = st.sidebar.number_input("Object Mass (kg)", value=14000.0, step=100.0)
cd = st.sidebar.number_input("Drag Coefficient (Cd)", value=0.80, step=0.01)
area = st.sidebar.number_input("Surface Area (m²)", value=10.0, step=0.5)
friction = st.sidebar.number_input(
    "Ground Friction (mu)", value=0.04, step=0.01
)
displacement = st.sidebar.slider("Displacement Distance (m)", 5.0, 500.0, 75.0)

# Instant Physics Calculation (Runs automatically on any change)
rho = 1.225
g = 9.81
num_iterations = 10000

sim_mass = np.random.normal(mass, mass * 0.05, num_iterations)
sim_cd = np.random.normal(cd, cd * 0.05, num_iterations)
sim_area = np.random.normal(area, area * 0.05, num_iterations)
sim_friction = np.random.normal(friction, friction * 0.05, num_iterations)

base_vel = np.sqrt(
    (2.0 * sim_friction * sim_mass * g) / (rho * sim_cd * sim_area)
)
simulated_speeds_mph = base_vel * 2.23694 * np.sqrt(displacement / 10.0)

mean_speed = np.mean(simulated_speeds_mph)
ci_lower = np.percentile(simulated_speeds_mph, 2.5)
ci_upper = np.percentile(simulated_speeds_mph, 97.5)


def get_ef_rating(speed):
  if speed < 86:
    return "EF-0"
  elif speed < 111:
    return "EF-1"
  elif speed < 136:
    return "EF-2"
  elif speed < 166:
    return "EF-3"
  elif speed < 200:
    return "EF-4"
  else:
    return "EF-5"


# Display Results Immediately
col1, col2 = st.columns(2)
with col1:
  st.metric(
      label="Estimated Mean Peak Wind Speed", value=f"{mean_speed:.1f} mph"
  )
  st.metric(
      label="95% Confidence Interval",
      value=f"{ci_lower:.1f} mph – {ci_upper:.1f} mph",
  )
  st.metric(label="Implied EF-Rating", value=get_ef_rating(mean_speed))

with col2:
  st.subheader("📈 Probability Distribution")
  hist_values, bin_edges = np.histogram(simulated_speeds_mph, bins=30)
  chart_data = pd.DataFrame(
      {"Frequency": hist_values}, index=np.round(bin_edges[:-1], 1)
  )
  st.line_chart(chart_data)
