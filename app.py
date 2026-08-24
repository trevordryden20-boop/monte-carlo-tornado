import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# --- Page Configuration ---
st.set_page_config(
page_title="Tornado Wind Speed Inversion Engine",
page_icon="🌪️",
layout="wide",
initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Styling ---
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
.stMetric { background-color: #1e222b; padding: 12px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- Physical Engine Class ---
class AdvancedRankineTornadoEngine:
def __init__(self, core_radius_m, trans_speed_m_s, heading_deg, alpha, inflow_ratio, air_density=1.225):
self.r_c = core_radius_m
self.v_t = trans_speed_m_s
self.heading_rad = np.radians(heading_deg)
self.alpha = alpha
self.inflow_ratio = inflow_ratio
self.rho = air_density

def velocity_vector_field(self, x, y, x0, y0, v_max):
dx = x - x0
dy = y - y0
r = np.hypot(dx, dy)
theta = np.arctan2(dy, dx)

v_theta = np.where(r <= self.r_c, v_max * (r / self.r_c), v_max * ((self.r_c / np.maximum(r, 1e-5)) ** self.alpha))
v_r = -self.inflow_ratio * v_theta

u_rot = v_r * np.cos(theta) - v_theta * np.sin(theta)
v_rot = v_r * np.sin(theta) + v_theta * np.cos(theta)

u_trans = self.v_t * np.sin(self.heading_rad)
v_trans = self.v_t * np.cos(self.heading_rad)

return u_rot + u_trans, v_rot + v_trans

def get_wind_and_pressure(self, x, y, x0, y0, v_max):
u, v = self.velocity_vector_field(x, y, x0, y0, v_max)
v_mag = np.hypot(u, v)
q = 0.5 * self.rho * (v_mag ** 2)
return v_mag, q, u, v

def physical_damage_model(self, q, failure_pressure_pa=3500.0, shape_k=3.0):
return 1.0 - np.exp(- (np.maximum(q, 0) / failure_pressure_pa) ** shape_k)

def estimate_v_max_multi_point(self, observations):
def loss_function(params):
v_max_cand, x0_cand, y0_cand = params
err = 0.0
for obs in observations:
_, q, _, _ = self.get_wind_and_pressure(obs['x'], obs['y'], x0_cand, y0_cand, v_max_cand)
sim_damage = self.physical_damage_model(q)
err += (sim_damage - obs['observed_damage']) ** 2
return err / len(observations)

initial_guess = [60.0, 0.0, 0.0]
bounds = [(20.0, 160.0), (-500.0, 500.0), (-500.0, 500.0)]
res = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')

return {
'v_max': res.x[0],
'center': (res.x[1], res.x[2]),
'mse': res.fun,
'success': res.success
}

# --- Sidebar Inputs ---
st.sidebar.title("🌪️ Model Parameters")
st.sidebar.markdown("---")

st.sidebar.subheader("Vortex Dynamics")
core_radius = st.sidebar.slider("Core Radius (r_c) [m]", 20.0, 200.0, 80.0, 5.0)
trans_speed = st.sidebar.slider("Translational Speed [m/s]", 0.0, 30.0, 14.0, 1.0)
heading = st.sidebar.slider("Heading Angle [°]", 0, 360, 45, 5)
alpha = st.sidebar.slider("Rankine Decay (α)", 0.3, 1.0, 0.65, 0.05)
inflow = st.sidebar.slider("Radial Inflow Ratio", 0.0, 0.6, 0.25, 0.05)

engine = AdvancedRankineTornadoEngine(core_radius, trans_speed, heading, alpha, inflow)

# --- Default Survey Points ---
default_survey = [
{'x': -120.0, 'y': 50.0, 'observed_damage': 0.88},
{'x': 150.0, 'y': -30.0, 'observed_damage': 0.35},
{'x': 80.0, 'y': 90.0, 'observed_damage': 0.95},
{'x': -200.0, 'y': -100.0,'observed_damage': 0.12},
]

# --- Main Interface ---
st.title("Tornado Kinematics & Damage Inversion Engine")
st.caption("Modified Rankine Vortex model paired with non-linear L-BFGS-B damage optimization.")

col_main, col_data = st.columns([3, 1])

with col_data:
st.subheader("Survey Points")
st.markdown("Coordinates relative to estimated track (m) & observed damage index (0-1).")

# Interactive Table for Editing Survey Points
survey_data = st.data_editor(
default_survey,
num_rows="dynamic",
column_config={
"x": st.column_config.NumberColumn("X (m)", default=0.0),
"y": st.column_config.NumberColumn("Y (m)", default=0.0),
"observed_damage": st.column_config.NumberColumn("Damage (0-1)", min_value=0.0, max_value=1.0, step=0.05)
}
)

# --- Optimization Run ---
if len(survey_data) > 0:
results = engine.estimate_v_max_multi_point(survey_data)

# KPI Metrics Banner
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Estimated Max Wind", f"{results['v_max'] * 2.237:.1f} mph", f"{results['v_max']:.1f} m/s")
m2.metric("Center Offset X", f"{results['center'][0]:.1f} m")
m3.metric("Center Offset Y", f"{results['center'][1]:.1f} m")
m4.metric("Optimizer MSE Loss", f"{results['mse']:.5f}")

# --- Plotly Visualization ---
st.markdown("### 2D Vector Field & Velocity Profile")

# Grid Setup
grid_size = 300
grid_res = 30
x_range = np.linspace(-grid_size, grid_size, grid_res)
y_range = np.linspace(-grid_size, grid_size, grid_res)
X, Y = np.meshgrid(x_range, y_range)

V_mag, Q, U, V = engine.get_wind_and_pressure(X, Y, results['center'][0], results['center'][1], results['v_max'])
V_mph = V_mag * 2.23694

fig = go.Figure()

# Heatmap Layer
fig.add_trace(go.Heatmap(
x=x_range, y=y_range, z=V_mph,
colorscale="Viridis",
colorbar=dict(title="Wind Speed (mph)"),
hoverinfo="x+y+z"
))

# Observed Damage Indicators
obs_x = [p['x'] for p in survey_data]
obs_y = [p['y'] for p insurvey_data]
obs_dmg = [f"Damage: {p['observed_damage']:.2f}" for p in survey_data]

fig.add_trace(go.Scatter(
x=obs_x, y=obs_y,
mode="markers+text",
marker=dict(size=14, color="red", symbol="x", line=dict(width=2, color="white")),
text=[f"P{i+1}" for i in range(len(survey_data))],
textposition="top center",
hovertext=obs_dmg,
name="Survey Observations"
))

# Estimated Vortex Center Marker
fig.add_trace(go.Scatter(
x=[results['center'][0]], y=[results['center'][1]],
mode="markers",
marker=dict(size=16, color="yellow", symbol="star"),
name="Inferred Center"
))

fig.update_layout(
xaxis_title="X Offset (meters)",
yaxis_title="Y Offset (meters)",
template="plotly_dark",
height=600,
margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

else:
st.warning("Please add at leastat least one survey point in the table to compute wind speeds.")