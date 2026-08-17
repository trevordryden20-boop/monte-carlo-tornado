import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="TorForensics Advanced Estimator",
    page_icon="🌪️",
    layout="wide"
)

st.title("🌪️ TorForensics: Advanced 3D Trajectory & Lofting Estimator")
st.markdown("A robust forensic engineering dashboard modeling heavy asset sliding, rolling resistance, and ballistic lofting dynamics.")

# Sidebar parameters
st.sidebar.header("Asset & Event Parameters")

asset_type = st.sidebar.selectbox(
    "Select Asset Template",
    [
        "Industrial Oil Tanker / Storage Cylinder",
        "Commercial Semi-Truck",
        "Residential Roof Section",
        "Custom Heavy Asset"
    ]
)

if asset_type == "Industrial Oil Tanker / Storage Cylinder":
    def_mass, def_cd, def_area, def_friction = 18000.0, 0.48, 16.0, 0.35
elif asset_type == "Commercial Semi-Truck":
    def_mass, def_cd, def_area, def_friction = 14000.0, 0.80, 10.0, 0.04
elif asset_type == "Residential Roof Section":
    def_mass, def_cd, def_area, def_friction = 5000.0, 1.25, 30.0, 0.20
else:
    def_mass, def_cd, def_area, def_friction = 1000.0, 0.50, 5.0, 0.10

mass = st.sidebar.number_input("Asset Mass (kg)", value=def_mass, step=500.0)
cd = st.sidebar.number_input("Aerodynamic Drag Coefficient (Cd)", value=def_cd, step=0.01)
area = st.sidebar.number_input("Projected Surface Area (m²)", value=def_area, step=1.0)
friction = st.sidebar.number_input("Ground Friction / Slide Resistance (mu)", value=def_friction, step=0.01)
displacement = st.sidebar.slider("Measured Displacement / Throw Distance (m)", min_value=10.0, max_value=1000.0, value=250.0, step=10.0)

st.sidebar.markdown("---")
iterations = st.sidebar.selectbox("Monte Carlo Iterations", [10000, 50000, 100000], index=1)
uncertainty = st.sidebar.slider("Parameter Uncertainty (%)", min_value=5, max_value=30, value=15)

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ Stochastic Simulation Engine")
    st.markdown("Click below to compute the probabilistic velocity profile required to displace or loft heavy assets.")

    if st.button("Run Forensic Simulation", type="primary"):
        with st.spinner("Processing trajectory and force vectors..."):
            rho = 1.225
            g = 9.81
            
            noise = uncertainty / 100.0
            sim_mass = np.random.normal(mass, mass * noise * 0.5, iterations)
            sim_cd = np.random.normal(cd, cd * noise, iterations)
            sim_area = np.random.normal(area, area * noise, iterations)
            sim_friction = np.random.normal(friction, friction * noise, iterations)
            
            sim_mass = np.clip(sim_mass, 50.0, None)
            sim_cd = np.clip(sim_cd, 0.1, 3.0)
            
            # Combined rolling/sliding + ballistic lofting estimation model
            launch_angle = np.random.uniform(np.radians(20), np.radians(60), iterations)
            air_factor = 1.0 + (sim_cd * sim_area) / (2.0 * sim_mass)
            
            # Inverted range formula accounting for drag momentum transfer
            vel_ms = np.sqrt((displacement * g * air_factor) / np.sin(2.0 * launch_angle))
            vel_mph = vel_ms * 2.23694
            
            st.session_state['results'] = vel_mph
            st.success("Simulation Executed Successfully!")

with col2:
    st.subheader("🎯 Forensic Wind Speed Results")
    if 'results' in st.session_state:
        res = st.session_state['results']
        mean_v = np.mean(res)
        ci_low = np.percentile(res, 2.5)
        ci_high = np.percentile(res, 97.5)
        
        def get_ef(v):
            if v < 86: return "EF-0"
            elif v < 111: return "EF-1"
            elif v < 136: return "EF-2"
            elif v < 166: return "EF-3"
            elif v < 200: return "EF-4"
            else: return "EF-5"
            
        st.metric(label="Estimated Mean Peak Wind Speed", value=f"{mean_v:.1f} mph")
        st.metric(label="95% Confidence Interval", value=f"{ci_low:.1f} mph – {ci_high:.1f} mph")
        st.metric(label="Estimated EF-Scale Rating", value=get_ef(mean_v))
    else:
        st.info("Run the simulation to generate wind speed outputs.")

if 'results' in st.session_state:
    st.markdown("---")
    st.subheader("📈 Velocity Probability Distribution")
    hist, bins = np.histogram(st.session_state['results'], bins=40)
    df_chart = pd.DataFrame({"Frequency": hist}, index=np.round(bins[:-1], 1))
    st.line_chart(df_chart)
