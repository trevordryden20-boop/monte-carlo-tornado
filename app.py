import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="TorForensics MC-Est v2.1", page_icon="🌪️", layout="wide"
)

# App Header
st.title(
    "🌪️ TorForensics MC-Est: Monte Carlo Tornado Wind Speed Estimator"
)
st.markdown(
    "Advanced forensic engineering dashboard for estimating peak wind speeds from displaced heavy objects and debris dynamics."
)

# Sidebar: Object and Physics Parameters
st.sidebar.header("Simulation & Object Parameters")

object_type = st.sidebar.selectbox(
    "Select Object Template",
    [
        "Commercial Semi-Truck",
        "Passenger Vehicle",
        "Residential Roof Truss",
        "Storage Shed",
        "Custom Object",
    ],
)

# Preset values based on template
if object_type == "Commercial Semi-Truck":
    default_mass, default_cd, default_area, default_friction = (
        14000.0,
        0.80,
        10.0,
        0.04,
    )
elif object_type == "Passenger Vehicle":
    default_mass, default_cd, default_area, default_friction = (
        1600.0,
        0.35,
        2.2,
        0.05,
    )
elif object_type == "Residential Roof Truss":
    default_mass, default_cd, default_area, default_friction = (
        300.0,
        1.25,
        6.0,
        0.40,
    )
elif object_type == "Storage Shed":
    default_mass, default_cd, default_area, default_friction = (
        2500.0,
        1.10,
        15.0,
        0.30,
    )
else:
    default_mass, default_cd, default_area, default_friction = (
        1000.0,
        0.50,
        5.0,
        0.10,
    )

mass = st.sidebar.number_input(
    "Object Mass (kg)", value=default_mass, step=100.0
)
cd = st.sidebar.number_input(
    "Aerodynamic Drag Coefficient (Cd)", value=default_cd, step=0.01
)
area = st.sidebar.number_input(
    "Projected Surface Area (m²)", value=default_area, step=0.5
)
friction = st.sidebar.number_input(
    "Ground Friction / Resistance Coefficient (mu)",
    value=default_friction,
    step=0.01,
)
displacement = st.sidebar.slider(
    "Measured Displacement Distance (meters)",
    min_value=5.0,
    max_value=500.0,
    value=75.0,
    step=5.0,
)

st.sidebar.markdown("---")
num_iterations = st.sidebar.selectbox(
    "Monte Carlo Iterations", [10000, 50000, 100000], index=1
)
uncertainty = st.sidebar.slider(
    "Parameter Uncertainty (%)", min_value=5, max_value=30, value=15
)

# Main Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Monte Carlo Simulation Engine")
    st.markdown(
        "Click below to execute stochastic iterations accounting for gusts, surface variation, and aerodynamic lift/drag vectors."
    )

    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running simulations across variable vectors..."):
            # Physics Model Constants
            rho = 1.225  # Air density at sea level (kg/m^3)
            g = 9.81  # Gravity (m/s^2)

            # Vectorized Monte Carlo inputs with Gaussian distribution noise
            noise_factor = uncertainty / 100.0
            sim_mass = np.random.normal(
                mass, mass * noise_factor * 0.2, num_iterations
            )
            sim_cd = np.random.normal(cd, cd * noise_factor, num_iterations)
            sim_area = np.random.normal(
                area, area * noise_factor * 0.1, num_iterations
            )
            sim_friction = np.random.normal(
                friction, friction * noise_factor, num_iterations
            )

            # Ensure physical constraints
            sim_mass = np.clip(sim_mass, 10.0, None)
            sim_cd = np.clip(sim_cd, 0.05, 3.0)
            sim_friction = np.clip(sim_friction, 0.01, 1.0)

            # Velocity calculation derived from force balance
            base_vel = np.sqrt(
                (2.0 * sim_friction * sim_mass * g)
                / (rho * sim_cd * sim_area)
            )
            distance_scaling = np.random.normal(
                1.0, 0.15, num_iterations
            ) * np.sqrt(displacement / 10.0)
            simulated_speeds_ms = base_vel * (1.0 + 0.05 * distance_scaling)

            # Convert m/s to mph
            simulated_speeds_mph = simulated_speeds_ms * 2.23694

            # Store results in session state
            st.session_state["sim_results"] = simulated_speeds_mph
            st.success("Simulation Complete!")

with col2:
    st.subheader("🎯 Results & EF-Rating Analysis")
    if "sim_results" in st.session_state:
        results = st.session_state["sim_results"]
        mean_speed = np.mean(results)
        ci_lower = np.percentile(results, 2.5)
        ci_upper = np.percentile(results, 97.5)


        # EF Rating Estimator based on 3-second gust thresholds
        def get_ef_rating(speed):
            if speed < 86:
                return "EF-0"
            elif speed < 111:
                return "EF-1"
            elif speed < 136:
                return "EF-2"
            elif speed < 166:
 