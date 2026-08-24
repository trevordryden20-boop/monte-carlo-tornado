import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Distance-Based Rankine Inversion Engine", layout="wide")

class MassDistanceRankineEngine:
    def __init__(self, r_c=80.0, v_t=14.0, heading=45.0, alpha=0.65, inflow=0.25, rho=1.225):
        self.r_c, self.v_t, self.head, self.alpha, self.inflow, self.rho = (
            r_c, v_t, np.radians(heading), alpha, inflow, rho
        )

    def _field(self, x, y, x0, y0, v_max):
        """Calculates 2D velocity magnitude at (x, y) relative to center (x0, y0)."""
        dx, dy = x - x0, y - y0
        r, theta = np.hypot(dx, dy), np.arctan2(dy, dx)
        v_th = np.where(r <= self.r_c, v_max * (r / self.r_c), v_max * ((self.r_c / np.maximum(r, 1e-5)) ** self.alpha))
        v_r = -self.inflow * v_th
        u = v_r * np.cos(theta) - v_th * np.sin(theta) + self.v_t * np.sin(self.head)
        v = v_r * np.sin(theta) + v_th * np.cos(theta) + self.v_t * np.cos(self.head)
        return np.hypot(u, v)

    def estimate_displacement_distance(self, v_ms, mass_kg, cd=0.8, mu=0.5):
        """
        Estimates displacement distance (d) given local wind speed and mass.
        Uses an empirical power-law mass scaling for cross-sectional area (A ~ m^(2/3)).
        """
        if v_ms <= 0:
            return 0.0
            
        # Characteristic cross-sectional area estimated from mass density (assuming steel/wood/concrete mix)
        area_est = 0.03 * (mass_kg ** (2/3))
        
        q = 0.5 * self.rho * (v_ms ** 2)
        f_drag = q * cd * area_est
        weight = mass_kg * 9.81
        f_friction = mu * weight
        
        # Threshold wind force required for initial movement
        if f_drag <= f_friction:
            return 0.0
            
        # Net acceleration and physics-based displacement scaling
        a_net = (f_drag - f_friction) / mass_kg
        
        # Distance modeled via kinetic impulse duration over storm passage window
        t_exposure = 3.0  # seconds of peak impact
        dist_sim = 0.5 * a_net * (t_exposure ** 2)
        
        # Flight/ballistic boost scaling for high wind dynamic pressures
        if f_drag > (2.0 * weight):
            dist_sim *= (f_drag / weight) ** 1.5
            
        return dist_sim

    def invert(self, obs):
        """Inverts core v_max to match observed displacement distances."""
        ox = np.array([p['x'] for p in obs])
        oy = np.array([p['y'] for p in obs])
        target_dists = np.array([p['distance_m'] for p in obs])
        masses = np.array([p['mass_kg'] for p in obs])
        
        def loss(params):
            v_max_cand, x0_cand, y0_cand = params
            v_mags = self._field(ox, oy, x0_cand, y0_cand, v_max_cand)
            
            sim_dists = np.array([
                self.estimate_displacement_distance(v, m) 
                for v, m in zip(v_mags, masses)
            ])
            
            return np.mean((sim_dists - target_dists) ** 2)

        res = minimize(loss, [60.0, 0.0, 0.0], bounds=[(20, 180), (-500, 500), (-500, 500)], method='L-BFGS-B')
        return {'v_max': res.x[0], 'center': res.x[1:], 'mse': res.fun}

# --- Sidebar Controls ---
st.sidebar.title("🌪️ Vortex Physics")
r_c = st.sidebar.slider("Core Radius (r_c) [m]", 20.0, 200.0, 80.0)
v_t = st.sidebar.slider("Translation Speed [m/s]", 0.0, 30.0, 14.0)
head = st.sidebar.slider("Heading Angle (°)", 0, 360, 45)
alpha = st.sidebar.slider("Decay Exponent (α)", 0.3, 1.0, 0.65)
inflow = st.sidebar.slider("Inflow Ratio", 0.0, 0.6, 0.25)

engine = MassDistanceRankineEngine(r_c, v_t, head, alpha, inflow)

# --- App Interface ---
st.title("🚜 Mass & Distance Tornado Inversion Engine")
st.caption("Solves for core wind speeds using only the object's mass and total displacement distance.")

col_table, col_fig = st.columns([1, 2])

default_survey = [
    {'x': -120.0, 'y': 50.0,  'mass_kg': 1500.0, 'distance_m': 45.0},
    {'x': 150.0,  'y': -30.0, 'mass_kg': 1200.0, 'distance_m': 8.0},
    {'x': 80.0,   'y': 90.0,  'mass_kg': 2000.0, 'distance_m': 65.0},
    {'x': -250.0, 'y': -100.0,'mass_kg': 800.0,  'distance_m': 0.0},
]

with col_table:
    st.subheader("Field Observations")
    survey = st.data_editor(
        default_survey,
        num_rows="dynamic",
        column_config={
            "x": st.column_config.NumberColumn("X (m)"),
            "y": st.column_config.NumberColumn("Y (m)"),
            "mass_kg": st.column_config.NumberColumn("Object Mass (kg)", min_value=1.0, default=1000.0),
            "distance_m": st.column_config.NumberColumn("Displacement Distance (m)", min_value=0.0, default=0.0)
        }
    )

if len(survey) > 0:
    res = engine.invert(survey)
    
    with col_table:
        st.markdown("---")
        st.metric("Estimated Peak Wind", f"{res['v_max'] * 2.237:.1f} mph", f"{res['v_max']:.1f} m/s")
        st.metric("Inferred Vortex Center", f"({res['center'][0]:.1f}, {res['center'][1]:.1f}) m")
        st.metric("Distance Error (MSE)", f"{res['mse']:.4f}")

    with col_fig:
        # Generate Grid
        x_g = y_g = np.linspace(-300, 300, 100)
        X, Y = np.meshgrid(x_g, y_g)
        Z_mph = engine._field(X, Y, res['center'][0], res['center'][1], res['v_max']) * 2.23694

        # Matplotlib Plotting
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")

        contour = ax.contourf(X, Y, Z_mph, levels=25, cmap="viridis")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Wind Speed (mph)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

        # Survey Points
        obs_x = [p['x'] for p in survey]
        obs_y = [p['y'] for p in survey]
        obs_labels = [f"{p['mass_kg']}kg ({p['distance_m']}m)" for p in survey]
        
        ax.scatter(obs_x, obs_y, color="red", marker="x", s=80, linewidths=2, label="Survey Objects")
        for x, y, lbl in zip(obs_x, obs_y, obs_labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8), ha='center', color="white", fontsize=8)

        # Center Marker
        ax.scatter(res['center'][0], res['center'][1], color="yellow", marker="*", s=180, label="Inferred Center")

        ax.set_xlabel("X Offset (m)", color="white")
        ax.set_ylabel("Y Offset (m)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        
        ax.legend(facecolor="#1e222b", edgecolor="none", labelcolor="white")
        plt.tight_layout()

        st.pyplot(fig)