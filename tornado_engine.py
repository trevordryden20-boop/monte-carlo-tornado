import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Rankine Object Displacement Inversion Engine", layout="wide")

MODE_MAP = {
    "Stationary": 0,
    "Rolled / Slid": 1,
    "Bounced / Tumbling": 2,
    "Lofted / Airborne": 3
}

class RankineDisplacementEngine:
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

    def evaluate_displacement(self, v_ms, mass_kg=1500.0, area_m2=4.0, cd=0.8, cl=0.4, mu=0.6):
        """Maps local wind speed to physical heavy object motion mode."""
        q = 0.5 * self.rho * (v_ms ** 2)
        f_drag = q * cd * area_m2
        f_lift = q * cl * area_m2
        weight = mass_kg * 9.81
        
        net_weight = max(0.0, weight - f_lift)
        f_friction = mu * net_weight
        
        if f_lift >= weight:
            return 3.0  # Lofted / Airborne
        elif f_drag > f_friction and f_lift > (weight * 0.5):
            return 2.0  # Bounced / Tumbling
        elif f_drag > f_friction:
            return 1.0  # Rolled / Slid
        else:
            return 0.0  # Stationary

    def invert(self, obs):
        """Inverts core v_max to match observed heavy object displacement modes."""
        ox = np.array([p['x'] for p in obs])
        oy = np.array([p['y'] for p in obs])
        target_modes = np.array([MODE_MAP[p['observed_mode']] for p in obs])
        
        def loss(params):
            v_max_cand, x0_cand, y0_cand = params
            v_mags = self._field(ox, oy, x0_cand, y0_cand, v_max_cand)
            
            sim_modes = np.array([
                self.evaluate_displacement(v, p.get('mass', 1500.0), p.get('area', 4.0)) 
                for v, p in zip(v_mags, obs)
            ])
            
            return np.mean((sim_modes - target_modes) ** 2)

        res = minimize(loss, [60.0, 0.0, 0.0], bounds=[(20, 160), (-500, 500), (-500, 500)], method='L-BFGS-B')
        return {'v_max': res.x[0], 'center': res.x[1:], 'mse': res.fun}

# --- Sidebar Controls ---
st.sidebar.title("🌪️ Vortex Physics")
r_c = st.sidebar.slider("Core Radius (r_c) [m]", 20.0, 200.0, 80.0)
v_t = st.sidebar.slider("Translation Speed [m/s]", 0.0, 30.0, 14.0)
head = st.sidebar.slider("Heading Angle (°)", 0, 360, 45)
alpha = st.sidebar.slider("Decay Exponent (α)", 0.3, 1.0, 0.65)
inflow = st.sidebar.slider("Inflow Ratio", 0.0, 0.6, 0.25)

engine = RankineDisplacementEngine(r_c, v_t, head, alpha, inflow)

# --- App Interface ---
st.title("🚜 Heavy Object Displacement Inversion Engine")
st.caption("Solves for core tornado winds by iteratively fitting physical drag/lift displacement modes.")

col_table, col_fig = st.columns([1, 2])

default_survey = [
    {'x': -120.0, 'y': 50.0,  'observed_mode': 'Lofted / Airborne', 'mass': 1500.0, 'area': 4.0},
    {'x': 150.0,  'y': -30.0, 'observed_mode': 'Rolled / Slid',      'mass': 1500.0, 'area': 4.0},
    {'x': 80.0,   'y': 90.0,  'observed_mode': 'Bounced / Tumbling', 'mass': 1500.0, 'area': 4.0},
    {'x': -250.0, 'y': -100.0,'observed_mode': 'Stationary',         'mass': 1500.0, 'area': 4.0},
]

with col_table:
    st.subheader("Field Observations")
    survey = st.data_editor(
        default_survey,
        num_rows="dynamic",
        column_config={
            "x": st.column_config.NumberColumn("X (m)"),
            "y": st.column_config.NumberColumn("Y (m)"),
            "observed_mode": st.column_config.SelectboxColumn("Observed Displacement", options=list(MODE_MAP.keys())),
            "mass": st.column_config.NumberColumn("Mass (kg)", default=1500.0),
            "area": st.column_config.NumberColumn("Area (m²)", default=4.0)
        }
    )

if len(survey) > 0:
    res = engine.invert(survey)
    
    with col_table:
        st.markdown("---")
        st.metric("Estimated Peak Wind", f"{res['v_max'] * 2.237:.1f} mph", f"{res['v_max']:.1f} m/s")
        st.metric("Inferred Vortex Center", f"({res['center'][0]:.1f}, {res['center'][1]:.1f}) m")
        st.metric("Displacement Fit MSE Loss", f"{res['mse']:.6f}")

    with col_fig:
        # Generate Grid
        x_g = y_g = np.linspace(-300, 300, 100)
        X, Y = np.meshgrid(x_g, y_g)
        Z_mph = engine._field(X, Y, res['center'][0], res['center'][1], res['v_max']) * 2.23694

        # Matplotlib Rendering
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")

        # Contour Field
        contour = ax.contourf(X, Y, Z_mph, levels=25, cmap="viridis")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Wind Speed (mph)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

        # Survey Points
        obs_x = [p['x'] for p in survey]
        obs_y = [p['y'] for p in survey]
        obs_labels = [p['observed_mode'].split()[0] for p in survey]
        
        ax.scatter(obs_x, obs_y, color="red", marker="x", s=80, linewidths=2, label="Field Objects")
        for x, y, lbl in zip(obs_x, obs_y, obs_labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8), ha='center', color="white", fontsize=9)

        # Inferred Center
        ax.scatter(res['center'][0], res['center'][1], color="yellow", marker="*", s=180, label="Inferred Center")

        # Formatting
        ax.set_xlabel("X Offset (m)", color="white")
        ax.set_ylabel("Y Offset (m)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        
        ax.legend(facecolor="#1e222b", edgecolor="none", labelcolor="white")
        plt.tight_layout()

        st.pyplot(fig)