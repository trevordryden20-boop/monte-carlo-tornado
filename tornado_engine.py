import numpy as np
from scipy.optimize import minimize

class AdvancedRankineTornadoEngine:
    def __init__(self, core_radius_m=80.0, trans_speed_m_s=12.0, 
                 heading_deg=0.0, alpha=0.6, inflow_ratio=0.3, air_density=1.225):
        """
        :param core_radius_m: Radius of maximum winds (r_c) in meters
        :param trans_speed_m_s: Forward translation speed of tornado center in m/s
        :param heading_deg: Translation heading (0 = North, 90 = East)
        :param alpha: Modified Rankine decay parameter (0.5 to 0.75)
        :param inflow_ratio: Ratio of radial velocity to tangential velocity
        :param air_density: Air density in kg/m^3
        """
        self.r_c = core_radius_m
        self.v_t = trans_speed_m_s
        self.heading_rad = np.radians(heading_deg)
        self.alpha = alpha
        self.inflow_ratio = inflow_ratio
        self.rho = air_density

    def velocity_vector_field(self, x, y, x0, y0, v_max):
        """Calculates 2D vector wind components (u, v) at coordinates (x, y)."""
        dx = x - x0
        dy = y - y0
        r = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)

        if r == 0:
            return 0.0, 0.0

        # Modified Rankine tangential speed
        if r <= self.r_c:
            v_theta = v_max * (r / self.r_c)
        else:
            v_theta = v_max * ((self.r_c / r) ** self.alpha)

        # Radial velocity (inward)
        v_r = -self.inflow_ratio * v_theta

        # Polar to Cartesian conversion
        u_rot = v_r * np.cos(theta) - v_theta * np.sin(theta)
        v_rot = v_r * np.sin(theta) + v_theta * np.cos(theta)

        # Translational addition
        u_trans = self.v_t * np.sin(self.heading_rad)
        v_trans = self.v_t * np.cos(self.heading_rad)

        return u_rot + u_trans, v_rot + v_trans

    def get_peak_wind_and_pressure(self, x, y, x0, y0, v_max):
        """Returns peak velocity magnitude and dynamic pressure at (x,y)."""
        u, v = self.velocity_vector_field(x, y, x0, y0, v_max)
        v_mag = np.hypot(u, v)
        dynamic_pressure = 0.5 * self.rho * (v_mag ** 2)
        return v_mag, dynamic_pressure

    def physical_damage_model(self, dynamic_pressure_pa, failure_pressure_pa=3500.0, shape_k=3.0):
        """Maps dynamic wind pressure (Pa) to structural damage index (0.0 to 1.0)."""
        if dynamic_pressure_pa <= 0:
            return 0.0
        return 1.0 - np.exp(- (dynamic_pressure_pa / failure_pressure_pa) ** shape_k)

    def estimate_v_max_multi_point(self, observations):
        """Solves for peak core wind speed and center offset given survey points."""
        def loss_function(params):
            v_max_cand, x0_cand, y0_cand = params
            total_error = 0.0

            for obs in observations:
                _, q = self.get_peak_wind_and_pressure(obs['x'], obs['y'], x0_cand, y0_cand, v_max_cand)
                sim_damage = self.physical_damage_model(q)
                total_error += (sim_damage - obs['observed_damage']) ** 2

            return total_error / len(observations)

        # Initial parameter guesses: [v_max, center_x, center_y]
        initial_guess = [60.0, 0.0, 0.0]
        bounds = [(20.0, 160.0), (-500.0, 500.0), (-500.0, 500.0)]

        res = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')

        if res.success:
            return {
                'v_max_m_s': res.x[0],
                'v_max_mph': res.x[0] * 2.23694,
                'estimated_center': (res.x[1], res.x[2]),
                'mse_loss': res.fun
            }
        else:
            raise RuntimeError("Optimizer failed to find a valid fit.")

# --- Running a custom simulation ---
if __name__ == "__main__":
    # 1. Initialize engine with storm parameters
    engine = AdvancedRankineTornadoEngine(
        core_radius_m=90.0,   # Core width in meters
        trans_speed_m_s=14.0, # Tornado motion speed in m/s
        heading_deg=45.0,     # Moving Northeast (45 degrees)
        alpha=0.65,           # Decay rate outside core
        inflow_ratio=0.25     # Inward flow intensity
    )

    # 2. Input damage survey coordinates (X, Y in meters relative to estimated track)
    # observed_damage ranges from 0.0 (no damage) to 1.0 (complete destruction)
    field_data = [
        {'x': -120.0, 'y': 50.0,  'observed_damage': 0.88},
        {'x': 150.0,  'y': -30.0, 'observed_damage': 0.35},
        {'x': 80.0,   'y': 90.0,  'observed_damage': 0.95},
        {'x': -200.0, 'y': -100.0,'observed_damage': 0.12},
    ]

    # 3. Solve for maximum wind speed
    output = engine.estimate_v_max_multi_point(field_data)

    print("=== ESTIMATION RESULTS ===")
    print(f"Max Core Wind Speed : {output['v_max_m_s']:.2f} m/s ({output['v_max_mph']:.1f} mph)")
    print(f"Center Position Offset: X={output['estimated_center'][0]:.1f}m, Y={output['estimated_center'][1]:.1f}m")
    print(f"Model Fit Loss (MSE)  : {output['mse_loss']:.6f}")
