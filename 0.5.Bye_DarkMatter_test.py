import numpy as np
import matplotlib.pyplot as plt

# Apply a 'Science/Space' dark theme for better visualization
plt.style.use("dark_background")

# --- CONFIGURATION & CONSTANTS ---
CONSTANTS = {
    "A_0": 1.2e-10,  # Cosmic Acceleration Floor [m/s^2] (Milgrom's Constant)
    "G": 6.67430e-11,  # Gravitational Constant
    "C": 299792458,  # Speed of Light
    "M_SUN": 1.98847e30,  # Solar Mass [kg]
    "KPC": 3.086e19,  # Kiloparsec to Meters
    "ALPHA_G": 0.007297,  # Fine Structure Scaling
}


class GTS_Galactic_Engine:
    """
    GTS - The Topological Anchor
    ----------------------------------
    Implements the 'Cubic Stress Logic' to naturally emerge flat rotation curves
    by stiffening the vacuum metric at low accelerations.
    """

    @staticmethod
    def get_metrics(mass_enclosed, radius_m):
        """
        Calculates the topological metrics (Boost, D_eff) for a specific point in the galaxy.

        Args:
            mass_enclosed (float): Mass inside the current radius (kg).
            radius_m (float): Distance from galactic center (meters).

        Returns:
            tuple: (gravitational_boost, effective_dimension, chi_stress)
        """

        # 1. Standard Newtonian Acceleration
        # This is what standard physics predicts at this radius.
        accel_n = (CONSTANTS["G"] * mass_enclosed) / (radius_m**2)

        # 2. Causal Stress (Chi)
        # Logarithmic measure of how "weak" the gravity is compared to the cosmic floor (A_0).
        # In deep space (galactic edge), this value drops significantly.
        chi = np.log10(accel_n / CONSTANTS["A_0"])

        # 3. THE ANCHOR: Cubic Elasticity Logic
        # We shift the center to 12.4 (Solar System Baseline).
        # Using a CUBIC power (tau**3) ensures the "Forbidden Zone" (Solar System)
        # remains flat and strictly Newtonian (Stiffness ~ 0).
        tau = (chi - 12.4) * CONSTANTS["ALPHA_G"]

        # The Gain (1200) determines how sharply the physics snaps
        # from Newton to Modified Gravity once we leave the Solar System regime.
        stiffness = np.tanh(1200 * tau**3)

        # 4. Emergent Dimension (D_eff) Calculation
        # Stiffness < 0 means the vacuum is "hardening" due to low energy density.
        # This reduces the effective spatial dimension from 3.0 down towards ~2.0.
        if stiffness < 0:
            # Low Acceleration Regime (Galactic Halo) -> Dimension Drops
            # 0.98 is the 'Dark Matter Amplitude'
            d_eff = 3.0 + (stiffness * 0.98)
        else:
            # High Acceleration Regime (Compact Objects) -> Dimension Expands slightly
            d_eff = 3.0 + (stiffness * 0.24)

        # 5. Flux Pinning (The 'Dark Matter' Effect)
        # As D_eff drops, the gravitational flux is "pinned" or concentrated.
        # This prevents gravity from dissipating as fast as 1/r^2.
        gamma = 3.0 / d_eff

        if d_eff < 3.0:
            # Strong pinning calculation for halo regions
            p = 1.0 + 3.0 * (3.0 - d_eff) / (max(d_eff, 1.1) - 1.0)
        else:
            # Dissipation for high-density cores
            p = 1.0 / (1.0 + (d_eff - 3.0))

        boost = gamma**p
        return boost, d_eff, chi


class GalaxySimulation:
    """
    Simulates a spiral galaxy to compare Newtonian vs. GTS physics.
    """

    def __init__(self, total_mass_solar, max_radius_kpc):
        self.total_mass = total_mass_solar * CONSTANTS["M_SUN"]
        self.max_radius_kpc = max_radius_kpc
        self.radii = np.linspace(0.1, max_radius_kpc, 300)  # Resolution points

    def run(self):
        v_newton = []
        v_gts = []
        d_eff_profile = []

        print(
            f"Simulating Galaxy ({self.total_mass / CONSTANTS['M_SUN']:.1e} M_sun)..."
        )

        for r_kpc in self.radii:
            r_m = r_kpc * CONSTANTS["KPC"]

            # --- Mass Distribution Model ---
            # Using a simplified exponential disk model for cumulative mass.
            # M(<r) = M_total * (1 - e^(-r/scale_length))
            scale_length = 4.0  # kpc
            m_enclosed = self.total_mass * (1 - np.exp(-r_kpc / scale_length))

            # --- GTS Engine Calculation ---
            boost, d_eff, _ = GTS_Galactic_Engine.get_metrics(m_enclosed, r_m)

            # --- Velocity Calculation ---
            # V = sqrt( G * M / r )
            base_v_sq = (CONSTANTS["G"] * m_enclosed) / r_m

            # 1. Newtonian Velocity (Standard Model)
            v_n = np.sqrt(base_v_sq)

            # 2. GTS Velocity (Topological Model)
            # The 'boost' factor increases the effective gravity.
            v_g = np.sqrt(base_v_sq * boost)

            v_newton.append(v_n / 1000.0)  # Convert to km/s
            v_gts.append(v_g / 1000.0)  # Convert to km/s
            d_eff_profile.append(d_eff)

        return v_newton, v_gts, d_eff_profile

    def plot_results(self, v_n, v_g, d_eff):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # --- Plot 1: Rotation Curve ---
        ax1.plot(
            self.radii,
            v_n,
            color="#FF4B4B",
            linestyle="--",
            label="Newtonian (Visible Mass Only)",
            alpha=0.7,
        )
        ax1.plot(
            self.radii,
            v_g,
            color="#00E5FF",
            linewidth=3,
            label="GTS v15.2 (Emergent Gravity)",
        )

        # Stylistic details
        ax1.set_ylabel("Orbital Velocity [km/s]", fontsize=12, color="white")
        ax1.set_title(
            f"Galactic Rotation Curve: GTS v15.2 'Anchor' Logic",
            fontsize=14,
            color="white",
        )
        ax1.legend(loc="lower right", frameon=True, facecolor="#222")
        ax1.grid(color="white", alpha=0.1)

        # Annotation for the 'Dark Matter' gap
        gap_idx = -1
        ax1.annotate(
            "Dark Matter Illusion\n(Topological Boost)",
            xy=(self.radii[gap_idx], (v_n[gap_idx] + v_g[gap_idx]) / 2),
            xytext=(self.radii[gap_idx] - 15, v_g[gap_idx] + 20),
            arrowprops=dict(facecolor="white", shrink=0.05),
            color="white",
            fontsize=10,
        )

        # --- Plot 2: Dimensional Profile ---
        ax2.plot(self.radii, d_eff, color="#FFD700", linewidth=2.5)

        # The Critical Line (D=3)
        ax2.axhline(
            3.0,
            color="white",
            linestyle=":",
            alpha=0.5,
            label="Standard Dimension (D=3)",
        )

        ax2.set_ylabel("Effective Dimension ($D_{eff}$)", fontsize=12, color="white")
        ax2.set_xlabel("Radius [kpc]", fontsize=12, color="white")
        ax2.set_ylim(1.9, 3.1)
        ax2.grid(color="white", alpha=0.1)
        ax2.text(
            2,
            2.1,
            "Galactic Halo Regime\n(Stiffened Vacuum D < 3)",
            color="#FFD700",
            fontsize=10,
        )
        ax2.text(2, 3.02, "Newtonian Core Regime (D = 3)", color="white", fontsize=10)

        print("Rendering simulation plot...")
        plt.tight_layout()
        plt.show()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Simulate a Milky Way-sized galaxy
    # Mass: ~6e10 Solar Masses (Visible/Baryonic only)
    # Radius: 50 kpc
    sim = GalaxySimulation(total_mass_solar=6.0e10, max_radius_kpc=50)
    v_newton, v_gts, d_eff = sim.run()
    sim.plot_results(v_newton, v_gts, d_eff)
