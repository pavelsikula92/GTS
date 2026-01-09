import numpy as np


class GTS_Anchor_Engine:
    """
    GTS - "The Anchor"
    ------------------------
    A physics engine that models gravity not as a fixed force, but as a variable
    emerging from dimensional stress.

    Cubic Stress Logic (tau^3)
    This logic creates a natural 'Forbidden Zone' around the Solar System regime,
    locking the dimension to exactly 3.0 locally, while allowing it to phase-shift
    into D < 3 (Dark Matter mimicry) or D > 3 (High Energy density) at the extremes.
    """

    def __init__(self):
        # --- Fundamental Constants ---
        self.C = 299792458.0  # Speed of light (m/s)
        self.G = 6.67430e-11  # Gravitational Constant

        # --- GTS Tuning Parameters ---
        # A0: The cosmic acceleration floor (Milgrom's constant variant).
        # Acts as the boundary where Dark Matter effects typically begin.
        self.A0 = 1.2e-10

        # ALPHA_G: Fine structure constant for gravity scaling.
        # Controls the sensitivity of the dimensional shift.
        self.ALPHA_G = 0.007297

    def get_substrate_metrics(self, mass, radius):
        """
        Calculates the dimensional metrics for a given mass/radius configuration.
        """

        # ---------------------------------------------------------
        # 1. Causal Stress (Chi)
        # ---------------------------------------------------------
        # We calculate the Newtonian acceleration.
        # Chi is the logarithmic magnitude of this acceleration relative to the cosmic floor A0.
        # High Chi = High Acceleration (Stars, Black Holes).
        # Low Chi = Low Acceleration (Galactic edges).
        accel = (self.G * mass) / (radius**2)
        chi = np.log10(accel / self.A0)

        # ---------------------------------------------------------
        # 2. Causal Saturation (Tau)
        # ---------------------------------------------------------
        # We normalize Chi against the "Solar System Anchor".
        # The value 12.4 is empirically where the Sun sits on the Chi scale.
        # tau represents the deviation from "Standard Physics" (Solar System).
        tau = (chi - 12.4) * self.ALPHA_G

        # ---------------------------------------------------------
        # 3. THE ANCHOR LOGIC (Non-Linear Elasticity)
        # ---------------------------------------------------------
        # This is the heart of v15.1.
        # Instead of a linear response, we use a CUBIC power (tau**3).
        # WHY? The cubic function is extremely flat near 0.
        # This ensures that for small deviations (like inside the Solar System),
        # the Stiffness remains effectively 0, enforcing strict Newton/Einstein physics.
        # The factor '1000' is the 'Gain' - once we leave the flat zone, it snaps quickly.
        stiffness = np.tanh(1000 * tau**3)

        # ---------------------------------------------------------
        # 4. Emergent Dimension (D_eff)
        # ---------------------------------------------------------
        # stiffness < 0: Low acceleration (Galactic Halo).
        # The vacuum "stiffens", acting like a lower dimension (D < 3).
        # This causes gravity to decay slower (1/r instead of 1/r^2), mimicking Dark Matter.
        #
        # stiffness > 0: Extreme acceleration (Neutron Stars).
        # The vacuum becomes "superconductive" (D > 3), potentially screening gravity.
        if stiffness < 0:
            # Drop dimension significantly to mimic Halo Mass
            d_eff = 3.0 + (stiffness * 1.0)
        else:
            # Rise dimension slightly in high-density cores
            d_eff = 3.0 + (stiffness * 0.4)

        # ---------------------------------------------------------
        # 5. Flux Pinning Exponent (p)
        # ---------------------------------------------------------
        # This converts the Dimension (D_eff) into a gravitational modifier.
        # It determines how strongly the flux lines are 'pinned' or concentrated.
        gamma = 3.0 / d_eff

        if d_eff < 3.0:
            # For Galactic Edges: Force gravity to act stronger than Newton.
            # We map the geometric drop directly to flux concentration.
            p = 1.0 + 3.0 * (3.0 - d_eff) / (max(d_eff, 1.1) - 1.0)
        else:
            # For Compact Objects: Gravity dissipates slightly faster/differently.
            p = 1.0 / (1.0 + (d_eff - 3.0))

        # ---------------------------------------------------------
        # 6. Final GTS Potential (Z_GTS)
        # ---------------------------------------------------------
        # Phi_n: Standard Newtonian Potential (dimensionless via C^2)
        # Boost: The multiplier derived from the dimensional shift.
        phi_n = (self.G * mass) / (radius * self.C**2)
        boost = gamma**p
        z_gts = phi_n * boost

        return {
            "chi": chi,
            "d_eff": d_eff,
            "boost": boost,
            "z_gts": z_gts,
            "stiffness": stiffness,  # Useful for debugging the Anchor
        }


# ==============================================================================
# VALIDATION RUN
# ==============================================================================
if __name__ == "__main__":
    engine = GTS_Anchor_Engine()

    # Test Cases: (Name, Mass [kg], Radius [m])
    # 1. MW Edge: Low acceleration regime (Where Dark Matter is usually needed)
    # 2. Sun: The reference point (Must be D=3.0, Boost=1.0)
    # 3. Sirius B: White Dwarf (High gravity)
    # 4. Neutron Star: Extreme gravity regime
    test_cases = [
        ("MW Edge", 6e10 * 1.98e30, 50 * 3.08e19),
        ("Sun", 1.98e30, 6.95e8),
        ("Sirius B", 1.018 * 1.98e30, 0.0081 * 6.95e8),
        ("Neutron Star", 1.4 * 1.98e30, 10000),
    ]

    print(
        f"{'OBJECT':<15} | {'CHI (Log A)':<12} | {'STIFFNESS':<10} | {'D_eff':<8} | {'BOOST':<8}"
    )
    print("-" * 65)

    for name, m, r in test_cases:
        res = engine.get_substrate_metrics(m, r)

        # Color coding for terminal output (optional visualization)
        # D < 3 (Dark Matter) -> Red
        # D = 3 (Newton) -> Green
        # D > 3 (High Energy) -> Blue
        indicator = "="
        if res["d_eff"] < 2.99:
            indicator = "v"  # Dimension Drop
        if res["d_eff"] > 3.01:
            indicator = "^"  # Dimension Spike

        print(
            f"{name:<15} | {res['chi']:>12.2f} | {res['stiffness']:>10.4f} | {res['d_eff']:>8.4f} | {res['boost']:>7.2f}x {indicator}"
        )
