import numpy as np


def gts_engine_v10_3(density, clustering=0.3, alpha_g=0.007297):
    """
    GTS Engine - Causal Substrate Response
    -------------------------------------------
    Implements the core mechanics of the Geometric Unification Theory.
    Models the vacuum as a dynamic medium where spacetime geometry
    is an emergent state driven by causal stress.

    Axiom Reference:
    - Phase Transitions: Controlled by local clustering (substrate stiffness).
    - Forbidden Zone: Stability plateau preserving 3D Euclidean space.
    - Lumen Shift: Changes in causal propagation speed (Topological Impedance).
    """

    # --- 1. ADAPTIVE TOPOLOGICAL THRESHOLDS ---
    # The thresholds for state-shifts are modulated by local connectivity (clustering).
    # Stiffer networks (high clustering) trigger transitions at different energy levels.
    rho_crit_stiff = 1.5 * (1.0 + (clustering - 0.3))
    rho_crit_super = 25.0 * (1.0 - (clustering - 0.3))

    # --- 2. VACUUM AMPLITUDE & IMPEDANCE BRIDGE ---
    # Derived from the 1/pi projection axiom.
    # Represents the substrate's inherent capacity to transmit causal influence.
    a_0_base = 0.3183  # 1/pi
    a_0 = a_0_base * (1.0 + alpha_g * np.log(density + 1e-9))

    # --- 3. TOPOLOGICAL PHASE DYNAMICS (Sigmoids) ---
    k_visc = 12.0  # Represents the "viscosity" of the topological substrate

    # PHASE A: GALACTIC STIFFENING (D < 3)
    # Occurs in low-stress regimes (voids/galaxy edges).
    # Substrate "freezes," concentrating gravitational flux (Dark Matter effect).
    stiff = -a_0 * (1.0 / (1.0 + np.exp(k_visc * (density - rho_crit_stiff))))

    # PHASE B: TOPOLOGICAL SUPERCONDUCTIVITY (D > 3)
    # Occurs in extreme-stress regimes (White Dwarf cores).
    # Substrate allows "shortcuts," diluting gravitational flux (Redshift damping).
    superc = a_0 * (1.0 / (1.0 + np.exp(-k_visc * (density - rho_crit_super))))

    # --- 4. THE FORBIDDEN ZONE GATE (Stability Axiom) ---
    # Standard GR (D=3) is locked in for intermediate stress levels.
    # This prevents GTS effects from disrupting Solar System measurements.
    if 5.0 < density < 13.5:
        total_resp = 0.0
    else:
        total_resp = stiff + superc

    # --- 5. EMERGENT EFFECTIVE DIMENSION (D_eff) ---
    # Spacetime dimension is a local variable, not a universal constant.
    d_eff = 3.0 * (1.0 + total_resp)

    # --- 6. GEOMETRIC FLUX PINNING (Exponent p) ---
    # Determines the intensity of the gravitational field based on local D_eff.
    gamma = 3.0 / d_eff

    if d_eff < 3.0:
        # Concentration of flux lines in restricted geometry.
        p = 1.0 + 3.0 * (3.0 - d_eff) / (max(d_eff, 1.1) - 1.0)
    else:
        # Leakage of flux lines into hyper-dimensional degrees of freedom.
        p = 1.0 / (1.0 + 0.8 * (d_eff - 3.0))

    # --- 7. THE LUMEN SHIFT (Topological Impedance) ---
    # The speed of causal propagation (light) is inversely proportional to Zv.
    lumen_shift = 1.0 / (1.0 + total_resp)

    # Final GTS Gain Factor: Combined Geometric and Impedance effects.
    gain = (gamma**p) * lumen_shift

    return d_eff, gain, density


# --- DIAGNOSTIC VALIDATION ---
if __name__ == "__main__":
    print(f"\n{'GTS v10.3 STABILITY TEST':^65}")
    print("=" * 65)
    print(f"{'Physical Regime':<25} | {'D_eff':<8} | {'GTS Gain':<10}")
    print("-" * 65)

    # Test cases representing the three major phases of the GTS universe
    test_suite = [
        ("White Dwarf (Anomalous)", 27.0, 0.35),
        ("Solar System (Euclidean)", 12.4, 0.30),
        ("Galactic Outskirts (Halo)", 0.8, 0.70),
    ]

    for label, dens, clust in test_suite:
        de, g, _ = gts_engine_v10_3(dens, clustering=clust)
        regime = "STABLE" if 2.99 < de < 3.01 else "SHIFT"
        print(f"{label:<25} | {de:<8.2f} | {g:<9.4f}x ({regime})")

    print("=" * 65)
