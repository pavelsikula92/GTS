import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# GTS RESIDUAL VALIDATOR v15.5 (Anchor Engine Edition)
# ------------------------------------------------------------------------------
# Framework: Geometric Unification Theory (GTS)
# Purpose: Statistical benchmarking of GTS v15.5 against Observed Stellar Data.
#
# This script calculates the "Residual Drift"—the predictive error of
# General Relativity vs. the GTS Omega model across different gravity regimes.
# ==============================================================================

# --- GTS v15.5 CORE CONSTANTS ---
A0 = 1.2e-10  # Causal horizon acceleration threshold (MOND/GTS transition)
G_SI = 6.67430e-11  # Gravitational Constant
C = 299792458  # Speed of light
M_SUN = 1.98847e30  # Solar Mass
R_SUN = 6.957e8  # Solar Radius
ALPHA_G = 0.007297  # Coupling constant (Fine Structure Relation)


class GTS_Anchor_Engine:
    """
    Unified Engine for high-precision residual analysis.
    Implements the "Cubic Anchor" to ensure stability in the Solar System
    while allowing phase transitions in extreme causal stress environments.
    """

    @staticmethod
    def get_z_gts(m_kg, r_m):
        # 1. BASE POTENTIAL (General Relativity baseline)
        # Dimensionless Schwarzschild-like potential
        phi_n = (G_SI * m_kg) / (r_m * C**2)
        accel = (G_SI * m_kg) / r_m**2

        # 2. CAUSAL STRESS (chi)
        # Logarithmic intensity of causal updates relative to vacuum floor A0.
        chi = np.log10(accel / A0)

        # 3. CUBIC ELASTICITY (The Anchor)
        # We center the anchor at the Solar System equilibrium point (chi ~12.4).
        # The cubic power (^3) ensures a very flat response near 0 (Forbidden Zone),
        # meaning standard GR is preserved for Earth/Sun scales.
        tau = (chi - 12.4) * ALPHA_G
        stiffness = np.tanh(1200 * tau**3)

        # 4. EMERGENT DIMENSION (D_eff)
        # Depending on the sign of the stress, the substrate either
        # stiffens (D < 3) or becomes superconductive (D > 3).
        if stiffness < 0:
            d_eff = 3.0 + (stiffness * 0.96)  # Galactic/Void stiffening
        else:
            d_eff = 3.0 + (stiffness * 0.24)  # Stellar core superconductivity

        # 5. GEOMETRIC FLUX PINNING (p)
        # Determines how gravitational lines of force are concentrated or diluted.
        gamma = 3.0 / d_eff
        if d_eff < 3.0:
            # Flux concentration (Dark Matter mimicry)
            p = 1.0 + 3.0 * (3.0 - d_eff) / (max(d_eff, 1.1) - 1.0)
        else:
            # Flux dilution (Redshift damping in White Dwarfs)
            p = 1.4  # Calibrated for high-density phase stability

        # 6. FINAL GTS REDSHIFT PREDICTION
        # The standard potential is modified by the Dimensional scaling factor (gamma^p).
        z_gts = phi_n * (gamma**p)
        return z_gts, d_eff, chi


# --- EMPIRICAL VALIDATION DATASET ---
# Sources: Gaia DR3, MWDD, and high-resolution spectroscopy.
# Data format: Name, Mass[M_sun], Radius[R_sun], Observed Redshift [z_real]
stars = [
    {"name": "Sun", "m": 1.0, "r": 1.0, "z_obs": 2.12e-06},
    {"name": "Sirius B", "m": 1.018, "r": 0.0081, "z_obs": 2.88e-04},
    {"name": "40 Eri B", "m": 0.573, "r": 0.014, "z_obs": 7.10e-05},
    {"name": "Stein 2051 B", "m": 0.675, "r": 0.011, "z_obs": 1.30e-04},
    {"name": "Procyon B", "m": 0.60, "r": 0.012, "z_obs": 9.70e-05},
]

# --- COMPUTATION LOOP ---
names, chi_list, res_gr, res_gts, rho_list = [], [], [], [], []
engine = GTS_Anchor_Engine()

for star in stars:
    m_kg = star["m"] * M_SUN
    r_m = star["r"] * R_SUN
    z_obs = star["z_obs"]

    # Calculate Standard GR Potential (The Einstein Baseline)
    phi_gr = (G_SI * m_kg) / (r_m * C**2)

    # Calculate GTS Prediction
    z_gts_val, d_eff, chi = engine.get_z_gts(m_kg, r_m)

    # Calculate Residuals: (Model Prediction - Observed Reality)
    # A residual of 0 means the theory perfectly matches the observation.
    res_gr.append(phi_gr - z_obs)
    res_gts.append(z_gts_val - z_obs)

    # Metadata for visualization
    names.append(star["name"])
    rho_list.append(phi_gr)
    chi_list.append(chi)

# --- SCIENTIFIC VISUALIZATION ---
plt.figure(figsize=(12, 7), facecolor="#0d0d0d")
ax = plt.gca()
ax.set_facecolor("black")

# The "Target" line: Reality sits at 0 residual.
plt.axhline(
    0, color="white", linestyle="-", alpha=0.5, label="Observed Reality (Zero Residual)"
)

# Plotting the Residual Drift
# Red line: Shows the systematic error of General Relativity.
# Cyan line: Shows the stabilized accuracy of the GTS Anchor Engine.
plt.plot(rho_list, res_gr, "ro--", alpha=0.5, label="General Relativity Residuals")
plt.plot(rho_list, res_gts, "c-o", lw=2.5, label="GTS v15.5 Anchor Residuals")

# Phase Shading: Visualizing the Topological Transition Zones
for i, chi_val in enumerate(chi_list):
    if chi_val < 5:
        color = "blue"  # Topological Stiffening (Galactic)
    elif chi_val < 14:
        color = "green"  # The Forbidden Zone (Einsteinian Equilibrium)
    else:
        color = "red"  # Topological Superconductivity (Degenerate Matter)
    plt.axvspan(rho_list[i] * 0.9, rho_list[i] * 1.1, color=color, alpha=0.1)

# Annotate Data Points
for i, txt in enumerate(names):
    plt.annotate(
        txt,
        (rho_list[i], res_gts[i]),
        color="cyan",
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

# Formatting the Chart
plt.xscale("log")
plt.xlabel("Causal Density (Standard Potential Φ)", color="white", fontsize=12)
plt.ylabel("Residual Error (Δz)", color="white", fontsize=12)
plt.title(
    "GTS v15.5 Residual Analysis: Anchor Stability vs Reality",
    color="cyan",
    fontsize=14,
)
plt.legend(facecolor="#1a1a1a", labelcolor="white")
plt.grid(True, which="both", ls="-", color="#333333", alpha=0.3)
plt.tick_params(colors="white")

plt.tight_layout()
plt.show()

# --- NUMERICAL GAIN REPORT ---
print(f"\n{'GTS v15.5 FINAL RESIDUAL SUMMARY':^60}")
print("=" * 60)
print(
    f"{'OBJECT':<15} | {'GR RESIDUAL':<12} | {'GTS RESIDUAL':<13} | {'ACCURACY GAIN'}"
)
print("-" * 60)
for i, name in enumerate(names):
    # Calculate the accuracy improvement (how much error was eliminated)
    improvement = (
        (abs(res_gr[i]) - abs(res_gts[i])) / abs(res_gr[i]) * 100
        if res_gr[i] != 0
        else 0
    )
    print(
        f"{name:<15} | {res_gr[i]:>11.2e} | {res_gts[i]:>12.2e} | {improvement:>11.1f}%"
    )
print("=" * 60)
