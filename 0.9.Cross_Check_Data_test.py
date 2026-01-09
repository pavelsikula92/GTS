import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# GTS UNIVERSE BATCH VALIDATOR (Omega Core v16.5)
# ------------------------------------------------------------------------------
# Framework: Geometric Unification Theory (GTS)
# Purpose: Cross-catalog validation of the Omega Phase Transition.
#
# This script standardizes disparate datasets (MWDD, Gaia, etc.) and
# calculates the emergent effective dimension (Deff) for each star.
# ==============================================================================

# --- GTS OMEGA CORE CONSTANTS ---
A_0 = 1.2e-10  # Causal Horizon Acceleration (m/s^2)
G_SI = 6.67430e-11  # Gravitational Constant
C = 299792458  # Speed of Light
M_SUN = 1.98847e30  # Solar Mass
R_SUN = 6.957e8  # Solar Radius


def gts_engine(m, logg):
    """
    Calculates GTS metrics from mass and surface gravity (log g).

    1. Derives radius from mass and log(g).
    2. Computes Causal Stress (chi).
    3. Calculates Emergent Dimension (Deff) via the Omega Phase function.
    4. Computes Z-Ratio (The discrepancy factor vs. General Relativity).
    """
    # 1. Radius Derivation
    # log(g) in cgs -> convert to m/s^2: g = 10^logg / 100
    g = (10**logg) / 100
    r_m = np.sqrt((G_SI * m * M_SUN) / g)
    r_sol = r_m / R_SUN

    # 2. Causal Stress (chi)
    # Logarithmic intensity of causal updates relative to vacuum floor A_0.
    accel = (G_SI * m * M_SUN) / (r_m**2)
    chi = np.log10(accel / A_0)

    # 3. Effective Dimension (Omega Phase Transition)
    # This sigmoidal function models the shift toward D > 3.0 in high-density cores.
    # Center = 15.8 (Threshold for degenerate stellar matter).
    # Amplitude = 0.22 (Refined for high-mass WD anomalies).
    deff = 3.0 * (1.0 + 0.22 / (1.0 + np.exp(-0.8 * (chi - 15.8))))

    # 4. Z-Ratio (%)
    # Ratio of GTS gravitational potential vs. standard Einsteinian potential.
    # 100% = General Relativity (The Forbidden Zone).
    z_ratio = (3.0 / deff) * 100

    return pd.Series([r_sol, chi, deff, z_ratio])


# Automatic path detection - looks for the file in the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = "Jimenez-Esteban_2023_1767788491.csv"
FILE_PATH = os.path.join(script_dir, FILE_NAME)

if not os.path.exists(FILE_PATH):
    print(f"❌ Critical Error: Data source '{FILE_NAME}' not found in script directory!")
    print(f"Please ensure the CSV file is located at: {script_dir}")
    exit()

# List of target catalogs for batch processing
files = [
    "Jimenez-Esteban_2023_1767788491.csv",
    "MWDD_export.csv",
    "gaiaobser.csv",
]

all_results = []

print("🚀 Starting Universe Batch Validation...")

for file in files:
    path = os.path.join(DATA_FOLDER, file)
    if not os.path.exists(path):
        print(f"⚠️ Skipping: {file} (File not found)")
        continue

    print(f"📂 Processing catalog: {file}")
    df = pd.read_csv(path)

    # Header Normalization: lower-case and strip special characters
    df.columns = [c.strip().lower().replace("#", "") for c in df.columns]

    # --- COLUMN MAPPING ---
    # Different catalogs use different naming conventions for the same physics.
    # This map unifies them into 'mass' and 'logg'.
    col_map = {
        "m_msol": "mass",
        "mass [msun]": "mass",
        "log(g)": "logg",
        "logg_atmo": "logg",
        "wd_mass": "mass",
        "log_g": "logg",
    }
    df = df.rename(columns=col_map)

    # Validate that required physical inputs are present
    if "mass" in df.columns and "logg" in df.columns:
        # Prevent math errors by dropping NaN values
        df = df.dropna(subset=["mass", "logg"])

        # Execute GTS Physics Kernel
        # Result columns: [radius_sol, chi, d_eff, z_ratio]
        df[["radius_sol", "chi", "d_eff", "z_ratio"]] = df.apply(
            lambda row: gts_engine(row["mass"], row["logg"]), axis=1
        )

        # Track the source for comparative visualization
        df["source_file"] = file
        all_results.append(df)
    else:
        print(f"❌ Error: Required columns (mass/logg) not found in {file}.")
        print(f"Available columns: {df.columns.tolist()}")

# --- MASTER ANALYSIS ---
if all_results:
    # Merge all catalogs into a single unified GTS dataset
    master_df = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 80)
    print(f" GLOBAL ANOMALY REPORT (Total Population: {len(master_df)} Stars) ")
    print("=" * 80)

    # Sort by Effective Dimension to find the most significant topological defects
    top_10 = master_df.sort_values(by="d_eff", ascending=False).head(10)

    # Dynamic identification of the object ID column
    name_col = next(
        (
            c
            for c in ["dr3name", "identifier", "source", "name", "wdid"]
            if c in top_10.columns
        ),
        "index",
    )

    print(top_10[[name_col, "mass", "logg", "d_eff", "z_ratio", "source_file"]])
    print("=" * 80)

    # --- GLOBAL PHASE TRANSITION VISUALIZATION ---
    plt.figure(figsize=(12, 7), facecolor="#f5f5f5")

    for file in files:
        subset = master_df[master_df["source_file"] == file]
        if not subset.empty:
            plt.scatter(subset["chi"], subset["d_eff"], label=file, s=15, alpha=0.4)

    # Draw the Einsteinian Baseline (Standard Physics)
    plt.axhline(y=3.0, color="r", linestyle="--", label="Einsteinian Limit (D=3)")

    plt.title(
        "GTS Omega: Universal Phase Transition across Multi-Catalog Data", fontsize=14
    )
    plt.xlabel("Causal Stress (χ = log10(a / A0))")
    plt.ylabel("Effective Dimension (D_eff)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)

    # Highlight the Forbidden Zone Plateau
    plt.fill_between([0, 14], 2.9, 3.1, color="gray", alpha=0.1, label="Forbidden Zone")

    plt.show()
else:
    print("No data processed. Check file paths and column headers.")
