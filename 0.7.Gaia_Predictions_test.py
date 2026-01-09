import pandas as pd
import numpy as np
from astroquery.gaia import Gaia
import os
import matplotlib.pyplot as plt

# ==============================================================================
# GTS GAIA LIVE VALIDATOR v16.1
# ------------------------------------------------------------------------------
# Theoretical Framework: Geometric Unification Theory (GTS)
# Author: Pavel Sikula (Independent Researcher)
#
# DESCRIPTION:
# This script performs a live cross-match between the GTS theoretical framework
# and the ESA Gaia DR3 archive. It targets high-mass White Dwarfs (WD) to
# demonstrate the systematic redshift discrepancy predicted by the
# Topological Impedance model.
# ==============================================================================

# --- GTS FUNDAMENTAL CONSTANTS (Grounded in Causal-Topological Physics) ---
A_0 = 1.2e-10  # Causal Horizon Acceleration (MOND/GTS threshold)
G_SI = 6.67430e-11  # Gravitational Constant
C = 299792458  # Speed of Light
M_SUN = 1.98847e30  # Solar Mass
R_SUN = 6.957e8  # Solar Radius
ALPHA_G = 0.007297  # Fine Structure Constant (used as topological coupling)


def get_gts_prediction(m_sol, r_sol):
    """
    GTS v16.1 Physics Engine: Topological Impedance & Dimension Shift.

    Computes:
    1. Standard GR Redshift (Einstein Baseline)
    2. Emergent Effective Dimension (D_eff)
    3. GTS Corrected Redshift (Omega Prediction)
    """
    # 1. Standard General Relativity (GR) Baseline
    phi_n = (G_SI * m_sol * M_SUN) / (r_sol * R_SUN * C**2)
    v_gr_einstein = phi_n * C / 1000  # Gravitational Redshift in km/s

    # 2. Local Causal Stress (Chi)
    # The pressure on the substrate relative to the vacuum floor A_0.
    accel = (G_SI * m_sol * M_SUN) / (r_sol * R_SUN) ** 2
    chi = np.log10(accel / A_0)

    # 3. Topological Resistance (Iv) & Emergent Dimension (D_eff)
    # The "tau^3" anchor creates the Forbidden Zone (Newtonian plateau).
    # 1200 is the Gain factor, 12.4 is the Solar Anchor point.
    tau = (chi - 12.4) * ALPHA_G
    iv = 1.0 + np.tanh(1200 * tau**3) * 0.15
    deff = 3.0 * iv

    # 4. Topological Impedance (Pi_v) & Impedance Bridge (Zv)
    # This models the "Superconductivity" phase where gravity leaks into higher-D.
    pi_v = 1.0 / (1.0 - (iv - 1.0) * 0.5)
    zv = iv / pi_v

    # 5. Final GTS Prediction (Flux Pinning Logic)
    gamma = 3.0 / deff
    # Exponent p determines flux concentration/dilution.
    p = 1.4 if deff >= 3.0 else (1.0 + 3.0 * (3.0 - deff) / (max(deff, 1.1) - 1.0))

    v_gts = v_gr_einstein * (gamma**p) * zv

    return deff, v_gr_einstein, v_gts


# --- DATA ACQUISITION & PROCESSING ---
# Path to your local database of White Dwarfs
FILE_PATH = r"C:\Users\yxyx\yxyx\Jimenez-Esteban_2023_1767788491.csv"

if not os.path.exists(FILE_PATH):
    print("❌ Critical Error: Data source not found!")
    exit()

# Load and clean headers
df = pd.read_csv(FILE_PATH)
df.columns = [c.lower().replace("#", "") for c in df.columns]

# Derive Stellar Radius from log(g) and Mass
g_m_s2 = 10 ** (df["logg"] - 2)
df["radius"] = np.sqrt((G_SI * df["mass"] * M_SUN) / g_m_s2) / R_SUN

# Select top 100 high-mass objects for anomaly testing
analysis_df = df.sort_values(by="mass", ascending=False).head(100).copy()


def fetch_gaia_data(subset):
    """
    PULSE MODE QUERY: Fetches real-time Radial Velocity (RV) from Gaia DR3.
    Batched processing to ensure connection stability with ESA servers.
    """
    raw_ids = ["".join(filter(str.isdigit, str(x))) for x in subset["dr3name"]]
    raw_ids = [rid for rid in raw_ids if rid]

    batch_size = 20
    all_results = []

    print(f"📡 Initializing Pulse Mode: Querying Gaia for {len(raw_ids)} objects...")

    for i in range(0, len(raw_ids), batch_size):
        batch = raw_ids[i : i + batch_size]
        ids_str = ",".join(batch)

        # ADQL Query: Matching our local WD IDs with Gaia's observational RV
        query = f"""
        SELECT source_id, radial_velocity, radial_velocity_error 
        FROM gaiadr3.gaia_source 
        WHERE source_id IN ({ids_str}) 
        AND radial_velocity IS NOT NULL
        """

        try:
            print(f"   > Batch {i//batch_size + 1}: Querying ESA archive...")
            job = Gaia.launch_job(query)
            batch_df = job.get_results().to_pandas()
            all_results.append(batch_df)
        except Exception as e:
            print(f"   ⚠️ Batch Error: {e}")
            continue

    if not all_results:
        return pd.DataFrame(
            columns=["source_id", "radial_velocity", "radial_velocity_error"]
        )

    return pd.concat(all_results).drop_duplicates(subset="source_id")


if __name__ == "__main__":
    # Live data cross-match
    gaia_db = fetch_gaia_data(analysis_df)
    results = []

    print(f"📊 Validating Theoretical Predictions against Gaia data...")

    for _, row in analysis_df.iterrows():
        try:
            sid = int("".join(filter(str.isdigit, str(row["dr3name"]))))
            match = gaia_db[gaia_db["source_id"] == sid]

            # Compute GTS Prediction for every star in the subset
            deff, v_ein, v_gts = get_gts_prediction(row["mass"], row["radius"])

            res_entry = {
                "id": sid,
                "deff": deff,
                "einstein": v_ein,
                "omega": v_gts,
                "gaia": np.nan,
                "err": 0,
            }

            # Map Gaia RV if a match exists in the DR3 archive
            if not match.empty:
                rv_real = match.iloc[0]["radial_velocity"]
                rv_err = match.iloc[0]["radial_velocity_error"]
                if not np.isnan(rv_real):
                    res_entry["gaia"] = rv_real
                    res_entry["err"] = rv_err

            results.append(res_entry)
        except Exception:
            continue

    res_df = pd.DataFrame(results)

    # --- FINAL VISUALIZATION (THE SABINE CONFRONTATION) ---
    plt.figure(figsize=(14, 7), facecolor="#f0f0f0")

    # Plotting Observational Data
    valid_gaia = res_df.dropna(subset=["gaia"])
    if not valid_gaia.empty:
        plt.errorbar(
            valid_gaia.index,
            valid_gaia["gaia"],
            yerr=valid_gaia["err"],
            fmt="o",
            label="Gaia DR3 Radial Velocity (Observed)",
            color="black",
            alpha=0.7,
        )

    # Plotting General Relativity Prediction
    plt.plot(
        res_df.index,
        res_df["einstein"],
        "r--",
        label="General Relativity (Einstein)",
        alpha=0.8,
    )

    # Plotting GTS Topological Prediction
    plt.plot(
        res_df.index,
        res_df["omega"],
        "g-",
        linewidth=2.5,
        label="GTS Prediction (Omega Engine)",
    )

    plt.title(
        "GTS v16.1: Confronting General Relativity with Gaia DR3 Empirical Data",
        fontsize=15,
    )
    plt.xlabel("Target Star Index (High-Mass White Dwarfs)")
    plt.ylabel("Radial Velocity / Gravitational Redshift [km/s]")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    print("\nProcessing complete. Displaying verification plot...")
    plt.show()

    # Display raw numerical comparison for the first 20 stars
    print(res_df[["id", "deff", "einstein", "omega", "gaia"]].head(20))
