import pandas as pd
import numpy as np
from astroquery.vizier import Vizier
import os
import matplotlib.pyplot as plt

# ==============================================================================
# GTS UNIVERSAL VALIDATOR (Omega Kernel v16.5)
# ------------------------------------------------------------------------------
# Framework: Geometric Unification Theory (GTS)
# Features: Cubic Anchor, Impedance Bridge (Zv), and Gaia DR3 Live Sync.
# ==============================================================================

# --- GTS v15.6 OMEGA CONSTANTS ---
A_0 = 1.2e-10  # Causal Horizon Acceleration (m/s^2)
G_SI = 6.67430e-11  # Gravitational constant
C_0 = 299792458.0  # Standard speed of light
M_SUN = 1.98847e30
R_SUN = 6.957e8
ALPHA_G = 0.007297  # Coupling constant (fine-structure relation)


def get_gts_omega_metrics(mass_sol, logg):
    """
    Core GTS Omega Physics Engine.
    Calculates emergent dimensionality and topological impedance.
    """
    # 1. Surface gravity (logg cgs -> m/s^2)
    g = (10**logg) / 100.0
    r_m = np.sqrt((G_SI * mass_sol * M_SUN) / g)

    # 2. Causal Stress (chi) - Logarithmic pressure on the substrate
    chi = np.log10(g / A_0)

    # 3. TOPOLOGICAL RESISTANCE (Iv) - The Cubic Anchor Logic
    # Ensures D=3.0 (Forbidden Zone) for Solar-system scales.
    tau = (chi - 12.4) * ALPHA_G
    iv = 1.0 + np.tanh(1200 * tau**3) * 0.15

    # 4. CAUSAL PERMEABILITY (Pi_v) & IMPEDANCE (Zv)
    # Models how easily causal edges form under stress.
    pi_v = 1.0 / (1.0 - (iv - 1.0) * 0.5)
    zv = iv / pi_v

    # 5. EFFECTIVE CAUSAL SPEED (ceff)
    # The local speed of "light" is modified by topological impedance.
    c_eff = C_0 / np.sqrt(zv)

    # 6. EMERGENT DIMENSION (Deff)
    deff = 3.0 * iv

    # 7. REDSHIFT CALCULATION (Einstein GR vs. GTS Omega)
    v_gr_einstein = (G_SI * mass_sol * M_SUN) / (r_m * C_0) / 1000.0  # km/s

    gamma = 3.0 / deff
    # Exponent p: Calibrated for White Dwarf high-density pinning
    p = 1.4 if deff >= 3.0 else (1.0 + 3.0 * (3.0 - deff) / (max(deff, 1.1) - 1.0))

    # Final GTS prediction: Geometric Boost * Impedance Bridge Shift
    v_gts = v_gr_einstein * (gamma**p) * zv

    return deff, v_gr_einstein, v_gts, c_eff / 1000.0


# --- DATA LOADING (Robust MWDD-Safe Approach) ---
FILE_PATH = r"C:\Users\pavel\Desktop\Kauzální fyzika\MWDD_export.csv"
if not os.path.exists(FILE_PATH):
    print(f"❌ ERROR: File {FILE_PATH} not found!")
    exit()

# Load and clean headers (standardizing for case-insensitivity and whitespace)
df = pd.read_csv(FILE_PATH)
df.columns = [c.lower().strip().replace("#", "") for c in df.columns]

# --- DYNAMIC COLUMN DETECTION ---
# Automatically find the ID column regardless of whether it's 'dr3name', 'source_id', or 'wdid'
id_col = next(
    (c for c in df.columns if any(x in c for x in ["dr3", "source", "wdid", "name"])),
    None,
)
if not id_col:
    print(f"❌ ERROR: ID column not found! Available: {df.columns.tolist()}")
    exit()
print(f"✅ Using '{id_col}' as primary object identifier.")

# Data Cleaning: Drop rows missing crucial physical parameters
df = df.dropna(subset=["mass", "logg"])

# Sampling: Select top 100 high-mass WD anomalies for stress testing
sampled_df = df.sort_values(by="mass", ascending=False).head(100).copy()


# --- VIZIER GAIA SYNC (Gentile Fusillo 2021 Catalog) ---
def fetch_ground_truth_rv(subset):
    """Queries VizieR for real-world Gaia DR3 Radial Velocity measurements."""
    print(f"📡 Querying VizieR for {len(subset)} targets...")
    v = Vizier(columns=["Source", "RV", "e_RV"], catalog="J/MNRAS/508/3877/table1")
    v.ROW_LIMIT = -1
    ids = [str("".join(filter(str.isdigit, str(x)))) for x in subset[id_col]]
    try:
        results = v.query_constraints(Source=",".join(ids))
        return results[0].to_pandas() if results else pd.DataFrame()
    except Exception as e:
        print(f"❌ VizieR Connection Error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Fetch observational data
    ground_db = fetch_ground_truth_rv(sampled_df)
    results_list = []

    print("\n" + "Ω" * 105)
    print(
        f"{'OBJECT ID':<20} | {'DEFF':<6} | {'V_GR':<8} | {'V_GTS':<8} | {'CEFF':<9} | {'REAL GAIA RV'}"
    )
    print("-" * 105)

    for _, row in sampled_df.iterrows():
        # Clean ID for matching
        sid_str = "".join(filter(str.isdigit, str(row[id_col])))
        if not sid_str:
            continue
        sid = int(sid_str)

        # Calculate GTS predictions using the Omega Engine
        deff, v_ein, v_gts, ceff_km = get_gts_omega_metrics(row["mass"], row["logg"])

        real_rv_str = "N/A"
        if not ground_db.empty:
            match = ground_db[ground_db["Source"] == sid]
            if not match.empty and not np.isnan(match.iloc[0]["RV"]):
                rv, err = match.iloc[0]["RV"], match.iloc[0]["e_RV"]
                real_rv_str = f"{rv:>7.2f} ± {err:.1f}"
                results_list.append(
                    {"ein": v_ein, "gts": v_gts, "real": rv, "err": err, "deff": deff}
                )

        print(
            f"{sid:<20} | {deff:>6.3f} | {v_ein:>8.2f} | {v_gts:>8.2f} | {ceff_km:>9.1f} | {real_rv_str}"
        )

    # --- FINAL VISUALIZATION ---
    if results_list:
        res_df = pd.DataFrame(results_list)
        plt.figure(figsize=(12, 6), facecolor="#f5f5f5")
        plt.errorbar(
            res_df.index,
            res_df["real"],
            yerr=res_df["err"],
            fmt="ko",
            label="Gaia DR3 Observed RV",
            alpha=0.5,
        )
        plt.plot(
            res_df.index,
            res_df["ein"],
            "r--",
            label="Einstein (General Relativity)",
            alpha=0.8,
        )
        plt.plot(
            res_df.index,
            res_df["gts"],
            "g-",
            linewidth=2.5,
            label="GTS Omega (Topological)",
        )

        plt.title("GTS Omega v16.5: Empirical Confrontation (High-Mass White Dwarfs)")
        plt.ylabel("Radial Velocity [km/s]")
        plt.xlabel("Star Index (Sorted by Mass)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.show()
    else:
        print(
            "\n⚠️ No Gaia RV matches found. Check your Internet connection or ID formatting."
        )
