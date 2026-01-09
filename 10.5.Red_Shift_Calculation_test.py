import pandas as pd
import numpy as np
import os
import glob

# ==============================================================================
# GTS OMEGA VALIDATOR
# ------------------------------------------------------------------------------
# Framework: Geometric Unification Theory (GTS)
# Purpose: Mass-scale validation of Topological Impedance and D_eff across
#          disparate astronomical catalogs.
# ==============================================================================

# --- GTS CORE PHYSICAL CONSTANTS ---
# Standard values for the Causal Substrate
A_0, G_SI, C = 1.2e-10, 6.67430e-11, 299792458
M_SUN, R_SUN, ALPHA_G = 1.98847e30, 6.957e8, 0.007297


def calculate_gts_omega(mass_sol, logg):
    """
    Core Omega Engine Logic based on the Impedance Bridge Axiom.
    Calculates the shift in Effective Dimension (Deff) and Gravitational Gain.
    """
    # 1. Derived Physical Parameters
    g = (10**logg) / 100.0  # Convert cgs log(g) to m/s^2
    r_m = np.sqrt((G_SI * mass_sol * M_SUN) / g)

    # 2. Causal Stress (Chi) & The Cubic Anchor
    # Anchors the Forbidden Zone (D=3) at the Solar System equilibrium point.
    chi = np.log10(g / A_0)
    tau = (chi - 12.4) * ALPHA_G

    # 3. Topological Resistance (Iv) & Emergent Dimension (Deff)
    iv = 1.0 + np.tanh(1200 * tau**3) * 0.15
    d_eff = 3.0 * iv

    # 4. Redshift & Gain Calculation
    v_einstein = (G_SI * mass_sol * M_SUN) / (r_m * C) / 1000.0
    gamma = 3.0 / d_eff

    # Impedance Bridge: Zv represents the causal propagation ease
    zv = iv / (1.0 / (1.0 - (iv - 1.0) * 0.5))

    # High-density pinning for degenerate matter
    p = 1.4 if d_eff >= 3.0 else (1.0 + 3.0 * (3.0 - d_eff) / (max(d_eff, 1.1) - 1.0))
    v_gts = v_einstein * (gamma**p) * zv

    return pd.Series([d_eff, v_einstein, v_gts, chi])


def run_master_validation(folder_path):
    # Scan for CSV and SVC files in the target directory
    file_patterns = ["*.csv", "*.svc"]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(os.path.join(folder_path, pattern)))

    if not files:
        print(f"❌ ERROR: No catalog files found in {folder_path}")
        return

    all_results = []

    # DYNAMIC IDENTIFIER MAPPING
    # Maps variable naming conventions to standard GTS inputs
    mapping = {
        "mass": ["mass", "m_msol", "wd_mass", "mass_sol", "m_star", "m"],
        "logg": ["logg", "log_g", "logg_atmo", "logg_star", "g_log"],
        # Added 'name' and 'source_id' to resolve 'nan' issues
        "id": [
            "dr3name",
            "source_id",
            "wdid",
            "source",
            "name",
            "identifier",
            "simbad",
            "source_name",
        ],
    }

    print(f"🚀 Initializing GTS Batch Processing on {len(files)} files...")

    for file in files:
        try:
            # Automatic separator detection to handle different CSV dialects
            df = pd.read_csv(file, sep=None, engine="python")
            df.columns = [c.lower().strip().replace("#", "") for c in df.columns]

            # Find valid columns based on the mapping dictionary
            found_cols = {
                target: next((c for c in candidates if c in df.columns), None)
                for target, candidates in mapping.items()
            }

            if found_cols["mass"] and found_cols["logg"]:
                # Clean dataset of missing physical values
                df = df.dropna(subset=[found_cols["mass"], found_cols["logg"]])

                # Execute GTS Omega Physics Engine
                df[["d_eff", "v_ein", "v_gts", "chi"]] = df.apply(
                    lambda r: calculate_gts_omega(
                        r[found_cols["mass"]], r[found_cols["logg"]]
                    ),
                    axis=1,
                )

                # PRESERVE IDENTIFIER: Use found ID or fallback to Index
                if found_cols["id"]:
                    df["final_id"] = df[found_cols["id"]].astype(str)
                else:
                    df["final_id"] = "Index_" + df.index.astype(str)

                df["source_file"] = os.path.basename(file)
                all_results.append(df)
                print(f"✅ Processed: {os.path.basename(file)} ({len(df)} objects)")
            else:
                print(f"⚠️ Skipped {file}: Missing essential mass/logg parameters.")

        except Exception as e:
            print(f"❌ CRITICAL ERROR in {file}: {e}")

    if all_results:
        # Concatenate all processed datasets into one unified Master Table
        master_df = pd.concat(all_results, ignore_index=True)

        print("\n" + "Ω" * 100)
        print(f" GTS GLOBAL ANOMALY REPORT | TOTAL POPULATION: {len(master_df)} ")
        print("=" * 100)

        # Display Top 15 Anomalies (Highest Dimensional Shifts)
        top_anomalies = master_df.sort_values(by="d_eff", ascending=False).head(15)

        header = f"{'OBJECT IDENTIFIER':<30} | {'D_EFF':<6} | {'CHI':<6} | {'SOURCE CATALOG'}"
        print(header)
        print("-" * 100)

        for _, row in top_anomalies.iterrows():
            print(
                f"{str(row['final_id'])[:30]:<30} | {row['d_eff']:>6.3f} | {row['chi']:>6.1f} | {row['source_file']}"
            )

        # Final Summary Statistics
        print("=" * 100)
        print(f"Analysis Complete. Average D_eff: {master_df['d_eff'].mean():.4f}")
        return master_df
    else:
        print("No valid data was aggregated.")
        return None


if __name__ == "__main__":
    # Configure your data folder path here
    DATA_PATH = r"C:\Users\pavel\Desktop\Kauzální fyzika"
    master_data = run_master_validation(DATA_PATH)
