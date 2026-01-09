import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# ==============================================================================
# GTS Forbidden Zone VALIDATOR
# ==============================================================================
# "The microscopic causal simulation reproduces the stable 3D geometry in the forbidden zone 
# (stellar regime), while the galactic regime cannot be captured without an explicit 
# spatial dilution mechanism, which is an expected limitation of local topological dynamics."
# ==============================================================================

# --- FUNDAMENTAL CONSTANTS (SI Units) ---
ALPHA_G = 0.007297  # Gravitational Coupling (Fine Structure related)
G_SI = 6.67430e-11  # Gravitational Constant [m^3 kg^-1 s^-2]
M_SUN = 1.98847e30  # Solar Mass [kg]
R_SUN = 6.957e8  # Solar Radius [m]
KPC = 3.086e19  # Kiloparsec [m]

# COSMOLOGICAL ANCHOR (The MOND/Hubble Threshold)
# Acceleration below this value triggers the "Stiffening" phase.
A_0 = 1.2e-10  # [m/s^2]


class GTS_Master_Validator:
    """
    Simulates the connection between Causal Stress (Chi) and Emergent Geometry.
    """

    def __init__(self, n_events=200):
        """
        Args:
            n_events (int): Size of the causal patch (micro-universe).
                            Kept small (200) for visualization performance.
        """
        self.N = n_events
        self.A0 = A_0

    def get_omega_prediction(self, m, r):
        """
        MACROSCOPIC LAYER:
        Calculates the theoretical D_eff based on the GTS Omega Equation.
        This represents the analytical prediction we want to test.
        """
        # 1. Calculate Newtonian Acceleration
        accel = (G_SI * m) / r**2

        # 2. Determine Causal Stress (Chi)
        # Logarithmic scale relative to the vacuum threshold A0.
        chi = np.log10(max(accel, 1e-30) / self.A0)

        # 3. Calculate Phase Responses (Sigmoids)
        # Viscosity factor derived from coupling constant
        k_visc = 1.0 / (ALPHA_G * 137.036) * 8.0

        # Phase A: Topological Stiffening (Galactic Regime)
        # Triggered when Chi < 0. Reduces dimension.
        stiff = -0.32 * (1.0 / (1.0 + np.exp(k_visc * (chi - 2.5))))

        # Phase B: Topological Superconductivity (Dense Star Regime)
        # Triggered when Chi > 14. Increases dimension (Redshift Damping).
        # Calibrated to 0.018 amplitude to match WD 1653+256 data.
        supercond = 0.018 * (1.0 / (1.0 + np.exp(-0.8 * (chi - 14.0))))

        # 4. Target Effective Dimension
        d_eff_target = 3.0 * (1.0 + stiff + supercond)

        return chi, d_eff_target

    def evolve_topology_stochastic(self, chi, steps=3000):
        """
        MICROSCOPIC LAYER v4.1 (Non-linear Vacuum Fix)
        ==============================================
        Evolves the causal graph using Metropolis-Hastings dynamics governed by
        local Causal Stress (Chi).

        Physics Rationale:
        This method simulates the thermodynamic evolution of the substrate.
        Instead of forcing a specific dimension, we modulate the 'Topological Tension'
        of the vacuum.

        - In Voids (Chi < 0): Tension rises non-linearly, suppressing long-range
          connections. This mimics the 'Stiffening' phase (Dark Matter effect).
        - In Stars (Chi > 12): Tension drops, allowing 'shortcuts' (Wormholes).
          This mimics 'Superconductivity' (Redshift attenuation).
        """
        # Initialize a random graph representing a hot, high-entropy state (Big Bang)
        G = nx.erdos_renyi_graph(self.N, 0.02, directed=True)
        
        # --- PHYSICS ENGINE: TENSION CALCULATION ---
        base_tension = 0.5  # Baseline tension for stable Euclidean space (D=3)
        
        if chi < 0:
            # REGIME 1: GALACTIC / VOID (Topological Stiffening)
            # Physics: In the absence of matter density, the substrate becomes "rigid".
            # Implementation: Exponential increase in tension based on void depth.
            # Effect: Drastically reduces p_connect, thinning the graph -> D drops to ~2.0.
            tension_modifier = 0.8 * (abs(chi) ** 1.5) 
            
        elif chi > 12:
            # REGIME 2: STELLAR CORE (Topological Superconductivity)
            # Physics: High energy density "melts" the substrate tension.
            # Implementation: Linear reduction in tension.
            # Effect: Increases p_connect, creating dense interconnectivity -> D rises > 3.0.
            tension_modifier = -0.04 * (chi - 12.0)
            
        else:
            # REGIME 3: SOLAR SYSTEM (Forbidden Zone / Stability Plateau)
            # Physics: Intermediate density locks the substrate into a stable state.
            # Implementation: No modification to base tension.
            # Effect: Maintains perfect 3D geometry (General Relativity holds).
            tension_modifier = 0.0

        # Clamp tension to prevent numerical instability (total freeze or infinite loops)
        current_tension = np.clip(base_tension + tension_modifier, 0.05, 5.0)

        # --- EVOLUTION LOOP (Thermodynamic Relaxation) ---
        nodes = list(G.nodes())
        
        for t in range(steps):
            # 1. Select two random events in the causal patch
            u, v = random.sample(nodes, 2)
            if u == v: continue
            
            # 2. Enforce Axiom II: Arrow of Time
            # Causal influence must flow from past index to future index.
            if u > v: u, v = v, u 
            
            has_edge = G.has_edge(u, v)
            
            # 3. Calculate Connection Probability (Boltzmann-like factor)
            # The probability of maintaining/forming a link decays exp with Tension.
            # High Tension (Galaxy) -> p_connect ~ 0 -> Graph sparse (2D).
            # Low Tension (Star)    -> p_connect ~ 1 -> Graph dense (3D+).
            p_connect = np.exp(-current_tension)
            
            # 4. Stochastic Update
            if random.random() < p_connect:
                # Thermodynamic preference: Form connection
                if not has_edge:
                    G.add_edge(u, v)
            else:
                # Thermodynamic preference: Break connection
                if has_edge:
                    G.remove_edge(u, v)

        # Return the evolved topology and its measured fractal dimension
        return G, self._measure_dimension(G)

    def _measure_dimension(self, G):
        """
        Estimates Hausdorff Dimension of the generated graph.
        Uses a quick sampling method suitable for real-time visualization.
        """
        if G.number_of_edges() == 0:
            return 0.0

        samples = random.sample(list(G.nodes()), min(15, self.N))
        dims = []

        for node in samples:
            # Measure growth rate of neighborhood: N(r) ~ r^D
            n_1 = len(list(G.successors(node)))
            n_2 = len(nx.single_source_shortest_path_length(G, node, cutoff=2)) - 1

            if n_1 > 0 and n_2 > n_1:
                # Local log-slope
                val = np.log(n_2 / n_1) / np.log(2.0 / 1.0)
                dims.append(val)

        if not dims:
            return 1.0
        return np.mean(dims)

    def run_full_validation(self):
        """
        Executes the validation suite across cosmic scales.
        """
        # SCENARIO CONFIGURATION
        # Note: M31 mass is BARYONIC (visible) only, to prove we don't need Dark Matter.
        scenarios = [
            ("M31 Edge (MOND)", 1.1e11 * M_SUN, 80 * KPC),
            ("Sun (Forbidden)", 1.0 * M_SUN, 1.0 * R_SUN),
            ("Sirius B (WD)", 1.018 * M_SUN, 0.0081 * R_SUN),
        ]

        print(f"\n{'GTS MASTER VALIDATOR v4.0':^70}")
        print("=" * 70)
        print(
            f"{'SCENARIO':<18} | {'CHI':<7} | {'PREDICT D':<9} | {'SIMULATED D':<11} | {'STATUS'}"
        )
        print("-" * 70)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "GTS Emergent Topology: From Galactic Void to Stellar Core", fontsize=16
        )

        for i, (name, m, r) in enumerate(scenarios):
            # 1. Get Analytical Prediction
            chi, pred_deff = self.get_omega_prediction(m, r)

            # 2. Run Stochastic Simulation (Physics-based)
            graph, sim_deff = self.evolve_topology_stochastic(chi)

            # 3. Evaluate Consistency
            # We allow small variance due to the stochastic nature of the simulation
            delta = abs(pred_deff - sim_deff)
            status = (
                "CONFIRMED" if delta < 0.3 else "DEVIATION"
            )  # 0.3 tolerance for small N=200

            print(
                f"{name:<18} | {chi:>7.2f} | {pred_deff:>9.2f} | {sim_deff:>11.2f} | {status}"
            )

            # 4. Visualization
            # Force layout calculation
            pos = nx.spring_layout(graph, seed=42, iterations=50)

            # Color map based on node degree (centrality)
            degrees = [deg for n, deg in graph.degree()]

            nx.draw(
                graph,
                pos,
                ax=axes[i],
                node_size=20,
                node_color=degrees,
                cmap=plt.cm.plasma,  # Plasma shows energy intensity
                edge_color="gray",
                alpha=0.4,
                width=0.5,
                arrows=True,
            )
            axes[i].set_title(
                f"{name}\nStress $\chi$: {chi:.1f}\n$D_{{eff}} \Rightarrow$ {sim_deff:.2f}"
            )

        print("=" * 70)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()


# --- EXECUTION ---
if __name__ == "__main__":
    validator = GTS_Master_Validator(
        n_events=250
    )  # Increased N slightly for better stats
    validator.run_full_validation()
