import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# Professional visualization setup
plt.style.use("dark_background")


class GTS_Unified_Engine:
    """
    GTS  - CAUSAL ANCHOR UPDATE
    --------------------------------
    Merges:
    1. Discrete Causal Graph (Micro-structure of Spacetime)
    2. Continuous Anchor Logic (Macro-physics of Dimension)

    Updates in v16.1:
    - Replaced log10 with natural logarithm (ln) for Chi calculation.
    - Implemented vacuum threshold A0 as a scale-invariant anchor.
    """

    def __init__(self, n_events=600):
        self.N = n_events
        # Initial state: 1D string of time
        self.G = nx.path_graph(n_events, create_using=nx.DiGraph)
        for u, v in self.G.edges():
            self.G[u][v]["w"] = 1.0

        # --- CONSTANTS ---
        self.alpha_g = 0.007297
        self.H_graph = 0.06  # Hubble Erosion Rate
        self.dyn_lambda = 1.2  # Coupling strength
        self.A_0 = 1.0  # Vacuum threshold (Scale-invariant anchor)

    def get_anchor_dimension(self, rho):
        """
        NATURAL LOGARITHMIC ANCHOR:
        --------------------------
        This is the core mapping from substrate density to effective dimension.
        Using ln(rho/A0) ensures the Forbidden Zone is centered at the vacuum baseline.
        """
        # 1. Normalize Density (Chi) using natural logarithm
        # Formula: chi = ln(rho_local / A_0)
        # We add 1e-9 for numerical stability to avoid ln(0).
        chi = np.log((rho + 1e-9) / self.A_0)

        # 2. Causal Saturation (Tau)
        # The factor 12.0 represents the 'stiffness' of the vacuum.
        # It determines the sensitivity to topological shifts.
        tau = chi * 12.0

        # 3. THE CUBIC ANCHOR
        # tau**3 creates the 'Newtonian Plateau' (Forbidden Zone).
        # This keeps D_eff = 3.0 in the solar-system regime.
        stiffness = np.tanh(tau**3)

        # 4. Emergent Dimension (D_eff)
        # Transitioning between 2D (dark matter) and 4D (singularities).
        d_eff = 3.0 + (stiffness * 1.0)

        # Clipped for numerical stability within the causal limit (1.5 to 4.0)
        return np.clip(d_eff, 1.5, 4.0)

    def get_local_metrics(self, node):
        """
        Extracts topological parameters from the local graph neighborhood.
        """
        neighbors = list(nx.all_neighbors(self.G, node))
        if len(neighbors) < 2:
            return 1.0, 3.0, 0.0  # Vacuum defaults

        sub = self.G.subgraph(neighbors + [node])
        nt = sub.number_of_edges()

        # Topological Density (Triangles per Edge)
        ns = sum(nx.triangles(nx.Graph(sub)).values()) // 3
        rho = (3 * ns) / nt if nt > 0 else 0

        # Calculate D_eff using the updated Anchor logic
        d_eff = self.get_anchor_dimension(rho)

        # Mass Potential (Input Flux)
        phi_v = sum(data["w"] for u, v_target, data in self.G.in_edges(node, data=True))

        return rho, d_eff, phi_v

    def calculate_action(self, node):
        """
        THE HAMILTONIAN (Total Energy Function)
        S = Entropy Cost - Binding Energy
        """
        rho, d_eff, phi_v = self.get_local_metrics(node)

        # A. Information Volume (Entropy Cost)
        iv = (rho * 10.0) / d_eff

        # B. Binding Energy (Flux Pinning)
        # The (3.0 / d_eff) term increases pull in low-D regimes (Dark Matter effect).
        binding = (self.dyn_lambda * phi_v * self.alpha_g * 10.0) * (3.0 / d_eff)

        return iv - binding

    def evolve(self):
        """
        Monte Carlo Time Step for Spacetime Growth
        """
        nodes = list(self.G.nodes())
        u, v = random.sample(nodes, 2)
        if u > v:
            u, v = v, u

        # PROCESS 1: HUBBLE EROSION (Entropic Decay)
        dist_factor = abs(u - v) / self.N
        if self.G.has_edge(u, v) and random.random() < (dist_factor * self.H_graph):
            self.G.remove_edge(u, v)
            return

        # PROCESS 2: METROPOLIS-HASTINGS (Action Minimization)
        s_old = self.calculate_action(u)

        existed = self.G.has_edge(u, v)
        if existed:
            self.G.remove_edge(u, v)
        else:
            self.G.add_edge(u, v, w=1.0)

        # Causal Constraint: The graph must remain a DAG
        if not nx.is_directed_acyclic_graph(self.G):
            if existed:
                self.G.add_edge(u, v, w=1.0)
            else:
                self.G.remove_edge(u, v)
            return

        s_new = self.calculate_action(u)
        delta_s = s_new - s_old

        # Acceptance based on S minimization
        if delta_s > 0 and random.random() > np.exp(-delta_s):
            # Revert change if it increases entropy too much
            if existed:
                self.G.add_edge(u, v, w=1.0)
            else:
                self.G.remove_edge(u, v)

    def simulate(self, steps=100):
        """
        Executes the GTS simulation and tracks mass center stability.
        """
        print(f"{'STEP':<6} | {'DIST':<6} | {'MASS':<6} | {'D_eff':<6} | {'STATUS'}")
        print("-" * 55)

        # Initial Thermalization
        for _ in range(2000):
            self.evolve()

        # Identify high-density center
        try:
            center = max(self.G.nodes(), key=lambda n: self.get_local_metrics(n)[2])
            particle = random.choice(
                [n for n in self.G.nodes() if abs(n - center) > self.N * 0.2]
            )
            initial_dist = nx.shortest_path_length(
                self.G.to_undirected(), particle, center
            )
        except:
            print("Simulation failed to stabilize initial center.")
            return

        for t in range(steps):
            for _ in range(400):
                self.evolve()

            try:
                dist = nx.shortest_path_length(self.G.to_undirected(), particle, center)
                rho, d_eff, mass = self.get_local_metrics(center)

                status = "STABLE"
                if dist < initial_dist:
                    status = "CAPTURED"
                elif dist > initial_dist:
                    status = "DRIFTING"

                print(f"{t:<6} | {dist:<6} | {mass:<6.1f} | {d_eff:<6.3f} | {status}")

            except:
                print(
                    f"{t:<6} | {'LOST':<6} | {'---':<6} | {'---':<6} | HUBBLE HORIZON"
                )
                break


if __name__ == "__main__":
    # n_events increased for better resolution of the ln-plateau
    engine = GTS_Unified_Engine(n_events=500)
    engine.simulate()
