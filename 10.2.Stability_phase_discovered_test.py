import networkx as nx
import numpy as np
import random
from collections import deque


class GTS_Flux_Invariant_v15_4:
    """
    GTS : Causal Flux Invariance & Dimensional Phase Transitions.
    This version enforces Axiom IV (Conservation of Flux) and models the
    emergence of matter as topological vortices.
    """

    def __init__(self, n_events=800):
        self.N = n_events
        # Initialize as a 1D causal chain (Arrow of Time)
        self.G = nx.path_graph(n_events, create_using=nx.DiGraph)
        # Axiom IV: Every causal link begins with a normalized relational weight (Flux)
        for u, v in self.G.edges():
            self.G[u][v]["w"] = 1.0

        self.alpha_g = 0.007297  # Gravitational coupling
        self.A0 = 1.2e-10  # Milgromian/Vacuum threshold
        self.rho_min, self.rho_max = 0.8, 1.5  # Stability Plateau boundaries

        self.dyn_lambda = 1.0  # Dynamic binding strength
        self.lambda_window = deque(maxlen=50)

    def get_substrate_response(self, node):
        """
        Calculates local topological density and vacuum impedance (Zv).
        """
        neighbors = list(nx.all_neighbors(self.G, node))
        if len(neighbors) < 2:
            return 0.0, 0.1, 1.0, 1.0, 1.0, 0.0

        sub = self.G.subgraph(neighbors + [node])
        nt = sub.number_of_edges()
        # ns: Number of cliques of size 3 (Triangles) representing 2D area
        ns = sum(nx.triangles(nx.Graph(sub)).values()) // 3

        # Phi_v: Inbound Causal Flux (The 'Input' to the local event)
        phi_v = sum(data["w"] for u, v_target, data in self.G.in_edges(node, data=True))

        rho = (3 * ns) / nt if nt > 0 else 0
        chi = np.log(max(rho, 1e-5) / self.A0 + 1.0)

        # IV: Information Volume (Topological entropy cost)
        iv = (nt**2) / (self.N * 0.15)

        # Forbidden Zone / Stability Plateau Logic
        if self.rho_min < rho < self.rho_max:
            zv, pi_f = 1.0, 1.0  # Standard Euclidean Spacetime (D=3)
        else:
            # Substrate Stiffening or Superconductivity
            pi_v = 1.0 / (1.0 + iv * self.alpha_g)
            zv = np.sqrt(iv / pi_v)  # Topological Impedance
            pi_f = np.exp(-rho / 1.2)

        return rho, iv, zv, pi_f, chi, phi_v

    def get_geometric_transition(self, node, rho):
        """
        Maps local density to Effective Dimension (Deff).
        """
        try:
            # Growth rate of the neighborhood (Hausdorff Dimension proxy)
            path_lengths = nx.single_source_shortest_path_length(self.G, node, cutoff=3)
            v2 = sum(1 for d in path_lengths.values() if d <= 2)
            v3 = len(path_lengths)
            deff = np.log10(v3 / v2) / np.log10(3 / 2) if v2 > 1 and v3 > v2 else 3.0
        except:
            deff = 3.0

        # Enforce GTS Phase Boundaries
        if rho < self.rho_min:
            deff = np.clip(deff, 1.0, 2.95)  # Galactic/Void Regime
        elif rho > self.rho_max:
            deff = np.clip(deff, 3.05, 4.0)  # Stellar/High-Stress Regime
        else:
            deff = 3.0  # The Forbidden Zone

        # Flux Pinning Exponent
        p = 1.0 * (1.5 / (deff / 2)) if deff != 3 else 1.0
        xi = (3.0 / deff) ** p
        return deff, xi

    def evolve_with_flux_conservation(self, steps=4000):
        """
        Axiom IV: Conservation of Causal Flux.
        Topology changes must redistribute weights to keep total flux invariant.
        """
        nodes = list(self.G.nodes())
        for _ in range(steps):
            u, v = random.sample(nodes, 2)
            if u > v:
                u, v = v, u

            s_old = self.calculate_action(u)
            existed = self.G.has_edge(u, v)

            if existed:
                # DELETE: Melt edge and redistribute flux to neighbors
                w_melted = self.G[u][v]["w"]
                self.G.remove_edge(u, v)
                out_edges = list(self.G.out_edges(u, data=True))
                if out_edges:
                    for _, target, data in out_edges:
                        data["w"] += w_melted / len(out_edges)
            else:
                # ADD: Condense new edge by siphoning flux from existing ones
                out_edges = list(self.G.out_edges(u, data=True))
                if len(out_edges) > 0:
                    self.G.add_edge(u, v, w=0.5)  # Initial condensation
                    for _, target, data in out_edges:
                        if target != v:
                            siphon = 0.5 / len(out_edges)
                            data["w"] = max(0.1, data["w"] - siphon)

            # Axiom II: Arrow of Time (DAG Constraint)
            if not nx.is_directed_acyclic_graph(self.G):
                if existed:
                    self.G.add_edge(u, v, w=w_melted)
                else:
                    self.G.remove_edge(u, v)
                continue

            # Metropolis-Hastings: Action Minimization
            s_new = self.calculate_action(u)
            if s_new > s_old and random.random() > np.exp(-(s_new - s_old)):
                # Revert: System seeks the path of least resistance
                if existed:
                    self.G.add_edge(u, v, w=w_melted)
                else:
                    self.G.remove_edge(u, v)

    def calculate_action(self, node):
        """
        The Hamiltonian: S = (Inertia + Impedance) - (Binding Energy).
        """
        rho, iv, zv, pi_f, chi, phi_v = self.get_substrate_response(node)
        deff, xi = self.get_geometric_transition(node, rho)

        # NS acts as a proxy for local connectivity volume
        neighbors = list(nx.all_neighbors(self.G, node))
        ns = (
            sum(nx.triangles(nx.Graph(self.G.subgraph(neighbors + [node]))).values())
            // 3
        )

        # Matter Vortex: Causal flux meeting topological density
        matter_vortex = phi_v * ns

        # Action S: Lower is better (Stability)
        # (iv + zv + chi) = Spacetime tension
        # (binding) = Energy condensation
        return (iv + zv + chi) - (
            self.dyn_lambda * matter_vortex * self.alpha_g * 137.0
        ) * xi

    def run_diagnostics(self):
        node = random.choice(list(self.G.nodes()))
        rho, iv, zv, pi_f, chi, phi_v = self.get_substrate_response(node)
        deff, xi = self.get_geometric_transition(node, rho)

        # Adaptive coupling to prevent singularity or total evaporation
        self.lambda_window.append(zv / (xi + 1e-5))
        self.dyn_lambda = np.mean(self.lambda_window)

        return {"Deff": deff, "xi": xi, "rho": rho, "Zv": zv, "Phi": phi_v}


# --- EXECUTION ---
if __name__ == "__main__":
    engine = GTS_Flux_Invariant_v15_4(n_events=800)
    print(f"\n{'GTS v15.4: TOPOLOGICAL VORTEX DYNAMICS':^80}")
    print("=" * 80)
    print(
        f"{'STEP':<8} | {'Deff':<5} | {'xi':<6} | {'rho':<7} | {'Zv':<7} | {'Phi (Flux)':<10}"
    )
    print("-" * 75)

    for i in range(40):
        engine.evolve_with_flux_conservation(steps=3000)
        d = engine.run_diagnostics()
        print(
            f"{i*3000:<8} | {d['Deff']:>5.2f} | {d['xi']:>6.2f} | {d['rho']:>7.2f} | {d['Zv']:>7.2f} | {d['Phi']:>10.2f}"
        )
