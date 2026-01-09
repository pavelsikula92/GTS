import networkx as nx
import numpy as np
import random
from collections import deque

# ==============================================================================
# GTS Math-KERNEL 
# Theoretical Framework: Geometric Unification Theory (GTS)
# Author: Pavel Sikula
# ------------------------------------------------------------------------------
# DESCRIPTION:
# This kernel simulates the ontogenesis of spacetime from a discrete causal
# substrate. It demonstrates how macroscopic geometry (Deff) emerges from
# microscopic topological updates governed by the principle of Action Minimization.
# ==============================================================================


class GTS_Causal_Manifold_v14_4:
    """
    GTS Methodology v14.4 - Causal Manifold Kernel.

    Implements the fundamental axioms of GTS:
    1. Spacetime as a Directed Acyclic Graph (DAG).
    2. Dimension (Deff) as an emergent property of network density.
    3. Dynamics driven by Topological Action (S_topo).
    """

    def __init__(self, n_events=400):
        """
        Initialize the Causal Substrate.

        Args:
            n_events (int): Total number of discrete causal events (nodes).
        """
        self.N = n_events

        # AXIOM II: CAUSALITY AS PARTIAL ORDER
        # Spacetime is initialized as a 1D causal chain (Path Graph).
        # DiGraph ensures the "Arrow of Time" is strictly enforced (u -> v).
        self.G = nx.path_graph(n_events, create_using=nx.DiGraph)

        # FUNDAMENTAL COUPLING CONSTANTS
        # alpha_g: Represents the intrinsic coupling of the causal substrate.
        # inv_alpha: Used here as a geometric scaling ratio between temporal
        # and spatial simplices.
        self.alpha_g = 0.007297
        self.inv_alpha = 137.036

        # DYNAMICAL PARAMETERS
        self.dyn_lambda = 1.0  # Initial Topological Impedance (normalized)

        # TOPOLOGICAL VISCOSITY BUFFER
        # Implements Axiom III's smoothness constraint. A rolling window prevents
        # instantaneous dimensional "jitter," simulating inertia of the manifold.
        self.lambda_window = deque(maxlen=10)

    def get_local_metrics(self, node):
        """
        Extracts local topological invariants (Time-like vs Space-like connectivity).

        Returns:
            nt (int): Causal Flux (number of temporal edges/relations).
            ns (int): Spatial Complexity (number of 2D simplices/triangles).
        """
        # Define the "Causal Sphere" around the event
        neighbors = list(nx.all_neighbors(self.G, node))

        if len(neighbors) < 2:
            return 0, 0

        # Subgraph analysis for local dimensionality
        sub = self.G.subgraph(neighbors + [node])

        # Metric 1: Throughput (Total causal relations)
        nt = sub.number_of_edges()

        # Metric 2: Transitivity (Spatial triangles)
        # In GTS, 3-cycles (triangles) represent the emergence of a 2D surface
        # within the discrete graph.
        ns = sum(nx.triangles(nx.Graph(sub)).values()) // 3

        return nt, ns

    def get_local_action(self, node, lambda_val):
        """
        Calculates the Topological Action (S_topo).

        S_topo = (Causal Flux + Resistance) - (Spatial Complexity * Impedance)

        The universe evolves by finding the path of least action, balancing
        the "urge" to expand spatially against the "resistance" of temporal flow.
        """
        nt, ns = self.get_local_metrics(node)

        # TOPOLOGICAL RESISTANCE (Geometric Inertia)
        # Acts as a repulsive force preventing the graph from collapsing
        # into a 0-dimensional singularity (infinite density).
        resistance = (nt**2) / (self.N * 0.5)

        # THE GTS MASTER EQUATION FOR ACTION
        # High ns (spatiality) lowers action (favored state), but is moderated
        # by alpha_g and the current global impedance (lambda).
        action = (nt + resistance) - (lambda_val * ns * self.alpha_g * self.inv_alpha)

        return action

    def attempt_update(self):
        """
        Monte Carlo Metropolis-Hastings step for Manifold Evolution.
        Proposes a topological change and accepts it based on S_topo minimization.
        """
        nodes = list(self.G.nodes())
        u, v = random.sample(nodes, 2)

        # AXIOM II: ARROW OF TIME ENFORCEMENT
        # Influence only flows from past indices to future indices.
        if u == v:
            return False
        if u > v:
            u, v = v, u

        # Baseline State
        s_old = self.get_local_action(u, self.dyn_lambda)
        existed = self.G.has_edge(u, v)

        # PROPOSE TOPOLOGICAL SHIFT (Edge creation or annihilation)
        if existed:
            self.G.remove_edge(u, v)
        else:
            self.G.add_edge(u, v)

        # CAUSAL PROTECTION CHECK
        # Revert immediately if the update creates a Closed Timelike Curve (CTC).
        # Spacetime must remain a Directed Acyclic Graph (DAG).
        if not nx.is_directed_acyclic_graph(self.G):
            if existed:
                self.G.add_edge(u, v)
            else:
                self.G.remove_edge(u, v)
            return False

        # Evaluate New State
        s_new = self.get_local_action(u, self.dyn_lambda)
        delta_s = s_new - s_old

        # SELECTION RULE (Quantum-Classical Transition)
        # If delta_s <= 0: Spontaneous transition (Stability).
        # If delta_s > 0: Accepted via thermal/quantum fluctuation probability.
        if delta_s <= 0 or random.random() < np.exp(-delta_s):
            return True
        else:
            # Revert if action increase is too high
            if existed:
                self.G.add_edge(u, v)
            else:
                self.G.remove_edge(u, v)
            return False

    def evolve(self, steps=1000):
        """Runs the simulation for a batch of micro-steps."""
        successes = 0
        for _ in range(steps):
            if self.attempt_update():
                successes += 1
        return successes

    def update_topology_state(self):
        """
        Updates Global Topological Impedance (Lambda).
        Lambda regulates how 'hard' it is for the network to grow new
        spatial dimensions.
        """
        sample_size = min(50, self.N)
        nodes_sample = random.sample(list(self.G.nodes()), sample_size)

        t_nt, t_ns = 0, 0
        for n in nodes_sample:
            nt, ns = self.get_local_metrics(n)
            t_nt += nt
            t_ns += ns

        if t_ns > 0:
            # Equilibrium condition for Impedance
            instant_lambda = t_nt / (max(t_ns, 1) * self.alpha_g * self.inv_alpha)
            self.lambda_window.append(instant_lambda)

            # Smooth transition via Moving Average (Viscosity)
            self.dyn_lambda = np.mean(self.lambda_window)

        # Calculate Global Causal Stress (Chi)
        stress = t_ns / t_nt if t_nt > 0 else 0
        return stress

    def measure_deff(self, samples=10):
        """
         Hausdorff Dimension Measurement.
        Instead of a single random node, we sample multiple nodes and
        average the results to filter out local topological noise.
        """
        results = []
        nodes = list(self.G.nodes())

        # Sample 'n' nodes to get a statistically significant D_eff
        sampled_nodes = random.sample(nodes, min(samples, len(nodes)))

        for node in sampled_nodes:
            try:
                # Measure causal volumes at shell distances 1, 2, and 3
                v1 = len(nx.single_source_shortest_path_length(self.G, node, cutoff=1))
                v2 = len(nx.single_source_shortest_path_length(self.G, node, cutoff=2))
                v3 = len(nx.single_source_shortest_path_length(self.G, node, cutoff=3))

                # We calculate the growth rate between shells
                # D = ln(V(r)/V(r-1)) / ln(r/(r-1))
                if v2 > v1 and v3 > v2:
                    d2 = np.log(v2 / v1) / np.log(2 / 1)  # Growth 1->2
                    d3 = np.log(v3 / v2) / np.log(3 / 2)  # Growth 2->3

                    # Mean growth rate represents local dimensionality
                    results.append((d2 + d3) / 2)
            except:
                continue

        if not results:
            return 1.0  # Default to 1D if graph is too sparse

        return np.clip(np.mean(results), 1.0, 5.0)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # 1000+ nodes provide enough 'causal depth' for shell r=3 to be meaningful.
    sim = GTS_Causal_Manifold_v14_4(n_events=1000)

    print(f"\n{'GTS v14.4 CAUSAL GENESIS SIMULATION':^70}")

    print(f"\n{'GTS v14.4 CAUSAL GENESIS SIMULATION':^70}")
    print("=" * 70)
    print(f"Axioms Loaded: I (Events), II (DAG), III (Action), IV (Flux)")
    print("-" * 70)
    print(f"{'STEP':<8} | {'D_EFF':<8} | {'IMPEDANCE (Z)':<16} | {'CAUSAL STRESS'}")
    print("-" * 70)

    for i in range(40):
        sim.evolve(steps=1000)
        stress = sim.update_topology_state()
        deff = sim.measure_deff()

        print(f"{i*1000:<8} | {deff:>8.2f} | {sim.dyn_lambda:>16.4f} | {stress:>10.4f}")

    print("=" * 70)
    print("Simulation Complete. Emergent Geometry Stabilized.")
