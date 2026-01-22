import os
import numpy as np
from base_model import HolmeNewmanSimulation

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)



def run_fig1():
    print('run figure 1')
    sim = HolmeNewmanSimulation(N=60, k_avg=4, gamma=6, phi=0.2, seed=1)
    for _ in range(sim.N * 10):
        sim.step()

    edges = np.array(list(sim.graph.edges(keys=False)), dtype=int)
    opinions = np.array(sim.opinions, dtype=int)

    out = os.path.join(DATA_DIR, "fig1_network.npz")
    np.savez(out, edges=edges, opinions=opinions)
    print("Saved", out)


def run_fig2():
    print('run figure 2')
    N = 3200
    k_avg = 4
    gamma = 10
    phis = [0.04, 0.458, 0.96]
    realizations = 20

    for phi in phis:
        all_sizes = []
        for _ in range(realizations):
            sim = HolmeNewmanSimulation(N=N, k_avg=k_avg, gamma=gamma, phi=phi)
            sim.run_until_consensus(check_every=N)
            all_sizes.extend(sim.get_community_sizes())

        out = os.path.join(DATA_DIR, f"fig2_phi_{phi}.npz")
        np.savez(out, N=N, phi=phi, sizes=np.array(all_sizes, dtype=int))
        print("Saved", out)


def run_fig3():
    print('run figure 3')
    Ns = (200, 400, 800)
    runs_per_point = 10

    phi_grid = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 41),
        np.linspace(0.44, 0.47, 21)
    ]))
    phi_grid.sort()

    S = np.zeros((len(Ns), len(phi_grid)))

    for i, N in enumerate(Ns):
        for j, phi in enumerate(phi_grid):
            S_sum = 0.0
            for _ in range(runs_per_point):
                sim = HolmeNewmanSimulation(N=N, k_avg=4, gamma=10, phi=float(phi))
                sim.run_until_consensus(check_every=N)
                S_sum += sim.get_max_community_fraction()
            S[i, j] = S_sum / runs_per_point

    out = os.path.join(DATA_DIR, "fig3_scaling.npz")
    np.savez(out, Ns=np.array(Ns), phi_grid=phi_grid, S=S)
    print("Saved", out)


if __name__ == "__main__":
    run_fig1()
    run_fig2()
    run_fig3()
