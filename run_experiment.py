import os
import numpy as np
from base_model import HolmeNewmanSimulation, HeterogeneousSimulation

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
def estimate_phi_c_internal(phi_grid,Ns,S,a=0.61):
    Y = (Ns[:, None] ** a)*S
    phis =[]
    for i in range(len(Ns)-1):
        diff = Y[i] - Y[i + 1]
        for p in range(len(phi_grid) - 1):
            d0, d1 = diff[p], diff[p + 1]
            if d0 * d1 < 0:
                x0, x1 = phi_grid[p], phi_grid[p+1]
                exact_crossing=x0 - d0 *(x1-x0)/(d1-d0)
                phis.append(exact_crossing)
                break
    phis = np.array(phis,float)
    if len(phis) == 0:
        return 0.458
    return phis.mean()
    
def run_fig5_hetero_scaling():
    print('run figure 5')
    Ns=(200,400,800)
    runs_per_point=10
    phi_grid=np.linspace(0.3,0.7,20)
    probs=[0.10,0.85,0.05]
    type_stubbornness_values={0:0.0, 1:0.5, 2:1.0}

    S=np.zeros((len(Ns),len(phi_grid)))
    for i,N in enumerate(Ns):
        for j,phi in enumerate(phi_grid):
            val=0
            for _ in range(runs_per_point):
                sim=HeterogeneousSimulation(N=N,type_probs=probs,type_phi_values={0:0.05, 1:float(phi), 2:1.0},type_stubbornness_values=type_stubbornness_values)
                sim.run_until_consensus(check_every=N)
                val += sim.get_max_community_fraction()
            S[i,j]=val/runs_per_point
    out=os.path.join(DATA_DIR,"fig5_hetero_scaling.npz")
    np.savez(out,Ns=np.array(Ns),phi_grid=phi_grid,S=S)
    print("Saved",out)

def run_fig4_hetero_distribution():
    try:
        d=np.load(os.path.join(DATA_DIR,"fig5_hetero_scaling.npz"))
        phi_c=estimate_phi_c_internal(d["phi_grid"],d["Ns"],d["S"])
        print(f"Calculated Hetero Phi_c: {phi_c:.4f}")
    except FileNotFoundError:
        print("error: run Fig 5 first!")
        return

    print("run figure 4 with phi_c={phi_c:.4f}")
    N =3200
    realizations=20
    probs =[0.10,0.85,0.05]
    phis =[0.04,phi_c,0.96]
    type_stubbornness_values={0:0.0, 1:0.5, 2:1.0}
    for phi in phis:
        all_sizes=[]
        for _ in range(realizations):
            sim =HeterogeneousSimulation(
                N=N,
                type_probs=probs,
                type_phi_values={0: 0.05, 1: float(phi), 2: 1.0},
                type_stubbornness_values=type_stubbornness_values
            )
            sim.run_until_consensus(check_every=N)
            all_sizes.extend(sim.get_community_sizes())

        out = os.path.join(DATA_DIR,f"fig4_hetero_phi_{phi:.4f}.npz")
        np.savez(out,sizes=np.array(all_sizes),phi=phi,N=N)
        print(f"Saved {out}")
if __name__ == "__main__":
    run_fig1()
    run_fig2()
    run_fig3()
    run_fig5_hetero_scaling()
    run_fig4_hetero_distribution()
