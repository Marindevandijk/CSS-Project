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
    Ns = (200, 400, 800,1600)
    runs_per_point = 20

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
    

def run_fig5_hetero_scaling():
    print('run figure 5')
    Ns=(200,400,800,1600)
    runs_per_point=20
    phi_grid=np.unique(np.concatenate([np.linspace(0.0,1.0,41),np.linspace(0.3,0.4,21)]))
    phi_grid.sort()
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

def run_society_comparison():
    print("\nRunning Society Comparison Experiment")
    societies = {
        "Baseline":   [0.10, 0.85, 0.05, 0.00],  
        "Polarized":  [0.10, 0.70, 0.20, 0.00], 
        "Diplomatic": [0.10, 0.80, 0.05, 0.05]   
    }
    N = 800  
    phi_grid = np.linspace(0.0, 0.6, 11) 
    results = {}
    
    for name, probs in societies.items():
        print(f"Testing {name} Society: {probs}")
        S_means, S_errs = [], []
        for phi in phi_grid:
            vals = []
            for _ in range(5): 
                sim = HeterogeneousSimulation(
                    N=N, type_probs=probs,
                    type_phi_values={0:0.05, 1:float(phi), 2:1.0, 3:float(phi)},
                    type_stubbornness_values={0:0.05, 1:0.45, 2:1.0, 3:0.45} 
                )
                sim.run_until_consensus(max_steps=N*2000)
                vals.append(sim.get_max_agree_component_fraction())
            S_means.append(np.mean(vals))
            S_errs.append(np.std(vals))
        results[name] = {"S": np.array(S_means), "err": np.array(S_errs)}
    
    np.savez(os.path.join(DATA_DIR, "society_comparison.npz"), results=results, phi_grid=phi_grid)

def run_hysteresis_experiment():
    print("\nRunning Hysteresis Experiment")
    societies = {
        "Polarized":  [0.10, 0.70, 0.20, 0.00], 
        "Diplomatic": [0.10, 0.80, 0.05, 0.05]  
    }
    N = 800
    phi_steps = 15
    phi_max = 0.5
    phi_up = np.linspace(0.0, phi_max, phi_steps)
    phi_down = np.linspace(phi_max, 0.0, phi_steps)
    results = {}

    for name, probs in societies.items():
        print(f"Testing {name} Society...")
        avg_S_up = np.zeros(len(phi_up))
        avg_S_down = np.zeros(len(phi_down))
        trials = 5 
        for t in range(trials):
            sim = HeterogeneousSimulation(
                N=N, type_probs=probs,
                type_phi_values={0:0.05, 1:0.0, 2:1.0, 3:0.0},
                type_stubbornness_values={0:0.0, 1:0.5, 2:1.0, 3:0.0}
            )
            # UP LEG
            trial_S_up = []
            for phi in phi_up:
                p = float(phi)
                sim.type_phi_values = {0:0.05, 1:p, 2:1.0, 3:p}
                sim.run_steps(N * 500) 
                trial_S_up.append(sim.get_max_agree_component_fraction())
            
            # DOWN LEG
            trial_S_down = []
            for phi in phi_down:
                p = float(phi)
                sim.type_phi_values = {0:0.05, 1:p, 2:1.0, 3:p}
                sim.run_steps(N * 500)
                trial_S_down.append(sim.get_max_agree_component_fraction())
            
            avg_S_up += np.array(trial_S_up)
            avg_S_down += np.array(trial_S_down)
            
        results[name] = {"up": avg_S_up / trials, "down": avg_S_down / trials}
    np.savez(os.path.join(DATA_DIR, "hysteresis.npz"), results=results, phi_up=phi_up, phi_down=phi_down)

def run_complexity_analysis():
    print("\nRunning Complexity Analysis")
    societies = {
        "Polarized":  [0.10, 0.70, 0.20, 0.00], 
        "Diplomatic": [0.10, 0.80, 0.05, 0.05]   
    }
    N = 800
    phi_grid = np.linspace(0.0, 1.0, 21)
    results = {}

    for name, probs in societies.items():
        complexities = []
        for phi in phi_grid:
            vals = []
            for _ in range(5): 
                sim = HeterogeneousSimulation(N=N, type_probs=probs,
                    type_phi_values={0:0.05, 1:float(phi), 2:1.0, 3:float(phi)},
                    type_stubbornness_values={0:0.05, 1:0.45, 2:1.0, 3:0.45})
                sim.run_until_consensus(max_steps=N*1000)
                vals.append(sim.get_kolmogorov_complexity())
            complexities.append(np.mean(vals))
        results[name] = complexities
    np.savez(os.path.join(DATA_DIR, "complexity.npz"), results=results, phi_grid=phi_grid)

def run_statistical_suite():
    print("\nRunning Statistical Suite")
    stress_phi = 0.35 
    N_trials = 20
    N_agents = 800
    phi_vals_hetero = {0:0.05, 1:float(stress_phi), 2:1.0, 3:float(stress_phi)}
    
    # 1. Mediator Test Data
    scenarios = {"Polarized": [0.10, 0.70, 0.20, 0.00], "Diplomatic": [0.10, 0.80, 0.05, 0.05]}
    mediator_data = {}
    for name, probs in scenarios.items():
        vals = []
        for _ in range(N_trials):
            sim = HeterogeneousSimulation(N=N_agents, type_probs=probs,
                type_phi_values=phi_vals_hetero, type_stubbornness_values={0:0.0, 1:0.5, 2:1.0, 3:0.0})
            sim.run_until_consensus(max_steps=N_agents*2000)
            vals.append(sim.get_max_agree_component_fraction())
        mediator_data[name] = vals

    # 2. Base vs Hetero Data
    base_data = {}
    # Homogeneous
    vals = []
    for _ in range(N_trials):
        sim = HeterogeneousSimulation(N=N_agents, type_probs=[0.0, 1.0, 0.0, 0.0], 
            type_stubbornness_values={1:0.0}, type_phi_values={1:float(stress_phi)})
        sim.run_until_consensus(max_steps=N_agents*2000)
        vals.append(sim.get_max_agree_component_fraction())
    base_data["Homogeneous"] = vals
    
    # Heterogeneous
    vals = []
    for _ in range(N_trials):
        sim = HeterogeneousSimulation(N=N_agents, type_probs=[0.10, 0.85, 0.05, 0.00],
            type_stubbornness_values={0:0.0, 1:0.5, 2:1.0, 3:0.0}, type_phi_values=phi_vals_hetero)
        sim.run_until_consensus(max_steps=N_agents*2000)
        vals.append(sim.get_max_agree_component_fraction())
    base_data["Heterogeneous"] = vals
    
    np.savez(os.path.join(DATA_DIR, "stats_suite.npz"), 
             mediator_pol=mediator_data["Polarized"], mediator_dip=mediator_data["Diplomatic"],
             base_homo=base_data["Homogeneous"], base_hetero=base_data["Heterogeneous"])
    
if __name__ == "__main__":
    run_fig1()
    run_fig2()
    run_fig3()
    run_fig5_hetero_scaling()
    run_fig4_hetero_distribution()
    run_society_comparison()
    run_hysteresis_experiment()
    run_complexity_analysis()
    run_statistical_suite()
