import os
import numpy as np
from base_model import HolmeNewmanSimulation, HeterogeneousSimulation

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)



def run_fig1():
    print('Running Fig 1')
    sim = HolmeNewmanSimulation(N=60, k_avg=4, gamma=6, phi=0.2, seed=1)
    for _ in range(sim.N * 10):
        sim.step()

    edges = np.array(list(sim.graph.edges(keys=False)), dtype=int)
    opinions = np.array(sim.opinions, dtype=int)

    out = os.path.join(DATA_DIR, "fig1_network.npz")
    np.savez(out, edges=edges, opinions=opinions)
    print("Saved", out)


def run_fig2():
    print('Running Fig 2')
    N = 3200
    for phi in [0.04, 0.458, 0.96]:
        all_sizes = []
        valid_runs, attempts = 0, 0
        while valid_runs < 5 and attempts < 15:
            attempts += 1
            sim = HolmeNewmanSimulation(N=N, phi=phi)
            steps, converged = sim.run_until_consensus(max_steps=N*5000)
            if converged:
                all_sizes.extend(sim.get_community_sizes())
                valid_runs += 1
        np.savez(os.path.join(DATA_DIR, f"fig2_phi_{phi}.npz"), N=N, phi=phi, sizes=np.asarray(all_sizes, dtype=int))

def run_fig3():
    print('Running Fig 3 (Scaling)')
    Ns = (200, 400, 800) 
    phi_grid = np.linspace(0.0, 1.0, 21) 
    S = np.zeros((len(Ns), len(phi_grid)))
    conv = np.zeros((len(Ns), len(phi_grid)))
    for i, N in enumerate(Ns):
        print(f"  Simulating N={N}")
        target_runs = 10 if N < 800 else 5
        limit = N * 5000 
        for j, phi in enumerate(phi_grid):
            val, valid_runs, attempts = 0, 0, 0
            while valid_runs < target_runs and attempts < (target_runs * 3):
                attempts += 1
                sim = HolmeNewmanSimulation(N=N, phi=float(phi))
                steps, converged = sim.run_until_consensus(max_steps=limit)
                if converged:
                    val += sim.get_max_community_fraction()
                    valid_runs += 1
            S[i, j] = val / valid_runs if valid_runs > 0 else np.nan
            conv[i, j] = valid_runs / attempts if attempts > 0 else 0
    np.savez(os.path.join(DATA_DIR, "fig3_scaling.npz"), Ns=Ns, phi_grid=phi_grid, S=S, conv=conv)

def run_fig5_hetero_scaling():
    print('Running Fig 5 (Hetero Scaling)')
    Ns = (200, 400, 800)
    phi_grid = np.linspace(0.0, 1.0, 15)
    S = np.zeros((len(Ns), len(phi_grid)))
    conv = np.zeros((len(Ns), len(phi_grid)))
    
    for i, N in enumerate(Ns):
        print(f"  Simulating N={N}...")
        target_runs = 10 
        limit = N * 10000 
        for j, phi in enumerate(phi_grid):
            val, converged_count = 0, 0
            for _ in range(target_runs): 
                sim = HeterogeneousSimulation(N=N, type_probs=[0.1, 0.85, 0.05, 0.0], 
                                              type_phi_values={0:0.05, 1:float(phi), 2:1.0, 3:float(phi)})
                steps, is_converged = sim.run_until_consensus(max_steps=limit)
                val += sim.get_max_agree_component_fraction()
                if is_converged: converged_count += 1
            S[i, j] = val / target_runs
            conv[i, j] = converged_count / target_runs
    np.savez(os.path.join(DATA_DIR, "fig5_hetero_scaling.npz"), Ns=Ns, phi_grid=phi_grid, S=S, conv=conv)

def run_fig4_hetero_distribution(target_phi_c):
    print(f"Running Fig 4 (Distribution at phi_c={target_phi_c:.3f})")
    N = 3200
    for phi in [0.04, target_phi_c, 0.96]:
        all_sizes = []
        for _ in range(5): 
            sim = HeterogeneousSimulation(N=N, type_probs=[0.1, 0.85, 0.05, 0.0], 
                                          type_phi_values={0:0.05, 1:float(phi), 2:1.0, 3:float(phi)})
            sim.run_until_consensus(max_steps=N*2000)
            all_sizes.extend(sim.get_community_sizes())
        np.savez(os.path.join(DATA_DIR, f"fig4_hetero_phi_{phi:.4f}.npz"), sizes=np.asarray(all_sizes, dtype=int), phi=phi, N=N)

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
    run_society_comparison()
    run_hysteresis_experiment()
    run_complexity_analysis()
    run_statistical_suite()
