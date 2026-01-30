import os
import numpy as np
from base_model import HolmeNewmanSimulation, HeterogeneousSimulation
import csv
import matplotlib.pyplot as plt

def run_once(m, seed=0):
    sim = HeterogeneousSimulation(
        N=400, seed=seed, type_probs=[0.1, 0.85-m, 0.05, m]
    )
    sim.run_until_consensus()
    return sim.get_max_agree_component_fraction()
m_list = np.linspace(0.0, 0.05, 30)
Critical_fraction = []

def bootstrap_ci(x, n_boot=00, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return lo, hi

m_list = np.linspace(0.0, 0.05, 30)
n_runs = 5
all_vals = []

for m in m_list:
    vals = [run_once(m,seed=i) for i in range(n_runs)]
    all_vals.append(vals)
    print(m)

all_vals = np.array(all_vals)           
means = all_vals.mean(axis=1)

n_boot = 300   
rng = np.random.default_rng(0)
idx = rng.integers(0, n_runs, size=(n_boot, n_runs))  

boot_samples = np.take(all_vals, idx, axis=1)      
boot_means = boot_samples.mean(axis=2)               

lo = np.percentile(boot_means, 2.5, axis=1)
hi = np.percentile(boot_means, 97.5, axis=1)

with open("critical_fraction_smooth_consensus.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["m", "mean", "ci_lo", "ci_hi"])
    for m, mean, l, h in zip(m_list, means, lo, hi):
        writer.writerow([m, mean, l, h])

plt.plot(m_list, means)
plt.fill_between(m_list, lo, hi, alpha=0.2)
plt.xlabel("m values")
plt.ylabel("Max consensus fraction")
plt.savefig("figures/consensus_size_.png")
plt.show()
