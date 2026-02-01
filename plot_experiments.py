import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import interp1d
from scipy import stats
import powerlaw
from helpers import _log_binned_density, estimate_phi_c_from_crossings, solve_consistent_hetero_params
from run_experiment import run_fig4_hetero_distribution
DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def _ensure_fig4_data(phi_c):
    
    phi_c = float(np.round(phi_c, 4))
    required_files = [
        os.path.join(DATA_DIR, f"fig4_hetero_phi_{0.04:.4f}.npz"),
        os.path.join(DATA_DIR, f"fig4_hetero_phi_{phi_c:.4f}.npz"),
        os.path.join(DATA_DIR, f"fig4_hetero_phi_{0.96:.4f}.npz"),
    ]

    if not all(os.path.exists(f) for f in required_files):
        print(f"[plot] Fig4 data missing → generating using run_fig4_hetero_distribution(phi_c={phi_c:.4f})")
        run_fig4_hetero_distribution(phi_c)
    else:
        print("Fig4 data already exists")

def plot_fig1():
    d = np.load(os.path.join(DATA_DIR, "fig1_network.npz"))
    edges = d["edges"]
    opinions = d["opinions"]
    N = len(opinions)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges.tolist())

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=250,
        node_color=opinions,
        cmap=plt.cm.tab20,
        edgecolors="k"
    )
    nx.draw_networkx_edges(G, pos, alpha=0.35)
    plt.title("Figure 1")
    plt.axis("off")

    plt.savefig(os.path.join(FIG_DIR, "figure1.pdf"))
    plt.show()

def plot_fig2():
    phis = [0.04, 0.458, 0.96]
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    for i, (ax, phi) in enumerate(zip(axes, phis)):
        d = np.load(os.path.join(DATA_DIR, f"fig2_phi_{phi}.npz"))
        sizes = d["sizes"]
        N = int(d["N"])

        fit = powerlaw.Fit(sizes, discrete=True, verbose=False)
        centers, P = _log_binned_density(sizes, N)

        mask = P > 0
        ax.loglog(centers[mask], P[mask], "o", mfc="none")
        ax.set_ylabel("P(s)")

        if i == 1:
            label_text = fr"$\phi_c$={phi}"
            R, p_val = fit.distribution_compare("power_law", "exponential")
            ax.text(0.05, 0.15, f"vs Exp: R={R:.2f}, p={p_val:.2e}",
                    transform=ax.transAxes, ha="left", fontsize=9, color="red")
        else:
            label_text = fr"$\phi$={phi}"

        ax.text(0.95, 0.85, label_text, transform=ax.transAxes, ha="right")

    axes[-1].set_xlabel("s")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure2.pdf"))
    plt.show()


def plot_fig3():
    d = np.load(os.path.join(DATA_DIR, "fig3_scaling.npz"))
    Ns, phi_grid, S = d["Ns"], d["phi_grid"], d["S"]

    a, b = 0.61, 0.7
    phi_c = estimate_phi_c_from_crossings(phi_grid, Ns, S, a=a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, N in enumerate(Ns):
        ax1.plot(phi_grid, (N**a) * S[i], "o-", markersize=3, label=f"N={N}")

    ax1.axvline(phi_c, linestyle="--", alpha=0.6)
    ax1.legend()
    ax1.set_xlabel(r"Rewiring Probability $\phi$")
    ax1.set_ylabel(r"Scaled Order Parameter $N^a S$")

    for i, N in enumerate(Ns):
        ax2.plot((N**b) * (phi_grid - phi_c), (N**a) * S[i], "o", markersize=3)

    ax2.set_xlabel(r"Rescaled Control Parameter $N^b(\phi - \phi_c)$")
    ax2.set_ylabel(r"Scaled Order Parameter $N^a S$")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure3.pdf"))
    plt.show()


def plot_fig5_hetero_scaling():
    d = np.load(os.path.join(DATA_DIR, "fig5_hetero_scaling.npz"))
    Ns, phi_grid, S, conv = d["Ns"], d["phi_grid"], d["S"], d["conv"]

    best_a, best_b, phi_c = solve_consistent_hetero_params(Ns, phi_grid, S, conv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, N in enumerate(Ns):
        ax1.plot(phi_grid, (N**best_a) * S[i], "o-", markersize=3, label=f"N={N}")
        ax2.plot((N**best_b) * (phi_grid - phi_c), (N**best_a) * S[i], "o", markersize=3)

    ax1.axvline(phi_c, linestyle="--", color="r", label=f"phi_c={phi_c:.3f}")
    ax1.legend()
    ax1.set_xlabel(r"Rewiring Probability $\phi$")
    ax1.set_ylabel(r"Scaled Order Parameter $N^a S$")

    ax2.set_xlabel(r"Rescaled Control Parameter $N^b(\phi - \phi_c)$")
    ax2.set_ylabel(r"Scaled Order Parameter $N^a S$")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure5_hetero_scaling.pdf"))
    plt.show()

    return float(phi_c)


def plot_fig4_hetero_distribution(target_phi_c):
    phis = [0.04, float(target_phi_c), 0.96]
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    for i, (ax, phi) in enumerate(zip(axes, phis)):
        d = np.load(os.path.join(DATA_DIR, f"fig4_hetero_phi_{phi:.4f}.npz"))
        sizes = d["sizes"]
        N = int(d["N"])

        if len(sizes) > 100:
            fit = powerlaw.Fit(sizes, discrete=True, verbose=False)

        centers, P = _log_binned_density(sizes, N)
        mask = P > 0
        ax.loglog(centers[mask], P[mask], "ro", mfc="none")
        ax.set_ylabel("P(s)")

        if i == 1:
            label_text = fr"Hetero $\phi_c$={phi:.3f}"
            if len(sizes) > 100:
                R, p_val = fit.distribution_compare("power_law", "exponential")
                ax.text(0.05, 0.15, f"vs Exp: R={R:.2f}, p={p_val:.2e}",
                        transform=ax.transAxes, ha="left", fontsize=9, color="blue")
        else:
            label_text = fr"Hetero $\phi$={phi:.3f}"

        ax.text(0.95, 0.85, label_text, transform=ax.transAxes, ha="right")

    axes[-1].set_xlabel("s")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure4_hetero_dist.pdf"))
    plt.show()

def plot_society_comparison():
    try:
        d = np.load(os.path.join(DATA_DIR, "society_comparison.npz"), allow_pickle=True)
        results = d["results"].item()
        phi_grid = d["phi_grid"]
        plt.figure(figsize=(8, 6))
        colors = {"Baseline": "black", "Polarized": "red", "Diplomatic": "blue"}
        styles = {"Baseline": "o-", "Polarized": "s--", "Diplomatic": "^-."}
        
        for name, data in results.items():
            plt.errorbar(phi_grid, data["S"], yerr=data["err"], label=name, 
                         color=colors[name], fmt=styles[name], capsize=3, alpha=0.8)
        
        plt.xlabel(r"Rewiring Probability $\phi$")
        plt.ylabel("Max Consensus Size $S_{agree}$")
        plt.title("Society Stability Profile")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "society_comparison.pdf"))
        plt.show()
    except FileNotFoundError: print("Society Comparison data not found.")

def plot_hysteresis():
    try:
        d = np.load(os.path.join(DATA_DIR, "hysteresis.npz"), allow_pickle=True)
        results = d["results"].item()
        phi_up, phi_down = d["phi_up"], d["phi_down"]
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {"Polarized": "red", "Diplomatic": "blue"}
        for name, res in results.items():
            ax.plot(phi_up, res["up"], color=colors[name], linestyle="-", marker="^", alpha=0.6, label=f"{name} (Fwd)")
            ax.plot(phi_down, res["down"], color=colors[name], linestyle="--", marker="v", alpha=0.9, label=f"{name} (Rev)")
            ax.fill_between(phi_up, res["up"], res["down"][::-1], color=colors[name], alpha=0.1)
        ax.legend()
        ax.set_xlabel(r"Rewiring Probability $\phi$")
        ax.set_ylabel(r"Max Consensus Size $S_{agree}$")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "hysteresis_plot.pdf"))
        plt.show()
    except FileNotFoundError: print("Hysteresis data not found.")

def plot_complexity():
    try:
        d = np.load(os.path.join(DATA_DIR, "complexity.npz"), allow_pickle=True)
        results = d["results"].item()
        phi_grid = d["phi_grid"]
        plt.figure(figsize=(10, 6))
        for name, compl in results.items():
            plt.plot(phi_grid, compl, 'o-', label=name)
        plt.legend()
        plt.xlabel(r"Rewiring Probability $\phi$")
        plt.ylabel("Kolmogorov Complexity")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "kolmogorov_complexity.pdf"))
        plt.show()
    except FileNotFoundError: print("Complexity data not found.")

def plot_stats():
    try:
        d = np.load(os.path.join(DATA_DIR, "stats_suite.npz"))
        pop1, pop2 = d["mediator_pol"], d["mediator_dip"]
        pop_base, pop_hetero = d["base_homo"], d["base_hetero"]
        
        _, p_val_med = stats.ttest_ind(pop1, pop2, equal_var=False)
        _, p_val_base = stats.ttest_ind(pop_base, pop_hetero, equal_var=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].boxplot([pop_base, pop_hetero], patch_artist=True, boxprops=dict(facecolor="salmon"))
        axes[0].set_xticklabels(["Homogeneous", "Heterogeneous"])
        axes[0].set_title(f"Baseline Collapse (p={p_val_base:.1e})")
        axes[0].set_ylabel("Max Consensus Size") # Added Y label
        
        axes[1].boxplot([pop1, pop2], patch_artist=True, boxprops=dict(facecolor="lightblue"))
        axes[1].set_xticklabels(["Polarized", "Diplomatic"])
        axes[1].set_title(f"Mediator Rescue (p={p_val_med:.1e})")
        axes[1].set_ylabel("Max Consensus Size") # Added Y label
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "statistical_analysis_full.pdf"))
        plt.show()
    except FileNotFoundError: print("Stats data not found.")
if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    phi_c = plot_fig5_hetero_scaling()
    _ensure_fig4_data(phi_c)
    plot_fig4_hetero_distribution(phi_c)
    plot_society_comparison()
    plot_hysteresis()
    plot_complexity()
    plot_stats()
