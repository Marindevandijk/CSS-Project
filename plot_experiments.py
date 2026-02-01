import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import interp1d
from helpers import _log_binned_density, estimate_phi_c_from_crossings, solve_consistent_hetero_params

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

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

    for ax, phi in zip(axes, phis):
        d = np.load(os.path.join(DATA_DIR, f"fig2_phi_{phi}.npz"))
        sizes = d["sizes"]
        N = int(d["N"])

        centers, P = _log_binned_density(sizes, N)
        mask = P > 0

        ax.loglog(centers[mask], P[mask], "o", markerfacecolor="none", markersize=5)
        ax.set_ylabel("P(s)")
        ax.text(0.95, 0.85, f"φ={phi}", transform=ax.transAxes, ha="right")

        if abs(phi - 0.458) < 1e-9:
            alpha = 3.5
            s0 = 30.0
            P0 = np.interp(s0, centers, P, left=np.nan, right=np.nan)
            if np.isfinite(P0) and P0 > 0:
                sline = np.array([10.0, 300.0])
                pline = P0 * (sline / s0) ** (-alpha)
                ax.loglog(sline, pline, "-", linewidth=1)

    axes[-1].set_xlabel("s")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure2.pdf"))
    plt.show()


def plot_fig3():
    d = np.load(os.path.join(DATA_DIR, "fig3_scaling.npz"))
    Ns = d["Ns"]
    phi_grid = d["phi_grid"]
    S = d["S"]

    a = 0.61
    b = 0.7
    phi_c_hat, phi_c_err = estimate_phi_c_from_crossings(phi_grid, Ns, S, a=a)
    print("Estimated phi_c=", phi_c_hat)
    print("error range:", phi_c_hat - phi_c_err, phi_c_hat + phi_c_err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, N in enumerate(Ns):
        ax1.plot(phi_grid, (N ** a) * S[i], "o-", markersize=3, label=f"N={N}")

    ax1.axvline(phi_c, linestyle="--", alpha=0.6)
    ax1.set_xlabel("φ")
    ax1.set_ylabel(r"$N^{a} S$")
    ax1.set_title("Base Crossing plot at $\phi_c={phi_c}$")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    for i, N in enumerate(Ns):
        x = (N ** b) * (phi_grid - phi_c)
        y = (N ** a) * S[i]
        ax2.plot(x, y, "o", markersize=3, label=f"N={N}")
    ax2.set_xlabel(r"$N^{b}(\phi-\phi_c)$")
    ax2.set_ylabel(r"$N^{a} S$")
    ax2.set_title("Base Data collapse")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure3.pdf"))
    plt.show()

def plot_fig5_hetero_scaling():
    try:
        d=np.load(os.path.join(DATA_DIR, "fig5_hetero_scaling.npz"))
        Ns,phi_grid,S=d["Ns"],d["phi_grid"],d["S"]
        
        # 1 Initial Guess
        current_a = 0.61
        current_phi_c, _ = estimate_phi_c_from_crossings(phi_grid,Ns,S,a=current_a)
        
        # 2 Optimization
        best_a,best_b= auto_find_scaling_exponents(Ns,phi_grid,S,current_phi_c)
        
        # 3 Update phi_c
        phi_c_final, _ =estimate_phi_c_from_crossings(phi_grid,Ns,S,a=best_a)
        
        # 4 Refine 'b' with FINAL phi_c 
        best_a_refined,best_b_refined =auto_find_scaling_exponents(Ns,phi_grid,S,phi_c_final)
        
        best_a =best_a_refined
        best_b =best_b_refined
        print(f"Final Hetero: phi_c={phi_c_final:.4f}, a={best_a:.2f}, b={best_b:.2f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for i, N in enumerate(Ns):
            ax1.plot(phi_grid, (N ** best_a) * S[i], "o-", markersize=3, label=f"N={N}")
            ax2.plot((N ** best_b) * (phi_grid - phi_c_final), (N ** best_a) * S[i], "o", markersize=3)
        
        ax1.axvline(phi_c_final, linestyle="--", color='r', label=f"New $\phi_c$")
        ax1.set_xlabel("φ")
        ax1.set_ylabel(r"$N^{a} S$")
        ax1.set_title(f"Hetero Crossing ($\phi_c={phi_c_final:.3f}$)")
        ax1.legend()
        ax2.set_xlabel(r"$N^{b}(\phi-\phi_c)$")
        ax2.set_ylabel(r"$N^{a} S$")
        ax2.set_title(f"Hetero Collapse ($a={best_a:.2f}, b={best_b:.2f}$)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "figure5_hetero_scaling.pdf"))
        plt.show()
        return phi_c_final
    except FileNotFoundError:
        print("Fig 5 data missing")
        return 0.458

def plot_fig4_hetero_distribution(phi_c_filename_target):
    phis = [0.04,phi_c_filename_target,0.96]
    fig,axes = plt.subplots(3,1,figsize=(7,10),sharex=True)
    
    for ax, phi in zip(axes,phis):
        try:
            fname = f"fig4_hetero_phi_{phi:.4f}.npz"
            d =np.load(os.path.join(DATA_DIR, fname))
            sizes,N =d["sizes"],int(d["N"])
            centers, P = _log_binned_density(sizes,N)
            mask = P>0
            ax.loglog(centers[mask],P[mask],"ro",mfc="none",markersize=5)
            ax.set_ylabel("P(s)")
            ax.text(0.95,0.85, f"Hetero $\phi$={phi:.3f}",transform=ax.transAxes,ha="right")
            
            if abs(phi - phi_c_filename_target) < 1e-9:
                 calc_alpha = auto_find_power_law_slope(sizes)
                 print(f"Calculated Alpha (Slope) for Hetero: {calc_alpha:.2f}")
                 
                 mask_tail=(centers[mask] > 30) 
                 if np.sum(mask_tail) > 0:
                     tail_idx =np.where(mask_tail)[0][0]
                     s0 =centers[mask][tail_idx]
                     P0 =P[mask][tail_idx]
                     sline=np.array([10.0,1000.0])
                     pline=P0*(sline/s0)**(-calc_alpha)
                     ax.loglog(sline,pline, "k--",linewidth=1.5,label=f"Fit Slope -{calc_alpha:.2f}")
                     ax.legend()
            
        except FileNotFoundError: 
            print(f"Data file for phi={phi} not found (Checked: {fname})")
            
    axes[-1].set_xlabel("s")
    plt.suptitle("Figure 4: Heterogeneous Distributions")
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
    new_phi_c = plot_fig5_hetero_scaling()
    d_fig5=np.load(os.path.join(DATA_DIR,"fig5_hetero_scaling.npz"))
    sim_phi_c=estimate_phi_c_from_crossings(d_fig5["phi_grid"],d_fig5["Ns"],d_fig5["S"])[0]
    plot_fig4_hetero_distribution(sim_phi_c)
    plot_society_comparison()
    plot_hysteresis()
    plot_complexity()
    plot_stats()
