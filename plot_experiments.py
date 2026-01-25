import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _log_binned_density(samples, N, nbins=25):
    samples = np.asarray(samples, dtype=float)
    edges = np.logspace(0, np.log10(N), nbins + 1)
    counts, _ = np.histogram(samples, bins=edges)
    widths = edges[1:] - edges[:-1]
    total = counts.sum()
    density = counts / (total * widths) if total > 0 else np.zeros_like(widths)
    centers = np.sqrt(edges[1:] * edges[:-1])
    return centers, density


def estimate_phi_c_from_crossings(phi_grid, Ns, S, a=0.61):

    # Y = n^a*S, at critical point these curves should intersect
    Y = (Ns[:, None] ** a) * S
    phis = []

    for i in range(len(Ns) - 1):
        # compare NS = 200 with Ns=400 for example
        # sign change in diff,  means crossing
        diff = Y[i] - Y[i + 1]

        best_phi = np.nan
        best_score = -1.0

        for p in range(len(phi_grid) - 1):
            d0, d1 = diff[p], diff[p + 1]
            if d0 * d1 < 0:
                # score with biggest difference, phase transition is steep
                x0=phi_grid[p]
                x1=phi_grid[p+1]
                exact_crossing=x0-d0*(x1-x0)/(d1-d0)
                phis.append(exact_crossing)
                break
    phis= np.array(phis,float)
    return phis.mean(),(phis.max()-phis.min())/2

    

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

    ax1.axvline(phi_c_hat, linestyle="--", alpha=0.6)
    ax1.set_xlabel("φ")
    ax1.set_ylabel(r"$N^{a} S$")
    ax1.set_title("Crossing plot")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    for i, N in enumerate(Ns):
        x = (N ** b) * (phi_grid - phi_c_hat)
        y = (N ** a) * S[i]
        ax2.plot(x, y, "o", markersize=3, label=f"N={N}")
    ax2.set_xlabel(r"$N^{b}(\phi-\phi_c)$")
    ax2.set_ylabel(r"$N^{a} S$")
    ax2.set_title("Data collapse")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure3.pdf"))
    plt.show()

def plot_fig5_hetero_scaling():
    try:
        d = np.load(os.path.join(DATA_DIR,"fig5_hetero_scaling.npz"))
        Ns,phi_grid,S = d["Ns"],d["phi_grid"],d["S"]
        a,b = 0.61,0.7
        phi_c_new, _ = estimate_phi_c_from_crossings(phi_grid,Ns,S,a=a)
        print(f"Hetero Phi_c: {phi_c_new:.4f}")

        fig, (ax1,ax2) =plt.subplots(1,2,figsize=(13,5))
        for i, N in enumerate(Ns):
            ax1.plot(phi_grid,(N ** a)*S[i], "o-", markersize=3,label=f"N={N}")
            ax2.plot((N ** b)*(phi_grid-phi_c_new),(N ** a)*S[i],"o",markersize=3)

        ax1.axvline(phi_c_new,linestyle="--",color='r',label=f"New $\phi_c$")
        ax1.set_title(f"Hetero Crossing (phi_c={phi_c_new:.3f})")
        ax1.legend()
        ax2.set_title("Hetero Collapse")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR,"figure5_hetero_scaling.pdf"))
        plt.show()
        return phi_c_new
    except FileNotFoundError:
        print("Fig 5 data missing")
        return 0.458

def plot_fig4_hetero_distribution(phi_c):
    phis = [0.04,phi_c,0.96]
    fig,axes =plt.subplots(3,1,figsize=(7,10),sharex=True)

    for ax, phi in zip(axes,phis):
        try:
            fname=f"fig4_hetero_phi_{phi:.4f}.npz"
            d=np.load(os.path.join(DATA_DIR,fname))
            sizes=d["sizes"]
            N =int(d["N"])
            centers,P= _log_binned_density(sizes, N)
            mask=P>0
            ax.loglog(centers[mask], P[mask],"ro",mfc="none",markersize=5)
            ax.set_ylabel("P(s)")
            ax.text(0.95, 0.85, f"Hetero $\phi$={phi:.3f}", transform=ax.transAxes, ha="right")

            if abs(phi-phi_c) < 1e-9:
                sline = np.array([10.0, 300.0])
                if len(P[mask]) > 0:
                    P0 =P[mask][0]
                    s0 =centers[mask][0]
                    pline =P0*(sline/s0)**(-3.5)
                    ax.loglog(sline,pline,"k--",linewidth=1,label="Slope -3.5")
                    ax.legend()
        except FileNotFoundError:
            pass

    axes[-1].set_xlabel("s")
    plt.suptitle("Figure 4: Heterogeneous Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure4_hetero_dist.pdf"))
    plt.show()

if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    new_phi_c = plot_fig5_hetero_scaling()
    plot_fig4_hetero_distribution(new_phi_c)
