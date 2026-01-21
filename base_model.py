import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class HolmeNewmanSimulation:
    #setting up the network with its properties
    def __init__(self,N=3200,k_avg=4,gamma=10,phi=0.458,seed=None):
        self.N=int(N)
        self.k_avg=float(k_avg)
        self.M=int(round(self.k_avg*self.N/2.0))
        self.gamma=int(gamma)
        assert self.N % self.gamma==0
        self.G_count=self.N//self.gamma
        self.phi=float(phi)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.graph=nx.MultiGraph()
        self.graph.add_nodes_from(range(self.N))
        edges=[(random.randrange(self.N),random.randrange(self.N)) for _ in range(self.M)]
        self.graph.add_edges_from(edges)

        self.opinions=np.random.randint(0,self.G_count,size=self.N)
        self.members=[[] for _ in range(self.G_count)]
        self.pos_in_members=np.empty(self.N,dtype=int)
        for i in range(self.N):
            op=int(self.opinions[i])
            self.pos_in_members[i]=len(self.members[op])
            self.members[op].append(i)
#updating our system when someone changes their belief
    def _move_member(self,node,old_op,new_op):
        old_list=self.members[old_op]
        idx=self.pos_in_members[node]
        last=old_list[-1]
        old_list[idx]=last
        self.pos_in_members[last]=idx
        old_list.pop()
        new_list=self.members[new_op]
        self.pos_in_members[node]=len(new_list)
        new_list.append(node)
#it picks one random friendship for a specific person. like you ask A, "Pick one of your friends at random." A might pick B
    def _random_incident_edge(self,i):
        edges_i =list(self.graph.edges(i,keys=True))
        if not edges_i:
            return None
        return random.choice(edges_i)
#it helps to identify the other person in a friendship. 
#we know a friendship exists between A and B. If A is i, then this function tells you that B is the neighbor
    @staticmethod
    def _other_endpoint(i,u,v):
        return v if u == i else u
#this is where the connection happens or breaks
    def step(self):
        i=random.randrange(self.N)

        e=self._random_incident_edge(i)
        if e is None:
            return True 
        u,v,k=e
        j =self._other_endpoint(i, u, v)

        if random.random() < self.phi:
            op_i=int(self.opinions[i])
            candidates=self.members[op_i]
            j_prime=random.choice(candidates)
            self.graph.remove_edge(u,v,key=k)
            self.graph.add_edge(i,j_prime)

        else:
            old_op=int(self.opinions[i])
            new_op=int(self.opinions[j])
            if new_op!=old_op:
                self.opinions[i]=new_op
                self._move_member(i, old_op, new_op)

        return True
#it scans the entire network to see if anyone is still fighting
#it looks at every single friendship. If the two friends have different opinions, it adds 1 to the count
#if this number is 0, it means everyone agrees with their friends. The simulation is finished
    def discordant_edge_count(self):
        c = 0
        for u,v,k in self.graph.edges(keys=True):
            if self.opinions[u] != self.opinions[v]:
                c +=1
        return c
#this runs the step function until the simulation ends
    def run_until_consensus(self,max_steps=20_000_000,check_every=None,verbose=False):
        
        if check_every is None:
            check_every =self.N

        steps=0
        while steps<max_steps:
            self.step()
            steps +=1

            if steps % check_every == 0:
                d=self.discordant_edge_count()
                if verbose:
                    print(f"steps={steps},discordant={d}")
                if d==0:
                    break

        return steps
#it counts the community size at the end of simulation
    def get_community_sizes(self):
        return [len(c) for c in nx.connected_components(self.graph)]
#it finds the biggest group and calculates what percentage of the population is in it
    def get_max_community_fraction(self):
        sizes = self.get_community_sizes()
        return (max(sizes)/self.N) if sizes else 0.0




def plot_figure_1_network_diagram():
    sim=HolmeNewmanSimulation(N=60,k_avg=4,gamma=6,phi=0.2,seed=1)

    for _ in range(sim.N*10):
        sim.step()

    plt.figure(figsize=(8,6))
    pos=nx.spring_layout(sim.graph,seed=42)
    node_colors=sim.opinions

    nx.draw_networkx_nodes(
        sim.graph,pos,
        node_size=250,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        edgecolors="k"
    )
    nx.draw_networkx_edges(sim.graph,pos,alpha=0.35)
    plt.title("Figure 1")
    plt.axis("off")
    plt.show()


def _log_binned_density(samples,N,nbins=25):

    samples=np.asarray(samples,dtype=float)
    edges=np.logspace(0, np.log10(N), nbins + 1) 
    counts,_ =np.histogram(samples,bins=edges)
    widths=edges[1:] - edges[:-1]
    total=counts.sum()
    density=counts/(total*widths) if total>0 else np.zeros_like(widths,dtype=float)
    centers =np.sqrt(edges[1:]*edges[:-1])       
    return centers,density


def plot_figure_2_distributions(realizations=50, nbins=25):
    """
    Fig 2: Community size distribution P(s) at consensus for:
      N=3200,k_avg=4,gamma=10,phi in {0.04, 0.458, 0.96}
    """
    N = 3200
    k_avg=4
    gamma=10
    phis=[0.04,0.458,0.96]

    fig, axes=plt.subplots(3,1,figsize=(7,10),sharex=True)

    for ax, phi in zip(axes, phis):
        all_sizes=[]
        for r in range(realizations):
            sim = HolmeNewmanSimulation(N=N,k_avg=k_avg,gamma=gamma,phi=phi,seed=None)
            sim.run_until_consensus(check_every=N) 
            all_sizes.extend(sim.get_community_sizes())

        centers,P= _log_binned_density(all_sizes,N, nbins=nbins)
        mask=P>0
        ax.loglog(centers[mask],P[mask],"o",markerfacecolor="none",markersize=5)
        ax.set_ylabel("P(s)")
        ax.text(0.95,0.85,f"φ={phi}",transform=ax.transAxes,ha="right")

        if abs(phi-0.458)<1e-9:
            alpha =3.5
            s0=30.0
            P0 = np.interp(s0,centers,P,left=np.nan,right=np.nan)
            if np.isfinite(P0) and P0>0:
                sline=np.array([10.0,300.0])
                pline =P0*(sline / s0)**(-alpha)
                ax.loglog(sline,pline,"-",linewidth=1)

    axes[-1].set_xlabel("s")
    plt.tight_layout()
    plt.show()


def plot_figure_3_finite_size_scaling(
    Ns=(200, 400, 800), runs_per_point=10, phi_grid=None):
    """
    Fig 3: Finite size scaling of S 
    Paper values:
      phi_c = 0.458, a = 0.61, b = 0.7 
    """
    k_avg =4
    gamma =10
    a =0.61
    b =0.7
    phi_c =0.458

    if phi_grid is None:
        phi_grid = np.unique(np.concatenate([
            np.linspace(0.0,1.0,41),
            np.linspace(0.44,0.47,21)
        ]))
        phi_grid.sort()

    data = {N: {"phi": [], "S": []} for N in Ns}

    for N in Ns:
        for phi in phi_grid:
            S_sum=0.0
            for _ in range(runs_per_point):
                sim = HolmeNewmanSimulation(N=N,k_avg=k_avg,gamma=gamma,phi=float(phi),seed=None)
                sim.run_until_consensus(check_every=N)
                S_sum += sim.get_max_community_fraction()
            data[N]["phi"].append(float(phi))
            data[N]["S"].append(S_sum/runs_per_point)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))

    for N in Ns:
        ph=np.array(data[N]["phi"])
        S=np.array(data[N]["S"])
        ax1.plot(ph,(N**a)*S,"o-",markersize=3,label=f"N={N}")
    ax1.axvline(phi_c,linestyle="--",alpha=0.6)
    ax1.set_xlabel("φ")
    ax1.set_ylabel(r"$N^{a} S$")
    ax1.set_title("Crossing plot")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    for N in Ns:
        ph=np.array(data[N]["phi"])
        S =np.array(data[N]["S"])
        x =(N**b)*(ph-phi_c)
        y =(N**a)*S
        ax2.plot(x,y,"o",markersize=3,label=f"N={N}")
    ax2.set_xlabel(r"$N^{b}(\phi-\phi_c)$")
    ax2.set_ylabel(r"$N^{a} S$")
    ax2.set_title("Data collapse")
    ax2.grid(True,alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_figure_1_network_diagram()
    plot_figure_2_distributions(realizations=20,nbins=25)
    plot_figure_3_finite_size_scaling(Ns=(200,400,800),runs_per_point=10)
