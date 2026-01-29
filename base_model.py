import networkx as nx
import numpy as np
import random
import time
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
        i = random.randrange(self.N)

        e = self._random_incident_edge(i)
        if e is None:
            return True 
        u, v, k = e
        j = self._other_endpoint(i, u, v)

        if random.random() < self.phi:
            op_i = int(self.opinions[i])
            candidates = self.members[op_i]
            valid_choices = [c for c in candidates if c != i]            
            if valid_choices:
                j_prime = random.choice(valid_choices)
                self.graph.remove_edge(u, v, key=k)
                self.graph.add_edge(i, j_prime)

        else:
            old_op = int(self.opinions[i])
            new_op = int(self.opinions[j])
            if new_op != old_op:
                self.opinions[i] = new_op
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
class HeterogeneousSimulation(HolmeNewmanSimulation):

    def __init__(self, N=3200,k_avg=4,gamma=10,seed=None,
                 type_probs=[0.1,0.8,0.05,0.05],
                 type_phi_values={0:0.05, 1:0.45, 2:1.0,3:0.45},
                 type_stubbornness_values={0:0.0, 1:0.5, 2:1.0,3:0.0}):
        super().__init__(N=N,k_avg=k_avg,gamma=gamma,phi=0,seed=seed)
        self.type_phi_values=type_phi_values
        self.agent_types=np.random.choice([0,1,2,3],size=self.N,p=type_probs)
        self.stubbornness=np.array(
            [type_stubbornness_values[int(t)] for t in self.agent_types],
            dtype=float
        )
    def step(self):
        i=random.randrange(self.N)
        e= self._random_incident_edge(i)
        if e is None: return True
        u,v,k = e
        j = self._other_endpoint(i,u,v)

        # lookup individual phi
        my_type=self.agent_types[i]
        my_phi=self.type_phi_values[my_type]

        if random.random() < my_phi:
            op_i=int(self.opinions[i])
            if my_type ==3:
                other_ops = [op for op in range(self.G_count) if self.members[op] and op != op_i]
                op_j = int(random.choice(other_ops))
                candidates = self.members[op_j]
            else:
                candidates =self.members[op_i]
                candidates =[c for c in candidates if c != i]
            if candidates:
                j_prime =random.choice(candidates)
                self.graph.remove_edge(u,v,key=k)
                self.graph.add_edge(i,j_prime)
        else:
            old_op =int(self.opinions[i])
            new_op=int(self.opinions[j])
            if new_op != old_op and random.random() < (1.0-float(self.stubbornness[i])):
                self.opinions[i]=new_op
                self._move_member(i,old_op,new_op)
        return True

def run_once(m):
    sim = HeterogeneousSimulation(
        N=400, seed=0,type_probs=[0.1,0.85-m,0.05,m])
    sim.run_until_consensus()
    return sim.get_max_community_fraction()

#print("no mediators:", run_once(0.0))
#print("1% mediators:", run_once(0.01))

def run_once_2(m):
    sim = HeterogeneousSimulation(
        N=400, seed=0,type_probs=[0.1,0.85-m,0.05,m])
    return sim.run_until_consensus()
    


times = []

def time_call(fn, var):
    start = time.perf_counter()
    result = fn(var)
    end = time.perf_counter()
    return result, end - start

intervals = [x / 1000.0 for x in range(0, 50, 1)]
print(intervals)
for i in intervals:
    res, elapsed = time_call(run_once_2, i)
    print(f"m: {i} time: {elapsed} steps_til_consensus: {res}")
    times.append(elapsed)
print(times)

def get_max_agree_component_fraction(self):
        G_agree = nx.Graph()
        G_agree.add_nodes_from(range(self.N))
        # add edges only if opinions match
        valid_edges = []
        for u, v, k in self.graph.edges(keys=True):
            if self.opinions[u] == self.opinions[v]:
                valid_edges.append((u, v))
        G_agree.add_edges_from(valid_edges)
        sizes = [len(c) for c in nx.connected_components(G_agree)]
        return max(sizes) / self.N if sizes else 0.0

