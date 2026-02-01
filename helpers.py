import numpy as np
from scipy.interpolate import interp1d

def _log_binned_density(samples, N, nbins=25):
    """Calculates P(s) using logarithmic binning"""
    samples=np.asarray(samples,dtype=float)
    if len(samples) == 0: return np.array([]), np.array([])
    edges=np.logspace(0,np.log10(N),nbins + 1)
    counts, _ =np.histogram(samples,bins=edges)
    widths=edges[1:] - edges[:-1]
    centers=np.sqrt(edges[1:] * edges[:-1])
    total=counts.sum()
    if total == 0: return centers,np.zeros_like(centers)
    density = counts/(total * widths)
    return centers,density

def estimate_phi_c_from_crossings(phi_grid,Ns,S,a=0.61,fallback=0.458):
    """Finds crossing point. Uses median of best-per-pair crossings"""
    Y=(Ns[:,None] ** a)*S
    per_pair_crossings=[]

    for i in range(len(Ns) - 1):
        diff=Y[i]-Y[i + 1]
        pair_candidates=[]
        for p in range(len(phi_grid) - 1):
            d0,d1 =diff[p],diff[p + 1]
            if np.isnan(d0) or np.isnan(d1): continue
            
            if d0*d1 < 0: # Standard Crossing
                x0,x1=phi_grid[p], phi_grid[p + 1]
                xc =x0- d0 * (x1 - x0) / (d1 - d0)
                score = abs(d0 - d1) 
                pair_candidates.append((score, xc))
            elif d0 == 0: # Exact Zero
                score = abs(d1)
                pair_candidates.append((score, phi_grid[p]))
            elif d1 == 0:
                score = abs(d0)
                pair_candidates.append((score, phi_grid[p+1]))

        if pair_candidates:
            pair_candidates.sort(key=lambda t: t[0], reverse=True)
            per_pair_crossings.append(pair_candidates[0][1])

    if not per_pair_crossings: return float(fallback)
    return float(np.median(per_pair_crossings))

def calculate_collapse_error(Ns,phi_grid,S,phi_c,a,b):
    """Calculates variance across overlapping regions."""
    window_mask=np.abs(phi_grid-phi_c)<=0.15
    if np.sum(window_mask) < 3: return float('inf')
        
    phi_window=phi_grid[window_mask]
    S_window=S[:, window_mask]
    
    scaled_S_list = []
    scaled_x_list = []
    
    for i, N in enumerate(Ns):
        if np.all(np.isnan(S_window[i])): continue
        x = (N ** b) * (phi_window - phi_c)
        y = (N ** a) * S_window[i]
        
        valid = ~np.isnan(y)
        if np.sum(valid) < 2: continue
        
        scaled_x_list.append(x[valid])
        scaled_S_list.append(y[valid])
    
    if len(scaled_x_list) < max(3, len(Ns)-1): 
        return float('inf')

    common_min = max([x.min() for x in scaled_x_list])
    common_max = min([x.max() for x in scaled_x_list])
    
    if common_max <= common_min: return float('inf')
    
    eval_x = np.linspace(common_min, common_max, 50)
    interpolated_vals = []
    
    for x, y in zip(scaled_x_list, scaled_S_list):
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
        interpolated_vals.append(f(eval_x))
    
    variance = np.nanvar(np.array(interpolated_vals), axis=0)
    return np.nansum(variance)

def solve_consistent_hetero_params(Ns, phi_grid, S, conv):
    """Iterative Solver for Heterogeneous Case"""
    print("  > Solving parameters iteratively...")
    
    low_conv_mask = conv < 0.2
    if np.any(low_conv_mask):
        S[low_conv_mask] = np.nan

    a, b, phi_c = 0.61, 0.7, 0.458
    
    for iteration in range(6):
        phi_c_new = estimate_phi_c_from_crossings(phi_grid, Ns, S, a)
        best_err = float('inf')
        best_a, best_b = a, b
        
        a_space = np.linspace(0.2, 0.9, 25)
        b_space = np.linspace(0.4, 1.0, 25)
        
        for test_a in a_space:
            for test_b in b_space:
                err = calculate_collapse_error(Ns, phi_grid, S, phi_c_new, test_a, test_b)
                if err < best_err:
                    best_err, best_a, best_b = err, test_a, test_b
        
        print(f"    Iter {iteration+1}: phi_c={phi_c_new:.4f}, a={best_a:.2f}, b={best_b:.2f}, err={best_err:.2e}")
        
        if abs(phi_c_new - phi_c) < 0.001 and abs(best_a - a) < 0.01 and abs(best_b - b) < 0.01:
            print("  Converged!")
            return best_a, best_b, phi_c_new
            
        a, b, phi_c = best_a, best_b, phi_c_new

    return a, b, phi_c
