# To run this script, you need to install matplotlib and numpy:
# pip install matplotlib numpy

import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: CORE LATTICE AND COST FUNCTIONS
# ==============================================================================

def delta_BKZ(b: int) -> float:
    """The root Hermite factor delta of BKZ-b."""
    return ((math.pi * b)**(1./b) * b / (2 * math.pi * math.exp(1)))**(1./(2.*b-2.))

def svp_classical(b: int) -> float:
    """log_2 of the best known classical cost of SVP in dimension b."""
    return b * math.log(math.sqrt(3./2)) / math.log(2)

# ==============================================================================
# SECTION 2: BKZ BASIS PROFILE SIMULATION
# ==============================================================================

def construct_BKZ_shape(q: int, nq: int, n1: int, b: int) -> tuple:
    """Simulate the (log) shape of a basis after BKZ-b reduction."""
    d = nq + n1
    if b == 0:
        L = nq * [math.log(q)] + n1 * [0]
        return (nq, nq, L)

    slope = -2 * math.log(delta_BKZ(b))
    lq = math.log(q)
    B = int(math.floor(lq / -slope))
    L = nq * [lq] + [lq + i * slope for i in range(1, B + 1)] + n1 * [0]

    x = 0
    lv = sum(L[:d])
    glv = nq * lq

    while lv > glv:
        lv -= L[x]
        lv += L[x + d]
        x += 1

    assert x <= B
    L = L[x:x+d]
    a = max(0, nq - x)
    B = min(B, d - a)

    diff = glv - lv
    assert abs(diff) < lq
    for i in range(a, a + B):
        L[i] += diff / B
    
    assert abs(sum(L) / glv - 1) < 1e-6
    return (a, a + B, L)

def BKZ_last_block_length(q: int, nq: int, n1: int, b: int) -> float:
    """Simulate the length of the expected Gram-Schmidt vector."""
    (_, _, L) = construct_BKZ_shape(q, nq, n1, b)
    return math.exp(L[nq + n1 - b])

# ==============================================================================
# SECTION 3: LWE PRIMAL ATTACK COST MODEL
# ==============================================================================

def LWE_primal_cost(q: int, n: int, m: int, s: float, b: int) -> float:
    """Return the cost of the primal attack."""
    log_infinity = 9999
    if s * math.sqrt(b) < BKZ_last_block_length(q, m, n + 1, b):
        return svp_classical(b)
    else:
        return log_infinity

# ==============================================================================
# SECTION 4: MODIFIED ATTACK OPTIMIZATION WITH VISUALIZATION
# ==============================================================================

def _plot_bkz_update(ax, q: int, nq: int, n1: int, b: int, cost: float):
    """Helper function to redraw the BKZ profile plot in the same window."""
    (a, a_plus_B, L) = construct_BKZ_shape(q, nq, n1, b)
    d = nq + n1

    ax.clear()
    ax.plot(np.arange(d), L, color='royalblue')
    ax.axhline(y=math.log(q), color='r', linestyle='--', label=f'log(q)')
    ax.axhline(y=0, color='g', linestyle='--', label='log(1)')
    ax.set_xlabel('Vector Index in Basis', fontsize=12)
    ax.set_ylabel('Log(GS-Length)', fontsize=12)
    ax.set_title('Best Basis Profile Found So Far', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    
    info_text = f"Cost: {math.ceil(cost)} bits\nBlocksize b: {b}\nSamples m: {nq}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Draw the canvas and pause briefly to create an animation effect
    plt.draw()
    plt.pause(0.02)

def MLWE_optimize_attack(q: int, n: int, max_m: int, s: float, visualize_progress: bool = False):
    """Find optimal parameters for a given attack, with optional live visualization."""
    log_infinity = 9999
    best_cost = log_infinity
    best_b, best_m = None, None
    
    # Setup for visualization
    fig, ax = None, None
    if visualize_progress:
        plt.ion() # Turn on interactive mode for live plotting
        fig, ax = plt.subplots(figsize=(10, 6))

    b_min, b_max = 50, n + max_m
    b_step = max(1, (b_max - b_min) // 4)
    while b_step > 0:
        for b in range(b_min, b_max + 1, b_step):
            if svp_classical(b) > best_cost:
                b_max = b - 1
                break
            for m in range(max_m, max(0, b - n), -1):
                cost = LWE_primal_cost(q, n, m, s, b)
                if cost == log_infinity:
                    break
                if cost <= best_cost:
                    best_cost, best_m, best_b = cost, m, b
                    b_min = max(b_min, b - b_step + 1)
                    
                    if visualize_progress:
                        _plot_bkz_update(ax, q, best_m, n + 1, best_b, best_cost)
        b_step = b_step // 2

    if visualize_progress:
        plt.ioff() # Turn off interactive mode
        print("\nOptimization complete. Close the plot window to see the final results.")
        plt.show() # Show the final plot and block until it's closed
        
    return (best_m, best_b, best_cost)

# ==============================================================================
# SECTION 5: RUNNING THE ESTIMATOR
# ==============================================================================

if __name__ == "__main__":
    # --- Define your LWE parameters here ---
    n = 1280       # LWE secret dimension
    q = 8380417      # Modulus
    k = 4         # Error distribution parameter for U(-k, k)

    # --- Calculations ---
    s = math.sqrt(sum([i**2 for i in range(-k, k + 1)]) / (2 * k + 1))
    max_m = n + 100 # Maximum number of samples to consider

    # --- Run the optimization with visualization ---
    print(f"Estimating primal attack cost for LWE(n={n}, q={q}, s={s:.2f})\n")
    print("Starting optimizer... A plot window will appear and update live.")

    (best_m, best_b, best_cost) = MLWE_optimize_attack(
        q, n, max_m, s, 
        visualize_progress=True # <-- SET TO TRUE FOR VISUALIZATION
    )

    # --- Final Results ---
    print("\n" + "="*40)
    print("              FINAL RESULTS")
    print("="*40)
    if best_m is not None:
        print(f"Optimal number of samples m = {best_m}")
        print(f"Optimal block size b        = {best_b}")
        print(f"Estimated cost (security)   = {math.ceil(best_cost)} bits")
    else:
        print("No successful attack found in the given parameter range.")
    print("="*40)
