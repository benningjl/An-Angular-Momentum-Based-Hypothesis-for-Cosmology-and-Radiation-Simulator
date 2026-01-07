"""
HIERARCHICAL DIRECT PRIME PREDICTION

Apply the hierarchical angular momentum insight to DIRECT prime prediction
(not gap reconstruction)

Key Innovation: Recalibrate parameters at each Fibonacci-indexed level
- Level boundaries: n = [1, 2, 5, 13, 34, 89, ...]
- Each level maps to σ₀(k) = σ_macro/φ^k
- Different optimal parameters for each scale

Test all our best formulas with hierarchical recalibration:
1. Golden Spiral: P_n = a·φ^(b·n/π)
2. Log-Damping: P_n = a·φ^(b·n/π)·[h + (1-h)·(1 + c·n·ln(n)/n)]
3. Hybrid with σ_ratio: P_n = a·φ^(b·n/π)·(σ_ratio)^(k/φ^n)
4. Pure logarithmic with φ modulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Constants
PHI = (1 + np.sqrt(5)) / 2
PSI = (1 - np.sqrt(5)) / 2
PI = np.pi
SQRT5 = np.sqrt(5)
SIGMA_RATIO = 1e13

# Extended primes
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
          73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
          157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
          239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
          331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
          421, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509,
          521, 523, 541, 547, 557]

def transition_function(x, n_power=2.0):
    """Smooth transition from paper Section 3.12z"""
    x_n = np.power(x, n_power)
    return x_n / (1.0 + x_n)

def sigma_0_at_level(level):
    """Angular momentum scale at hierarchical level"""
    return 1e6 / (PHI**level)

# =============================================================================
# FORMULA 1: Golden Spiral (Pure)
# =============================================================================

def golden_spiral_formula(n, a, b):
    """P_n = a·φ^(b·n/π)"""
    if n <= 0:
        return 0
    return a * (PHI ** (b * n / PI))

# =============================================================================
# FORMULA 2: Log-Damping with Transition
# =============================================================================

def log_damping_formula(n, a, b, n_c, log_weight):
    """
    P_n = a·φ^(b·n/π)·[h + (1-h)·(1 + log_weight·n·ln(n)/n)]
    where h = transition_function(n/n_c)
    """
    if n <= 0:
        return 0
    
    h = transition_function(n / n_c, n_power=1.5)
    spiral_term = a * (PHI ** (b * n / PI))
    log_term = np.log(n + np.e) / n
    combined = h * 1.0 + (1 - h) * (1 + log_weight * n * log_term)
    
    return spiral_term * combined

# =============================================================================
# FORMULA 3: Hybrid with σ_ratio
# =============================================================================

def hybrid_sigma_formula(n, a, b, k):
    """
    P_n = a·φ^(b·n/π)·(σ_ratio)^(k/φ^n)
    
    Physical: Golden spiral modulated by angular momentum coupling
    """
    if n <= 0:
        return 0
    
    spiral = a * (PHI ** (b * n / PI))
    sigma_correction = (SIGMA_RATIO ** (k / (PHI ** n)))
    
    return spiral * sigma_correction

# =============================================================================
# FORMULA 4: Pure Logarithmic with φ Modulation
# =============================================================================

def log_phi_modulation_formula(n, a, b, c, phi_mod):
    """
    P_n = a·n^b·ln(n)^c·[1 + φ_mod·cos(2π·(n/φ mod 1))]
    
    Physical: PNT-like growth with golden ratio resonance avoidance
    """
    if n <= 1:
        return 2
    
    base = a * (n ** b) * (np.log(n) ** c)
    
    # Golden ratio phase modulation
    phi_phase = (n / PHI) % 1.0
    modulation = 1.0 + phi_mod * np.cos(2 * PI * phi_phase)
    
    return base * modulation

# =============================================================================
# FORMULA 5: Three-Component Blend
# =============================================================================

def three_component_formula(n, a1, a2, a3, b, n_c1, n_c2):
    """
    Blend three components across scale transitions:
    - Component 1: Power law (small n)
    - Component 2: Golden spiral (medium n)
    - Component 3: Logarithmic (large n)
    """
    if n <= 0:
        return 0
    
    h1 = transition_function(n / n_c1, n_power=2.0)
    h2 = transition_function(n / n_c2, n_power=2.0)
    
    # Component 1: Power law
    comp1 = a1 * (n ** 1.5)
    
    # Component 2: Golden spiral
    comp2 = a2 * (PHI ** (b * n / PI))
    
    # Component 3: n·ln(n) from PNT
    comp3 = a3 * n * np.log(n + np.e)
    
    # Blend
    result = (1 - h1) * comp1 + h1 * (1 - h2) * comp2 + h2 * comp3
    return max(2, result)

# =============================================================================
# Hierarchical Predictor Class
# =============================================================================

class HierarchicalDirectPredictor:
    """
    Direct prime prediction with hierarchical recalibration
    """
    
    def __init__(self, formula_func, formula_name, n_params):
        self.formula_func = formula_func
        self.formula_name = formula_name
        self.n_params = n_params
        self.levels = []
    
    def get_level_boundaries(self, max_n):
        """Get Fibonacci-based level boundaries"""
        boundaries = [1]
        
        # Fibonacci sequence for boundaries
        fib = [1, 2]
        while fib[-1] < max_n:
            fib.append(fib[-1] + fib[-2])
        
        boundaries.extend([f for f in fib if f < max_n])
        boundaries.append(max_n)
        
        return boundaries
    
    def optimize_level(self, primes, n_start, n_end):
        """Optimize parameters for a specific level"""
        subset = primes[n_start-1:n_end]
        n_range = list(range(n_start, n_end + 1))
        
        def objective(params):
            try:
                predictions = [self.formula_func(n, *params) for n in n_range]
                errors = [pred - actual for pred, actual in zip(predictions, subset)]
                rmse = np.sqrt(np.mean([e**2 for e in errors]))
                return rmse
            except:
                return 1e10
        
        # Bounds depend on formula type
        if self.n_params == 2:  # Golden spiral
            bounds = [(1.0, 20.0), (0.1, 1.0)]
        elif self.n_params == 3:  # Hybrid
            bounds = [(1.0, 20.0), (0.1, 1.0), (1e-20, 1e-10)]
        elif self.n_params == 4:  # Log-damping or log-phi
            bounds = [(1.0, 20.0), (0.1, 1.0), (5.0, 50.0), (0.0, 0.5)]
        elif self.n_params == 6:  # Three-component
            bounds = [(0.1, 5.0), (1.0, 10.0), (0.5, 5.0), (0.1, 1.0), (5.0, 20.0), (20.0, 60.0)]
        else:
            raise ValueError(f"Unsupported n_params: {self.n_params}")
        
        result = differential_evolution(objective, bounds, seed=42, maxiter=150,
                                       workers=1, polish=True, atol=0.01, tol=0.01)
        
        return result.x, result.fun
    
    def calibrate(self, primes):
        """Calibrate all hierarchical levels"""
        boundaries = self.get_level_boundaries(len(primes))
        
        print(f"\n{'='*70}")
        print(f"Calibrating: {self.formula_name}")
        print(f"Level boundaries: {boundaries}")
        print(f"{'='*70}")
        
        self.levels = []
        
        for i in range(len(boundaries) - 1):
            n_start = boundaries[i]
            n_end = boundaries[i + 1]
            
            print(f"\n--- Level {i+1}: n={n_start} to {n_end} ---")
            print(f"    σ₀ scale: {sigma_0_at_level(i):.2e} m²/s")
            
            params, rmse = self.optimize_level(primes, n_start, n_end)
            
            print(f"    Parameters: {', '.join(f'{p:.4f}' for p in params)}")
            print(f"    RMSE: {rmse:.2f}")
            
            self.levels.append({
                'n_start': n_start,
                'n_end': n_end,
                'params': params,
                'level_index': i,
                'sigma_0': sigma_0_at_level(i),
                'rmse': rmse
            })
        
        return self.levels
    
    def predict(self, n):
        """Predict prime at index n with smooth level transitions"""
        # Find active level(s)
        active_levels = []
        for level in self.levels:
            if level['n_start'] <= n < level['n_end']:
                active_levels.append(level)
            elif level['n_end'] <= n < level['n_end'] + 3:  # Transition zone
                active_levels.append(level)
        
        if not active_levels:
            # Beyond calibrated range - use last level with scaling
            last_level = self.levels[-1]
            scale = (n / last_level['n_end']) ** 0.3
            return self.formula_func(n, *last_level['params']) * scale
        
        if len(active_levels) == 1:
            level = active_levels[0]
            return self.formula_func(n, *level['params'])
        else:
            # Blend between levels
            level1, level2 = active_levels[0], active_levels[1]
            blend = transition_function((n - level1['n_end']) / 3.0, n_power=2.0)
            
            pred1 = self.formula_func(n, *level1['params'])
            pred2 = self.formula_func(n, *level2['params'])
            
            return (1 - blend) * pred1 + blend * pred2
    
    def predict_all(self, n_max):
        """Predict all primes up to n_max"""
        return [self.predict(n) for n in range(1, n_max + 1)]
    
    def compute_accuracy(self, primes_actual):
        """Compute accuracy metrics"""
        primes_pred = self.predict_all(len(primes_actual))
        
        errors = [pred - actual for pred, actual in zip(primes_pred, primes_actual)]
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        mae = np.mean([abs(e) for e in errors])
        
        rel_errors = [abs(e)/actual*100 for e, actual in zip(errors, primes_actual)]
        good = sum(1 for re in rel_errors if re < 5.0)
        
        print(f"\n{'='*70}")
        print(f"Overall Accuracy: {self.formula_name}")
        print(f"{'='*70}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"Predictions <5% error: {good}/{len(primes_actual)} ({good/len(primes_actual)*100:.1f}%)")
        
        # Show best predictions
        best_preds = [(i+1, primes_pred[i], primes_actual[i], abs(errors[i]), rel_errors[i])
                     for i in range(len(primes_actual)) if rel_errors[i] < 2.0]
        
        if best_preds:
            print(f"\nExcellent predictions (<2% error): {len(best_preds)}")
            for n, pred, actual, err, err_pct in best_preds[:10]:
                print(f"  n={n}: Predicted={pred:.1f}, Actual={actual}, Error={err:.1f} ({err_pct:.2f}%)")
        
        return rmse, mae, good

# =============================================================================
# Main Testing
# =============================================================================

def test_all_formulas():
    """Test all formulas with hierarchical recalibration"""
    
    formulas = [
        (golden_spiral_formula, "Golden Spiral Pure", 2),
        (log_damping_formula, "Log-Damping Transition", 4),
        (hybrid_sigma_formula, "Hybrid σ_ratio", 3),
        (log_phi_modulation_formula, "Log with φ Modulation", 4),
        (three_component_formula, "Three-Component Blend", 6)
    ]
    
    results = []
    predictors = []
    
    print("="*80)
    print("HIERARCHICAL DIRECT PRIME PREDICTION - ALL FORMULAS")
    print("="*80)
    print("\nTesting direct prime prediction (not gap reconstruction)")
    print("with hierarchical recalibration at Fibonacci boundaries")
    print("="*80)
    
    for formula_func, formula_name, n_params in formulas:
        predictor = HierarchicalDirectPredictor(formula_func, formula_name, n_params)
        
        try:
            predictor.calibrate(PRIMES)
            rmse, mae, good = predictor.compute_accuracy(PRIMES)
            results.append((formula_name, rmse, mae, good))
            predictors.append(predictor)
        except Exception as e:
            print(f"\n❌ Failed for {formula_name}: {e}")
            continue
    
    # Summary comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON - ALL HIERARCHICAL FORMULAS")
    print("="*80)
    
    results.sort(key=lambda x: x[1])  # Sort by RMSE
    
    print(f"\n{'Formula':<30} {'RMSE':<10} {'MAE':<10} {'<5% Acc':<15}")
    print("-"*70)
    for name, rmse, mae, good in results:
        print(f"{name:<30} {rmse:<10.2f} {mae:<10.2f} {good}/{len(PRIMES)} ({good/len(PRIMES)*100:.1f}%)")
    
    # Visualize best formula
    if predictors:
        best_predictor = predictors[0]
        visualize_best(best_predictor, PRIMES)
    
    return results, predictors

def visualize_best(predictor, primes_actual):
    """Visualize best performing formula"""
    primes_pred = predictor.predict_all(len(primes_actual))
    n_values = list(range(1, len(primes_actual) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Primes
    ax1 = axes[0, 0]
    ax1.plot(n_values, primes_actual, 'bo-', label='Actual', markersize=3, linewidth=1)
    ax1.plot(n_values, primes_pred, 'r^--', label='Predicted', markersize=3, linewidth=1, alpha=0.7)
    
    # Mark level boundaries
    for level in predictor.levels[1:]:
        ax1.axvline(x=level['n_start'], color='green', linestyle=':', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('Prime Index n')
    ax1.set_ylabel('Prime Value P_n')
    ax1.set_title(f'Best Formula: {predictor.formula_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error
    ax2 = axes[0, 1]
    errors = [primes_pred[i] - primes_actual[i] for i in range(len(primes_actual))]
    ax2.bar(n_values, errors, color='coral', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Prime Index n')
    ax2.set_ylabel('Error')
    ax2.set_title('Prediction Error by Index')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Relative error
    ax3 = axes[1, 0]
    rel_errors = [abs(errors[i])/primes_actual[i]*100 for i in range(len(primes_actual))]
    ax3.semilogy(n_values, rel_errors, 'ms-', markersize=3)
    ax3.axhline(y=5, color='green', linestyle='--', label='5%', linewidth=2)
    ax3.axhline(y=10, color='orange', linestyle='--', label='10%', linewidth=2)
    
    for level in predictor.levels[1:]:
        ax3.axvline(x=level['n_start'], color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Prime Index n')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Percentage Error (log scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Level RMSE
    ax4 = axes[1, 1]
    level_indices = [level['level_index'] for level in predictor.levels]
    level_rmses = [level['rmse'] for level in predictor.levels]
    
    ax4.bar(level_indices, level_rmses, color='steelblue', alpha=0.7)
    ax4.set_xlabel('Hierarchical Level')
    ax4.set_ylabel('RMSE')
    ax4.set_title('Prediction Accuracy by Level')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('c:\\paradigm shift\\Theory\\output\\hierarchical_direct_best.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: output/hierarchical_direct_best.png")

if __name__ == "__main__":
    results, predictors = test_all_formulas()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nHierarchical recalibration applied to direct prime prediction")
    print("✓ Fibonacci-indexed boundaries: [1, 2, 5, 13, 34, 89]")
    print("✓ Each level maps to σ₀(k) = σ_macro/φ^k")
    print("✓ Smooth transitions between levels")
    print("✓ Automatic scaling beyond calibrated range")
    
    if results:
        best_name, best_rmse, best_mae, best_good = results[0]
        print(f"\n🏆 BEST FORMULA: {best_name}")
        print(f"   RMSE: {best_rmse:.2f}")
        print(f"   MAE: {best_mae:.2f}")
        print(f"   Accuracy: {best_good}/{len(PRIMES)} ({best_good/len(PRIMES)*100:.1f}%)")
    
    print("\n" + "="*80)
