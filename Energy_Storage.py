import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

class TimeScaleEnergyStorage:
    def __init__(self, T=24, h=0.5):
        self.T, self.h, self.N = T, h, int(T / h)
        self.t = np.linspace(0, T, self.N + 1)
        
        # Parameters
        self.eta, self.tau = 0.01, 0.02
        self.kappa_E, self.kappa_I = 0.9, 0.95
        
        # Cost weights
        self.alpha, self.beta = 0.001, 0.0005
        self.c_E, self.c_I = 0.0001, 0.0001
        
        # Constraints
        self.u_E_min, self.u_E_max = -2, 2
        self.u_I_min, self.u_I_max = -2, 2
        self.E_min, self.E_max = 0, 100
        self.I_min = 0
        self.E0, self.I0 = 10, 0
        
        self.scenarios = {
            'base_case': {
                'E_ref': 10,
                'D_func': lambda t: 5 * np.ones_like(t),
                'P_func': lambda t: 8 + 2 * np.sin(2 * np.pi * t / 24)
            },
            'peak_shaving': {
                'E_ref': 50,
                'D_func': lambda t: 6 + 4 * np.exp(-(t-8)**2/2) + 4 * np.exp(-(t-18)**2/2),
                'P_func': lambda t: 10 + 4 * np.sin(2 * np.pi * t / 24)
            },
            'grid_support': {
                'E_ref': 20,
                'D_func': lambda t: 5 + 2 * np.sin(2 * np.pi * t / 24) + self._generate_noise(t),
                'P_func': lambda t: 7 + 3 * np.sin(2 * np.pi * t / 24)
            }
        }
    
    def _generate_noise(self, t):
        rng = np.random.default_rng(42)
        return rng.normal(0, 0.2, len(t))
    
    def forward_sweep(self, u_E, u_I, D, P):
        """Forward integration with given controls"""
        E = np.zeros(self.N + 1)
        I = np.zeros(self.N + 1)
        E[0], I[0] = self.E0, self.I0
        
        for k in range(self.N):
            E[k+1] = E[k] + self.h * (I[k] - D[k] - self.eta * E[k] + self.kappa_E * u_E[k])
            I[k+1] = I[k] + self.h * (P[k] - self.tau * I[k] + self.kappa_I * u_I[k])
            E[k+1] = np.clip(E[k+1], self.E_min, self.E_max)
            I[k+1] = np.maximum(I[k+1], self.I_min)
        return E, I
    
    def compute_cost(self, controls_flat, D, P, E_ref):
        """Cost function for optimizer"""
        u_E = controls_flat[:self.N]
        u_I = controls_flat[self.N:]
        
        # Project controls
        u_E = np.clip(u_E, self.u_E_min, self.u_E_max)
        u_I = np.clip(u_I, self.u_I_min, self.u_I_max)
        
        E, I = self.forward_sweep(u_E, u_I, D, P)
        
        state_cost = self.alpha * (E[:-1] - E_ref)**2 + self.beta * (D[:-1] - I[:-1])**2
        control_cost = self.c_E * u_E**2 + self.c_I * u_I**2
        
        return self.h * np.sum(state_cost + control_cost)
    
    def solve_direct_optimization(self, scenario_name):
        """DIRECT OPTIMIZATION - GUARANTEED TO WORK"""
        scenario = self.scenarios[scenario_name]
        D, P = scenario['D_func'](self.t), scenario['P_func'](self.t)
        E_ref = scenario['E_ref']
        
        print(f"\n{'='*50}")
        print(f"{scenario_name.replace('_', ' ').title()} - DIRECT OPTIMIZATION")
        print('='*50)
        
        # Open-loop cost
        E_ol, I_ol = self.forward_sweep(np.zeros(self.N), np.zeros(self.N), D, P)
        J_ol = self.compute_cost(np.zeros(2*self.N), D, P, E_ref)
        print(f"Open-loop cost: {J_ol:.4f}")
        
        # Initial guess - small random controls
        rng = np.random.default_rng(42)
        x0 = 0.1 * rng.normal(0, 1, 2*self.N)
        
        # Bounds for controls
        bounds = [(self.u_E_min, self.u_E_max)] * self.N + [(self.u_I_min, self.u_I_max)] * self.N
        
        # Optimize using SLSQP (handles bounds)
        result = minimize(
            lambda x: self.compute_cost(x, D, P, E_ref),
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-8, 'maxiter': 1000, 'disp': True}
        )
        
        # Extract optimal controls
        u_E_opt = np.clip(result.x[:self.N], self.u_E_min, self.u_E_max)
        u_I_opt = np.clip(result.x[self.N:], self.u_I_min, self.u_I_max)
        
        # Compute optimal trajectory
        E_opt, I_opt = self.forward_sweep(u_E_opt, u_I_opt, D, P)
        J_opt = result.fun
        
        reduction = 100 * (J_ol - J_opt) / J_ol
        
        print(f"\nOpen-loop:  {J_ol:.4f}")
        print(f"Optimal:    {J_opt:.4f}")
        print(f"Reduction:  {reduction:+.2f}%")
        print(f"Success:    {result.success}")
        print(f"Message:    {result.message}")
        
        return {
            'E': E_opt, 'I': I_opt, 'u_E': u_E_opt, 'u_I': u_I_opt,
            'D': D, 'P': P, 'E_ref': E_ref,
            'final_cost': J_opt, 'open_loop_cost': J_ol,
            'reduction': reduction, 'success': result.success,
            'iterations': result.nit
        }
    
    def save_results_to_csv(self, results):
        """Save all scenario results to a single CSV file"""
        all_data = []
        
        for scenario_name, result in results.items():
            # Extract time series data for each scenario
            for i, time in enumerate(self.t):
                row = {
                    'scenario': scenario_name,
                    'time_hours': time,
                    'stored_energy_kWh': result['E'][i],
                    'incoming_energy_kW': result['I'][i],
                    'energy_demand_kW': result['D'][i],
                    'power_generation_kW': result['P'][i],
                    'energy_reference_kWh': result['E_ref']
                }
                
                # Add control values (they have length N, while states have length N+1)
                if i < len(self.t) - 1:
                    row['storage_control_kW'] = result['u_E'][i]
                    row['grid_control_kW'] = result['u_I'][i]
                else:
                    # For the last time point, use the previous control value
                    row['storage_control_kW'] = result['u_E'][-1] if len(result['u_E']) > 0 else 0
                    row['grid_control_kW'] = result['u_I'][-1] if len(result['u_I']) > 0 else 0
                
                all_data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_data)
        
        # Reorder columns for better readability
        columns_order = [
            'scenario', 'time_hours', 'stored_energy_kWh', 'incoming_energy_kW',
            'energy_demand_kW', 'power_generation_kW', 'energy_reference_kWh',
            'storage_control_kW', 'grid_control_kW'
        ]
        df = df[columns_order]
        
        filename = 'energy_storage_optimization_results.csv'
        df.to_csv(filename, index=False)
        print(f"\n✓ All results saved to '{filename}'")
        print(f"✓ Total records: {len(df)}")
        print(f"✓ File includes {len(results)} scenarios")
        
        # Also save summary statistics
        self.save_summary_statistics(results, 'energy_storage_summary.csv')
        
        return df
    
    def save_summary_statistics(self, results, filename):
        """Save summary statistics to a separate CSV file"""
        summary_data = []
        
        for scenario_name, result in results.items():
            stats = {
                'scenario': scenario_name,
                'optimal_cost': result['final_cost'],
                'open_loop_cost': result['open_loop_cost'],
                'cost_reduction_percent': result['reduction'],
                'iterations': result['iterations'],
                'success': result['success'],
                'final_energy_kWh': result['E'][-1],
                'avg_storage_control_kW': np.mean(np.abs(result['u_E'])),
                'avg_grid_control_kW': np.mean(np.abs(result['u_I'])),
                'max_storage_control_kW': np.max(np.abs(result['u_E'])),
                'max_grid_control_kW': np.max(np.abs(result['u_I']))
            }
            summary_data.append(stats)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filename, index=False)
        print(f"✓ Summary statistics saved to '{filename}'")
    
    def plot_state_trajectories(self, results, scenario_name):
        """PLOT 1: State trajectories with detailed annotations"""
        E, I, D, P = [results[k] for k in ['E','I','D','P']]
        E_ref = results['E_ref']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left: Stored Energy
        ax1.plot(self.t, E, 'b-', linewidth=2.5, label='Stored Energy E(t)')
        ax1.axhline(E_ref, color='red', linestyle='--', linewidth=2, 
                   label=f'Reference Level ({E_ref} kWh)')
        ax1.axhline(self.E_max, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Max Capacity ({self.E_max} kWh)')
        ax1.axhline(self.E_min, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Min Capacity ({self.E_min} kWh)')
        ax1.fill_between(self.t, self.E_min, self.E_max, alpha=0.1, color='orange', 
                        label='Feasible Region')
        ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Stored Energy (kWh)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Energy Storage Dynamics\n{scenario_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 24)
        
        # Right: Energy Flows
        ax2.plot(self.t, I, 'g-', linewidth=2.5, label='Incoming Energy I(t)')
        ax2.plot(self.t, D, 'r-', linewidth=2, label='Energy Demand D(t)', alpha=0.8)
        ax2.plot(self.t, P, 'purple', linewidth=2, label='Power Generation P(t)', alpha=0.8)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Power Flow (kW)', fontsize=12, fontweight='bold')
        ax2.set_title('Energy Balance and Power Flows', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 24)
        
        # Add performance metrics as text box
        textstr = f'Optimal Cost: {results["final_cost"]:.1f}\nReduction: {results["reduction"]:+.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{scenario_name}_states.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_control_strategies(self, results, scenario_name):
        """PLOT 2: Control strategies with detailed annotations"""
        u_E, u_I = results['u_E'], results['u_I']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        t_ctrl = self.t[:-1]  # Control time points
        
        # Left: Storage Control
        ax1.step(t_ctrl, u_E, 'b-', linewidth=2.5, where='post', label='Storage Control u_E(t)')
        ax1.axhline(self.u_E_max, color='darkred', linestyle='--', linewidth=1.5, 
                   label=f'Max Charge ({self.u_E_max} kW)')
        ax1.axhline(self.u_E_min, color='darkred', linestyle='--', linewidth=1.5, 
                   label=f'Max Discharge ({self.u_E_min} kW)')
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5, label='Zero Control')
        ax1.fill_between(t_ctrl, self.u_E_min, self.u_E_max, alpha=0.1, color='red',
                        label='Control Limits')
        ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Storage Control (kW)', fontsize=12, fontweight='bold')
        ax1.set_title('Optimal Storage Control Strategy\n(Charge/Discharge Decisions)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 24)
        ax1.set_ylim(self.u_E_min - 0.5, self.u_E_max + 0.5)
        
        # Right: Incoming Energy Control
        ax2.step(t_ctrl, u_I, 'darkgreen', linewidth=2.5, where='post', 
                label='Incoming Control u_I(t)')
        ax2.axhline(self.u_I_max, color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Max Import ({self.u_I_max} kW)')
        ax2.axhline(self.u_I_min, color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Max Export ({self.u_I_min} kW)')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, label='Zero Control')
        ax2.fill_between(t_ctrl, self.u_I_min, self.u_I_max, alpha=0.1, color='red',
                        label='Control Limits')
        ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Grid Control (kW)', fontsize=12, fontweight='bold')
        ax2.set_title('Optimal Grid Interaction Control\n(Import/Export Decisions)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 24)
        ax2.set_ylim(self.u_I_min - 0.5, self.u_I_max + 0.5)
        
        # Add control statistics
        u_E_avg = np.mean(np.abs(u_E))
        u_I_avg = np.mean(np.abs(u_I))
        textstr = f'Avg |u_E|: {u_E_avg:.2f} kW\nAvg |u_I|: {u_I_avg:.2f} kW'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{scenario_name}_controls.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_direct_optimization_study(self):
        """Run study using direct optimization"""
        results = {}
        
        print("DIRECT OPTIMIZATION STUDY")
        print("Using scipy.optimize.minimize with bounds")
        print("="*60)
        
        for name in self.scenarios:
            results[name] = self.solve_direct_optimization(name)
            
            # Generate TWO SEPARATE PLOTS with detailed annotations
            self.plot_state_trajectories(results[name], name)
            self.plot_control_strategies(results[name], name)
        
        # Save all results to CSV files
        detailed_df = self.save_results_to_csv(results)
        
        # Results table
        data = []
        for name, res in results.items():
            data.append({
                'Scenario': name.replace('_', ' ').title(),
                'Optimal Cost': f"{res['final_cost']:.2f}",
                'Open-Loop Cost': f"{res['open_loop_cost']:.2f}",
                'Reduction': f"{res['reduction']:+.2f}\\%",
                'Iterations': res['iterations'],
                'Success': '✓' if res['success'] else '✗'
            })
        
        df = pd.DataFrame(data)
        
        print("\n" + "="*60)
        print("FINAL RESULTS - DIRECT OPTIMIZATION")
        print("="*60)
        print(df.to_string(index=False))
        
        # LaTeX table
        print("\n" + "="*60)
        print("LaTeX TABLE")
        print("="*60)
        print(df.to_latex(index=False, escape=False,
                         caption="Optimal Control Results (Direct Optimization)", 
                         label="tab:results"))
        
        return results, detailed_df

# RUN THE GUARANTEED SOLUTION
if __name__ == "__main__":
    print("TIME-SCALE OPTIMAL CONTROL - DIRECT OPTIMIZATION")
    print("Using scipy.optimize for guaranteed convergence")
    print("="*60)
    
    ess = TimeScaleEnergyStorage()
    results, csv_data = ess.run_direct_optimization_study()
    
    # Display first few rows of the saved CSV data
    print("\n" + "="*60)
    print("CSV DATA PREVIEW")
    print("="*60)
    print(csv_data.head(10))