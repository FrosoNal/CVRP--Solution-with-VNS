"""
cvrp_visualization.py

Plotting utilities for a solved CVRPSolver instance.

visualize_solution() draws two side-by-side panels:
    1. The routes themselves (depot, customers, and colored route lines)
    2. The VNS convergence curve (best cost per iteration), optionally
       compared against a reference cost line
"""

import numpy as np
import matplotlib.pyplot as plt

from cvrp_solver import CVRPSolver


def visualize_solution(solver: CVRPSolver, save_filename: str = None,
                        reference_cost: float = None):
    """
    Plot the solver's best solution (routes) and its optimization history.

    Args:
        solver: a CVRPSolver that has already run solve_vns()
        save_filename: if given, save the figure to this path (PNG, 300 dpi)
        reference_cost: optional known-good cost to compare against
    """
    if solver.best_solution is None:
        print("No solution available for visualization!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left panel: the routes -------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, len(solver.best_solution)))

    for idx, route in enumerate(solver.best_solution):
        if not route:
            continue

        # Route path: depot -> customers in order -> depot
        x_coords = [solver.instance.depot[0]]
        y_coords = [solver.instance.depot[1]]

        route_demand = 0
        for cust_id in route:
            customer = solver.instance.customer_dict.get(cust_id)
            if customer:
                x_coords.append(customer['x'])
                y_coords.append(customer['y'])
                route_demand += customer['demand']

        x_coords.append(solver.instance.depot[0])
        y_coords.append(solver.instance.depot[1])

        ax1.plot(x_coords, y_coords, 'o-', color=colors[idx],
                 linewidth=2, markersize=6,
                 label=f'Route {idx + 1} (demand: {route_demand}/{solver.instance.capacity})')

    # Plot every customer with its id and demand as a label
    for customer in solver.instance.customers:
        ax1.plot(customer['x'], customer['y'], 'o', color='navy', markersize=8)
        ax1.text(customer['x'] + 1, customer['y'] + 1,
                 f"{customer['id']}\n({customer['demand']})",
                 fontsize=8, ha='left')

    # Depot marker
    ax1.plot(solver.instance.depot[0], solver.instance.depot[1], 's',
              color='red', markersize=15, label='Depot')

    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)

    title = f'CVRP Solution - Cost: {solver.best_cost:.2f}'
    if reference_cost:
        gap = ((solver.best_cost - reference_cost) / reference_cost) * 100
        title += f' (Gap: {gap:+.2f}%)'
    ax1.set_title(title, fontsize=14, fontweight='bold')

    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # --- Right panel: convergence curve ------------------------------
    ax2.plot(solver.history, linewidth=2, color='blue', label='VNS cost')

    if reference_cost:
        ax2.axhline(y=reference_cost, color='red', linestyle='--',
                     linewidth=2, label=f'Reference cost: {reference_cost:.2f}')

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Solution Cost', fontsize=12)
    ax2.set_title('VNS Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    improvement = ((solver.history[0] - solver.best_cost) / solver.history[0]) * 100
    stats_text = (f"Initial: {solver.history[0]:.2f}\n"
                  f"Final: {solver.best_cost:.2f}\n"
                  f"Improvement: {improvement:.1f}%")

    if reference_cost:
        gap = ((solver.best_cost - reference_cost) / reference_cost) * 100
        stats_text += f"\n\nReference: {reference_cost:.2f}\nGap: {gap:+.2f}%"

    ax2.text(0.65, 0.95, stats_text, transform=ax2.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_filename}")

    plt.show()
