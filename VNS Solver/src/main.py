"""
main.py

Entry point: loads a CVRP instance (and optionally a reference .sol
solution), runs the VNS solver, prints/saves the results, and
generates a visualization.
"""

import os

from cvrp_instance import CVRPInstance
from cvrp_solver import CVRPSolver
from cvrp_visualization import visualize_solution


def main():
    print("=" * 60)
    print("CVRP SOLVER - supports .txt, .vrp, .sol")
    print("=" * 60)

    # Example: load from a .vrp file
    instance_file = "eil23.vrp"  # or "instance.txt"
    solution_file = "instance1.sol"  # optional reference solution

    print(f"\nLoading instance from: {instance_file}")
    instance = CVRPInstance.load_from_file(instance_file)

    solver = CVRPSolver(instance)

    # Optionally load a starting solution from a .sol file
    initial_solution = None
    reference_cost = None

    if os.path.exists(solution_file):
        try:
            initial_solution, reference_cost = solver.load_initial_solution(solution_file)
        except Exception as e:
            print(f"Error loading solution: {e}")
            initial_solution = None
            reference_cost = None

    # Run VNS
    solution, cost = solver.solve_vns(
        max_iterations=1000,
        k_max=3,
        max_no_improve=100,
        verbose=True,
        initial_solution=initial_solution,
        reference_cost=reference_cost
    )

    solver.print_solution_details()

    # Save the solution
    solver.save_solution_to_sol('output_solution.sol')

    # Visualize, comparing against the reference cost if available
    visualize_solution(solver, save_filename='cvrp_solution.png',
                        reference_cost=reference_cost)


if __name__ == "__main__":
    main()
