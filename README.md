# CVRP Solver: Variable Neighborhood Search (VNS)

## Description
This project implements a metaheuristic solver for the **Capacitated Vehicle Routing Problem (CVRP)**. The solver utilizes the **Variable Neighborhood Search (VNS)** algorithm to find optimized vehicle routes that satisfy capacity constraints while minimizing total distance.

## Features
- **Instance Support:** Loads problem instances from `.vrp` and `.txt` files .
- **Heuristic Optimization:** Implements VNS with local search operators, including `swap`, `relocate`, and `two-opt`.
- **Initial Solution:** Provides a construction heuristic using the **Nearest Neighbor** algorithm.
- **Verification & Benchmarking:** Supports loading `.sol` files to compare the solver's output against best-known solutions .
- **Visualization:** Generates high-quality route maps and convergence plots (using `matplotlib`) to visualize the solution cost improvement over iterations.

## Implementation Details
- **Local Search:** Combines multiple neighborhood operators to escape local optima.
- **VNS Strategy:** Systematically explores neighborhood structures ($k_{max}=3$) to improve the incumbent solution.
- **Feasibility:** Ensures all routes respect vehicle capacity limits.

## How to Run
1. Ensure the required libraries are installed:
   ```bash
   pip install numpy matplotlib
