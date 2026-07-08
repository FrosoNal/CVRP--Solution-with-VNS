"""
cvrp_solver.py

Defines the CVRPSolver class: the optimization engine for a CVRPInstance.

Contains:
    - Basic geometry / cost helpers (distance, route cost, solution cost)
    - Feasibility check (vehicle capacity constraint)
    - A greedy Nearest Neighbor construction heuristic
    - Three local search "move" operators: swap, relocate, two_opt
    - A local search loop that repeatedly applies random operators
    - A Variable Neighborhood Search (VNS) metaheuristic that ties
      everything together to iteratively improve a solution
    - I/O helpers to load an initial solution and save the best one
"""

import copy
import random
import time
from typing import List, Tuple, Dict

import numpy as np

from cvrp_instance import CVRPInstance


class CVRPSolver:
    """Optimizes routes for a given CVRPInstance using VNS."""

    def __init__(self, instance: CVRPInstance):
        self.instance = instance
        self.best_solution = None
        self.best_cost = float('inf')
        self.history = []  # best cost recorded at each VNS iteration

    # ------------------------------------------------------------------
    # Cost / feasibility helpers
    # ------------------------------------------------------------------

    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def route_cost(self, route: List[int]) -> float:
        """Total travel distance of a single route, depot -> customers -> depot."""
        if not route:
            return 0

        cost = 0
        prev = self.instance.depot

        for cust_id in route:
            customer = self.instance.customer_dict.get(cust_id)
            if customer is None:
                print(f"WARNING: Customer {cust_id} not found!")
                continue
            curr = (customer['x'], customer['y'])
            cost += self.distance(prev, curr)
            prev = curr

        cost += self.distance(prev, self.instance.depot)
        return cost

    def solution_cost(self, routes: List[List[int]]) -> float:
        """Total cost of a full solution (sum of all route costs)."""
        return sum(self.route_cost(route) for route in routes)

    def is_feasible(self, route: List[int]) -> bool:
        """A route is feasible if its total demand doesn't exceed capacity."""
        total_demand = 0
        for cid in route:
            customer = self.instance.customer_dict.get(cid)
            if customer:
                total_demand += customer['demand']
        return total_demand <= self.instance.capacity

    # ------------------------------------------------------------------
    # Construction heuristic
    # ------------------------------------------------------------------

    def nearest_neighbor_solution(self) -> List[List[int]]:
        """
        Build an initial solution greedily: repeatedly start a new route
        from the depot and keep adding the closest feasible unvisited
        customer until no more customers fit, then start a new route.
        """
        unvisited = set(c['id'] for c in self.instance.customers)
        routes = []

        while unvisited:
            route = []
            current = self.instance.depot
            remaining_capacity = self.instance.capacity

            while unvisited:
                best_customer = None
                best_distance = float('inf')

                for cust_id in unvisited:
                    customer = self.instance.customer_dict[cust_id]
                    demand = customer['demand']

                    if demand <= remaining_capacity:
                        pos = (customer['x'], customer['y'])
                        dist = self.distance(current, pos)

                        if dist < best_distance:
                            best_distance = dist
                            best_customer = cust_id

                if best_customer is None:
                    # No remaining customer fits in this vehicle -> close the route
                    break

                route.append(best_customer)
                customer = self.instance.customer_dict[best_customer]
                remaining_capacity -= customer['demand']
                current = (customer['x'], customer['y'])
                unvisited.remove(best_customer)

            if route:
                routes.append(route)

        return routes

    # ------------------------------------------------------------------
    # Local search (neighborhood) operators
    # ------------------------------------------------------------------

    def swap(self, routes: List[List[int]]) -> List[List[int]]:
        """Swap one random customer between two random routes (if feasible)."""
        new_routes = copy.deepcopy(routes)

        if len(new_routes) < 2:
            return new_routes

        r1, r2 = random.sample(range(len(new_routes)), 2)

        if not new_routes[r1] or not new_routes[r2]:
            return new_routes

        i1 = random.randint(0, len(new_routes[r1]) - 1)
        i2 = random.randint(0, len(new_routes[r2]) - 1)

        new_routes[r1][i1], new_routes[r2][i2] = new_routes[r2][i2], new_routes[r1][i1]

        if self.is_feasible(new_routes[r1]) and self.is_feasible(new_routes[r2]):
            return new_routes

        # Reject the move if it breaks capacity constraints
        return routes

    def relocate(self, routes: List[List[int]]) -> List[List[int]]:
        """Move one random customer from one route to another (if feasible)."""
        new_routes = copy.deepcopy(routes)

        if len(new_routes) < 2:
            return new_routes

        r1 = random.randint(0, len(new_routes) - 1)
        if not new_routes[r1]:
            return new_routes

        i1 = random.randint(0, len(new_routes[r1]) - 1)
        customer = new_routes[r1].pop(i1)

        r2 = random.randint(0, len(new_routes) - 1)
        i2 = random.randint(0, len(new_routes[r2]))
        new_routes[r2].insert(i2, customer)

        # Drop any route that became empty
        new_routes = [r for r in new_routes if r]

        if all(self.is_feasible(r) for r in new_routes):
            return new_routes

        return routes

    def two_opt(self, routes: List[List[int]]) -> List[List[int]]:
        """Reverse a random segment within a single random route (classic 2-opt)."""
        new_routes = copy.deepcopy(routes)

        if not new_routes:
            return new_routes

        r = random.randint(0, len(new_routes) - 1)
        if len(new_routes[r]) < 2:
            return new_routes

        i, j = sorted(random.sample(range(len(new_routes[r])), 2))

        new_routes[r][i:j + 1] = reversed(new_routes[r][i:j + 1])

        return new_routes

    def local_search(self, routes: List[List[int]], max_iterations: int = 50) -> List[List[int]]:
        """
        Simple randomized local search: repeatedly try a random operator
        (swap/relocate/two_opt) and keep it only if it improves the cost.
        Stops early if no improving move is found in a round.
        """
        current = routes
        current_cost = self.solution_cost(current)

        operators = [self.swap, self.relocate, self.two_opt]

        for _ in range(max_iterations):
            improved = False

            for _ in range(10):
                operator = random.choice(operators)
                neighbor = operator(current)
                neighbor_cost = self.solution_cost(neighbor)

                if neighbor_cost < current_cost:
                    current = neighbor
                    current_cost = neighbor_cost
                    improved = True
                    break

            if not improved:
                break

        return current

    # ------------------------------------------------------------------
    # Solution I/O
    # ------------------------------------------------------------------

    def load_initial_solution(self, sol_filename: str):
        """Load a starting solution from a .sol file and verify its cost."""
        routes, cost = CVRPInstance.load_solution_from_sol(sol_filename)
        print(f"\nLoaded solution from {sol_filename}")
        print(f"  File cost: {cost}")
        print(f"  Routes: {len(routes)}")

        # Cross-check the cost stated in the file against our own calculation
        calculated_cost = self.solution_cost(routes)
        print(f"  Calculated cost: {calculated_cost:.2f}")

        if cost and abs(calculated_cost - cost) > 0.01:
            print(f"  WARNING: cost mismatch: {abs(calculated_cost - cost):.2f}")

        return routes, cost if cost else calculated_cost

    # ------------------------------------------------------------------
    # Main metaheuristic: Variable Neighborhood Search (VNS)
    # ------------------------------------------------------------------

    def solve_vns(self, max_iterations: int = 1000, k_max: int = 3,
                  max_no_improve: int = 100, verbose: bool = True,
                  initial_solution: List[List[int]] = None,
                  reference_cost: float = None):
        """
        Run Variable Neighborhood Search:
            - Start from an initial solution (given or built via nearest
              neighbor).
            - At each iteration, try increasingly disruptive random
              perturbations (k = 1..k_max operator applications), refine
              each candidate with local_search, and accept it if it
              improves on the current best (reset k on success, else
              increase k).
            - Stop after max_iterations or after max_no_improve
              iterations without any improvement.
        """
        print(f"\n{'=' * 60}")
        print(f"Running VNS for {self.instance.name}")
        print(f"Customers: {self.instance.num_customers}, Capacity: {self.instance.capacity}")
        if reference_cost:
            print(f"Reference cost (.sol): {reference_cost:.2f}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        # Build or reuse the initial solution
        if initial_solution is not None:
            best = initial_solution
            print("Using provided initial solution")
        else:
            best = self.nearest_neighbor_solution()
            print("Building initial solution (Nearest Neighbor)")

        best_cost = self.solution_cost(best)

        print(f"  Cost: {best_cost:.2f}")
        print(f"  Routes: {len(best)}")

        if reference_cost:
            gap = ((best_cost - reference_cost) / reference_cost) * 100
            print(f"  Gap vs reference: {gap:+.2f}%")
        print()

        self.best_solution = best
        self.best_cost = best_cost
        self.history = [best_cost]

        no_improve = 0
        operators = [self.swap, self.relocate, self.two_opt]

        for iteration in range(max_iterations):
            k = 0

            while k < k_max:
                # Perturb the current best with (k + 1) random operator applications
                current = copy.deepcopy(best)
                for _ in range(k + 1):
                    current = random.choice(operators)(current)

                # Refine the perturbed solution with local search
                current = self.local_search(current)
                current_cost = self.solution_cost(current)

                if current_cost < best_cost:
                    # Improvement found: accept it and restart the neighborhood scan
                    best = current
                    best_cost = current_cost
                    k = 0
                    no_improve = 0

                    if verbose and iteration % 50 == 0:
                        gap_str = ""
                        if reference_cost:
                            gap = ((best_cost - reference_cost) / reference_cost) * 100
                            gap_str = f" (Gap: {gap:+.2f}%)"
                        print(f"Iteration {iteration}: New best solution = {best_cost:.2f}{gap_str}")
                else:
                    # No improvement at this neighborhood size: widen the search
                    k += 1

            no_improve += 1
            self.history.append(best_cost)

            if no_improve >= max_no_improve:
                if verbose:
                    print(f"\nStopping (no improvement for {max_no_improve} iterations)")
                break

        self.best_solution = best
        self.best_cost = best_cost

        elapsed_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"FINAL RESULTS")
        print(f"{'=' * 60}")
        print(f"VNS cost: {best_cost:.2f}")

        if reference_cost:
            gap = ((best_cost - reference_cost) / reference_cost) * 100
            diff = best_cost - reference_cost
            print(f"Reference cost (.sol): {reference_cost:.2f}")
            print(f"Difference: {diff:+.2f}")
            print(f"Gap: {gap:+.2f}%")

            if best_cost < reference_cost:
                print(f"Found a better solution!")
            elif abs(gap) < 1:
                print(f"Very close to the reference solution!")
            elif abs(gap) < 5:
                print(f"Reasonably close to the reference solution")
            else:
                print(f"Needs improvement")

        print(f"Routes: {len(best)}")
        print(f"Iterations: {iteration + 1}")
        print(f"Time: {elapsed_time:.2f}s")
        print(f"{'=' * 60}\n")

        return best, best_cost

    def save_solution_to_sol(self, filename: str):
        """Write the best solution found so far to a .sol file."""
        if self.best_solution is None:
            print("No solution available to save!")
            return

        with open(filename, 'w') as f:
            for idx, route in enumerate(self.best_solution):
                route_str = ' '.join(map(str, route))
                f.write(f"Route #{idx + 1}: {route_str}\n")
            f.write(f"Cost {int(self.best_cost)}\n")

        print(f"Solution saved to: {filename}")

    def print_solution_details(self):
        """Print a per-route breakdown (customers, demand, cost) of the best solution."""
        if self.best_solution is None:
            print("No solution available!")
            return

        print(f"\n{'=' * 60}")
        print(f"SOLUTION DETAILS")
        print(f"{'=' * 60}\n")

        for idx, route in enumerate(self.best_solution):
            route_demand = 0
            for cid in route:
                customer = self.instance.customer_dict.get(cid)
                if customer:
                    route_demand += customer['demand']

            route_cost = self.route_cost(route)

            print(f"Route {idx + 1}:")
            print(f"  Customers: {' -> '.join(map(str, route))}")
            print(f"  Demand: {route_demand}/{self.instance.capacity}")
            print(f"  Cost: {route_cost:.2f}")
            print()
