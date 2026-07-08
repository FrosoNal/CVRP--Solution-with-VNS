"""
cvrp_instance.py

Defines the CVRPInstance class, which represents a Capacitated Vehicle
Routing Problem instance: the depot location, the list of customers
(with coordinates and demand), the vehicle capacity, and the number
of vehicles.

Also provides static loader methods to read instances from different
file formats:
    - .txt  : a simple custom text format
    - .vrp  : the standard CVRPLIB format
    - .sol  : a solution file (routes + cost), used to load a reference
              solution for comparison
"""

import os
from typing import List, Tuple, Dict


class CVRPInstance:
    """Container for a single CVRP problem instance."""

    def __init__(self, name: str, depot: Tuple[float, float],
                 customers: List[Dict], capacity: int, num_vehicles: int):
        self.name = name
        self.depot = depot
        self.customers = customers
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.num_customers = len(customers)

        # Quick lookup: customer id -> customer dict (id, x, y, demand)
        self.customer_dict = {c['id']: c for c in customers}

    @staticmethod
    def load_from_file(filename: str):
        """Dispatch to the correct loader based on file extension."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == '.vrp':
            return CVRPInstance.load_from_vrp(filename)
        elif ext == '.txt':
            return CVRPInstance.load_from_txt(filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def load_from_txt(filename: str):
        """
        Load an instance from a simple custom .txt format:
            DEPOT <x> <y>
            CAPACITY <cap>
            <id> <x> <y> <demand>
            ...
        Lines starting with '#' are treated as comments and skipped.
        """
        depot = None
        customers = []
        capacity = None

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("DEPOT"):
                    _, x, y = line.split()
                    depot = (float(x), float(y))

                elif line.startswith("CAPACITY"):
                    _, cap = line.split()
                    capacity = int(cap)

                elif line[0].isdigit():
                    cid, x, y, demand = line.split()
                    customers.append({
                        "id": int(cid),
                        "x": float(x),
                        "y": float(y),
                        "demand": int(demand)
                    })

        # Heuristic default: roughly one vehicle per 5 customers
        num_vehicles = max(1, len(customers) // 5)
        return CVRPInstance(filename, depot, customers, capacity, num_vehicles)

    @staticmethod
    def load_from_vrp(filename: str):
        """
        Load an instance from a standard CVRPLIB .vrp file.
        Parses the DIMENSION / CAPACITY header fields plus the
        NODE_COORD_SECTION, DEMAND_SECTION and DEPOT_SECTION blocks.
        """
        dimension = None
        capacity = None
        node_coords = {}
        demands = {}
        depot_id = None

        section = None  # tracks which section of the file we're in

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("EOF"):
                    continue

                # Header fields
                if "DIMENSION" in line:
                    dimension = int(line.split(":")[1].strip())
                elif "CAPACITY" in line:
                    capacity = int(line.split(":")[1].strip())

                # Section markers
                elif "NODE_COORD_SECTION" in line:
                    section = "coords"
                    continue
                elif "DEMAND_SECTION" in line:
                    section = "demands"
                    continue
                elif "DEPOT_SECTION" in line:
                    section = "depot"
                    continue

                # Section data
                if section == "coords":
                    parts = line.split()
                    if len(parts) >= 3:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        node_coords[node_id] = (x, y)

                elif section == "demands":
                    parts = line.split()
                    if len(parts) >= 2:
                        node_id = int(parts[0])
                        demand = int(parts[1])
                        demands[node_id] = demand

                elif section == "depot":
                    if line != "-1":
                        depot_id = int(line)

        # Build the instance from the parsed sections
        if depot_id is None:
            depot_id = 1  # Default depot if none was specified

        depot = node_coords[depot_id]
        customers = []

        for node_id in sorted(node_coords.keys()):
            if node_id == depot_id:
                continue

            x, y = node_coords[node_id]
            demand = demands.get(node_id, 0)

            customers.append({
                "id": node_id,
                "x": x,
                "y": y,
                "demand": demand
            })

        num_vehicles = max(1, len(customers) // 5)

        return CVRPInstance(filename, depot, customers, capacity, num_vehicles)

    @staticmethod
    def load_solution_from_sol(filename: str):
        """Load a solution (list of routes + cost) from a .sol file."""
        routes = []
        cost = None

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("Route"):
                    # Example line: "Route #1: 20 54 62 87 72 ..."
                    parts = line.split(":")
                    if len(parts) >= 2:
                        route_str = parts[1].strip()
                        route = [int(x) for x in route_str.split()]
                        routes.append(route)

                elif line.startswith("Cost"):
                    # Example line: "Cost 13332"
                    parts = line.split()
                    if len(parts) >= 2:
                        cost = float(parts[1])

        return routes, cost
