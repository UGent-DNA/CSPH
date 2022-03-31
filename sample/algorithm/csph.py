# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

import heapq
import itertools
import os.path
from typing import Set, Dict

from sample.datastructures.linkedlist import Linkedlist
from sample.datastructures.tree import Tree
from sample.general import read_data
from sample.general.general_functions import *

Node = int


def get_edge_cost(e: Edge):
    """
    For use in Dijkstra and SPH
     Adapt to edge-length for your graph
     """
    return e.weight


def get_edge_cost_tree(e: Edge, result_tree: Tree):
    """
    For use CSPH: combines trenching and fiber cost.
    Make sure this function is the same as the calculate-cost function
    """
    trenching_cost = 0 if result_tree.contains_edge(e.source, e.target) else e.weight
    return e.length + trenching_cost


def shortest_path_simple(g: Graph, source: int, target: int) -> Dict[Node, Tuple[float, Node]]:
    return shortest_path(g, {source}, {target}, 1)


def shortest_path(g: Graph, sources: Set[int], targets: Set[int], number_targets_to_finish) \
        -> Dict[Node, Tuple[float, Node]]:
    """ Run a Dijkstra Search from a set of sources to all targets. Returns a tree in Dict-format. """
    c = itertools.count()
    heap: List[Tuple[float, int, Node]] = [(0, next(c), s) for s in sources]
    heapq.heapify(heap)
    costs: Dict[Node, Tuple[float, Node]] = {s: (0, None) for s in sources}
    found_targets = set()
    while heap:
        dist, _, node = heapq.heappop(heap)
        if dist > costs[node][0]:
            continue
        if node in targets:
            found_targets.add(node)
            if len(found_targets) >= number_targets_to_finish:
                break

        for neigh, e in g.get_neighbor_dict(node).items():
            prev_neigh_cost = costs[neigh][0] if neigh in costs else float('inf')
            new_cost = dist + get_edge_cost(e)
            if new_cost < prev_neigh_cost:
                costs[neigh] = (new_cost, node)
                heapq.heappush(heap, (new_cost, next(c), neigh))

    return costs


def prim_improved(g: Graph, terminals: Set[Node]) -> Dict[Node, Tuple[float, Node]]:
    """ Or SPH. Quickly finds a good Steiner tree. """
    t = set(terminals)
    c = itertools.count()
    root = find_best_root(g, terminals)
    heap: List[Tuple[float, int, Node]] = [(0, next(c), root)]
    # heapq.heapify(heap)
    upper_bounds: Dict[Node, Tuple[float, Node]] = {root: (0, None)}
    result_tree = Tree(root)

    for i in range(len(t)):
        while heap:
            dist, _, node = heapq.heappop(heap)
            if dist > upper_bounds[node][0]:
                continue

            if node in t:
                path_node = node
                while path_node is not None:
                    next_node = upper_bounds[path_node][1]
                    upper_bounds[path_node] = (0, next_node)
                    heapq.heappush(heap, (0, next(c), path_node))
                    result_tree.add_relation(path_node, next_node)
                    path_node = next_node
                t.remove(node)
                break

            for neigh, e in g.get_neighbor_dict(node).items():
                prev_neigh_cost = upper_bounds[neigh][0] if neigh in upper_bounds else float('inf')
                new_cost = dist + get_edge_cost(e)
                if new_cost < prev_neigh_cost:
                    upper_bounds[neigh] = (new_cost, node)
                    heapq.heappush(heap, (new_cost, next(c), neigh))

    return upper_bounds


n_resets = 0


def csph(g: Graph, terminals: Set[Node]) -> Tree:
    """
    The Capacitated Shortest Path Heuristic, capable of quickly finding a good solution to
    the Capacitated Steiner tree problem.
    """
    global n_resets
    n_resets = 0
    t = set(terminals)
    c = itertools.count()
    root = find_best_root(g, terminals)
    heap: List[Tuple[float, int, Node]] = [(0, next(c), root)]
    # heapq.heapify(heap)
    steiner_upper_bound: Dict[Node, float] = {root: 0}

    full_edges = set()

    algo_tree = Tree(root)
    result_tree = Tree(root)

    flow_on_edges: Dict[Edge, int] = {}
    consecutive_resets = 0
    quit_algorithm = False
    while not quit_algorithm:
        while True:
            if not heap or not t or consecutive_resets > 5:
                quit_algorithm = True
                break

            dist, _, node = heapq.heappop(heap)

            if dist != steiner_upper_bound.get(node, float("inf")):
                continue

            if node in t:
                demand = 1

                path_node = node
                node_path: List[Node] = algo_tree.get_path_from_root_to_node(path_node)
                edge_path: List[Edge] = get_edge_path(g, node_path)
                edges_over_capacity = [e for e in edge_path if flow_on_edges.get(e, 0) + demand > e.capacity]

                if edges_over_capacity:
                    for edge in edges_over_capacity:
                        full_edges.add(edge)
                        flow_on_edges[edge] = 0

                    n = algo_tree.get_farthest_from_root(edges_over_capacity[0].source, edges_over_capacity[0].target)
                    neighbours = set()
                    for n in algo_tree.remove_subtree(n, False):
                        del steiner_upper_bound[n]
                        neighbours.update(g.structure[n].keys())
                    for neigh in neighbours:
                        if neigh in algo_tree.parents.keys():
                            heapq.heappush(heap, (steiner_upper_bound.get(neigh, float("inf")), next(c), neigh))
                    consecutive_resets += 1
                    n_resets += 1
                    break
                else:
                    new_upperbound = 0
                    for e in edge_path:
                        flow_on_edges[e] = flow_on_edges.get(e, 0) + demand
                    for i, n in enumerate(node_path):
                        if i > 0:
                            result_tree.add_relation(n, node_path[i - 1])
                            algo_tree.add_relation(n, node_path[i - 1])
                            new_upperbound += get_edge_cost_tree(g.structure[n][node_path[i - 1]], result_tree)
                        steiner_upper_bound[n] = new_upperbound
                        heapq.heappush(heap, (new_upperbound, next(c), n))

                    consecutive_resets = 0
                    t.remove(node)
                break

            for neigh, e in g.get_neighbor_dict(node).items():
                nodes_in_solution = neigh in result_tree.get_nodes() and node in result_tree.get_nodes()
                if nodes_in_solution and not result_tree.contains_edge(node, neigh):
                    # Do not create loops in result_tree
                    continue
                if e in full_edges:
                    continue
                prev_neigh_cost = steiner_upper_bound.get(neigh, float('inf'))
                new_cost = dist + get_edge_cost_tree(e, result_tree)  # (9999999 if e in full_edges else 0)
                if new_cost < prev_neigh_cost:
                    steiner_upper_bound[neigh] = new_cost
                    heapq.heappush(heap, (new_cost, next(c), neigh))
                    algo_tree.add_relation(neigh, node)

    result_tree.prune(terminals)

    if consecutive_resets == 5:
        raise RuntimeError("Not all terminals were connected")
    return result_tree


def find_best_root(g: Graph, terminals: Set[Node]):
    """ Choose root based on capacities, if there are any, otherwise on position. """
    if g.node_mapping and "pos" in g.node_mapping.keys():
        return find_best_root_pos(g, terminals)
    else:
        # Get terminal with max edge throughput
        return max((sum(e.capacity for e in g.structure[t].values()), t) for t in terminals)[1]


def find_best_root_pos(g: Graph, terminals: Set[Node]):
    """ Find a good root based on position (central in relation to the terminals. """
    terminal_positions = {t: g.node_mapping[t]["pos"] for t in terminals}
    pos_x = 0
    pos_y = 0
    for pos in terminal_positions.values():
        pos_x += pos[0]
        pos_y += pos[1]

    average_pos = (pos_x / len(terminals), pos_y / len(terminals))
    terminal_closest = None
    closest_dist = float("inf")
    for t in terminals:
        t_dist = euclid_distance(terminal_positions[t], average_pos)
        if t_dist < closest_dist:
            terminal_closest = t
            closest_dist = t_dist

    root = terminal_closest
    best_score = closest_dist * sum([e.capacity for e in g.structure[root].values()])
    counter = 0
    to_process = Linkedlist([root])
    for _ in range(len(terminals)):
        node = to_process.pop_head()
        node_throughput = 0
        for neigh, e in g.structure[node].items():
            to_process.add_tail(neigh)
            node_throughput += e.capacity

        node_score = euclid_distance(g.node_mapping[node]["pos"], average_pos) * node_throughput
        if node_score > best_score:
            best_score = node_score
            root = node
        counter += 1

    return root


def tree_to_list_of_edges(g: Graph, tree: Tree, terminals: List[Node]) -> List[List[Edge]]:
    """ Given a tree input, return a list of edge-paths that connects the root to the terminals."""
    tree_nodes = tree.get_nodes()
    return [get_edge_path(g, tree.get_path_from_root_to_node(t)) for t in terminals if t in tree_nodes]


def calculate_cost(terminal_paths: List[List[Edge]]):
    """
    Calculate the cost of a solution.
    If possible, multiply fiber cost with edge length (this is not given for all Steinlib graphs)
    """
    flow_on_edge: Dict[Edge, int] = {}
    for terminal_path in terminal_paths:
        terminal_demand = 5
        for e in terminal_path:
            flow_on_edge[e] = flow_on_edge.get(e, 0) + terminal_demand

    fiber_cost = sum(flow_on_edge.values())
    trenching_cost = sum(e.weight for e in flow_on_edge.keys())

    return fiber_cost, trenching_cost


def get_terminal_paths(g: Graph, dijkstra_output: Dict[int, Tuple[float, int]], terminals: List[int]) \
        -> List[List[Edge]]:
    """ Given a dict-tree input, return a list of edge-paths that connects the root to the terminals."""
    terminal_paths = []
    for terminal in terminals:
        terminal_path = []
        path_node = terminal
        while True:
            next_node = dijkstra_output.get(path_node)[1]
            if next_node is None:
                break
            terminal_path.append(g.structure[path_node][next_node])
            path_node = next_node
        terminal_paths.append(terminal_path)
    return terminal_paths


def experiment_2():
    """ For the CSPH, run different capacity settings"""
    import random
    import time

    from definitions import OUTPUT_PATH

    for dataset_name in ["lin", "alue", "puc", "vienna"]:
        for g, _, sol, name in read_data.generate_steinlib_graphs(dataset_name):
            print(f"{dataset_name} {g.name}")
            edges = g.get_edges()

            for seed_mod in range(1, 11):
                seed = 999 + seed_mod * 13
                random.seed(seed)
                for n_terminal_ratio_percent in range(1, 26, 5):
                    n_terminals = int(len(g.structure.keys()) * n_terminal_ratio_percent / 100)
                    terminals = random.sample(g.structure.keys(), n_terminals)

                    min_base_cap = max(1, int(n_terminals / 10))
                    for base_capacity in range(min_base_cap, int(n_terminals / 3), min_base_cap):
                        for cap_structure in ["random", "levels"]:
                            if cap_structure == "random":
                                for e in edges:
                                    e.capacity = random.randint(1, base_capacity * 3)
                            elif cap_structure == "levels":
                                for e in edges:
                                    e.capacity = random.randint(1, 5) * base_capacity

                            start = time.time()
                            tree_csph = csph(g, set(terminals))
                            cost_csph = calculate_cost(tree_to_list_of_edges(g, tree_csph, terminals))
                            total_time = time.time() - start

                            with open(os.path.join(OUTPUT_PATH, "results.txt"), 'a+') as f:
                                e_caps = [e.capacity for e in edges]

                                results = {"nodes": len(g.structure.keys()),
                                           "terminals": f"{len(terminals):5d}",
                                           "connected_terminals": f"{len([t for t in terminals if t in tree_csph.get_nodes()]):6d}",
                                           "fiber": f"{cost_csph[0]:6d}",
                                           "trench": f"{cost_csph[1]:6d}",
                                           "average_capacity": f"{sum(e_caps) / len(e_caps) :8.4f}",
                                           "n_different_capacities": f"{len(set(e_caps)):6d}",
                                           "n_resets": f"{n_resets:6d}",
                                           "seed": f"{seed:6d}",
                                           "time": f"{total_time:10.6f}",
                                           "dataset_name": f"{dataset_name:10s}",
                                           "name": f"{g.name:15s}",
                                           "terminal_ratio": f"{n_terminal_ratio_percent:8.4f}",
                                           "base_cap": f"{base_capacity:6d}",
                                           "cap_structure": f"{cap_structure:10s}",
                                           "method": "csph"}

                                f.write('; '.join([f'{k}={v}' for k, v in results.items()]) + "\n")


def experiment_1():
    """ For Dijkstra and SPH (prim_improved), run experiments"""
    import random
    import time

    from definitions import OUTPUT_PATH

    for dataset_name in ["lin", "alue", "puc", "vienna"]:
        for g, _, sol, name in read_data.generate_steinlib_graphs(dataset_name):
            print(f"{dataset_name} {g.name}")
            edges = g.get_edges()

            for seed_mod in range(1, 11):
                seed = 999 + seed_mod * 13
                random.seed(seed)
                for n_terminal_ratio_percent in range(1, 26, 5):
                    n_terminals = max(1, int(len(g.structure.keys()) * n_terminal_ratio_percent / 100))
                    terminals = random.sample(g.structure.keys(), n_terminals)

                    for method in ["dijkstra", "prim_improved"]:
                        connected_terminals = len(terminals)

                        terminal_set = set(terminals)
                        if method == "dijkstra":
                            start = time.time()
                            paths = shortest_path(g, {find_best_root(g, terminal_set)}, terminal_set, len(terminals))
                            total_time = time.time() - start
                            cost = calculate_cost(get_terminal_paths(g, paths, terminals))
                        elif method == "prim_improved":
                            start = time.time()
                            pi_result = prim_improved(g, terminal_set)
                            total_time = time.time() - start
                            paths = get_terminal_paths(g, pi_result, terminals)
                            cost = calculate_cost(paths)
                        elif method == "csph":
                            start = time.time()
                            tree_csph = csph(g, terminal_set)
                            total_time = time.time() - start
                            cost = calculate_cost(tree_to_list_of_edges(g, tree_csph, terminals))
                            connected_terminals = len([t for t in terminals if t in tree_csph.get_nodes()])
                        else:
                            raise ValueError

                        with open(os.path.join(OUTPUT_PATH, "results.txt"), 'a+') as f:
                            e_caps = [e.capacity for e in edges]

                            results = {"nodes": len(g.structure.keys()),
                                       "terminals": f"{len(terminals):5d}",
                                       "connected_terminals": f"{connected_terminals:6d}",
                                       "fiber": f"{cost[0]:9d}",
                                       "trench": f"{cost[1]:9d}",
                                       "average_capacity": f"{sum(e_caps) / len(e_caps) :9.4f}",
                                       "n_different_capacities": f"{len(set(e_caps)):6d}",
                                       "n_resets": f"{n_resets:6d}",
                                       "seed": f"{seed:6d}",
                                       "time": f"{total_time:10.6f}",
                                       "dataset_name": f"{dataset_name:10s}",
                                       "name": f"{g.name:15s}",
                                       "method": f"{method:15s}"}

                            f.write('; '.join([f'{k}={v}' for k, v in results.items()]) + "\n")


def general_data():
    """ Generate information on the datasets. """
    nodes = []
    edges = []
    for g, _, sol, name in read_data.generate_steinlib_graphs("vienna"):
        if g.name[0] == "I":
            continue
        print(f"{g.name}")
        edges.append(len(g.get_edges()))
        nodes.append(len(g.structure.keys()))
    print("Vienna G")
    print(len(nodes))
    print(sum(nodes) / len(nodes))
    print(sum(edges) / len(edges))
    print(sum([e / n for n, e in zip(nodes, edges)]) / len(nodes))
    nodes = []
    edges = []
    for g, _, sol, name in read_data.generate_steinlib_graphs("vienna"):
        if g.name[0] == "G" or int(g.name[1:]) > 15:
            continue
        print(f"{g.name}")
        edges.append(len(g.get_edges()))
        nodes.append(len(g.structure.keys()))
    print("Vienna I")
    print(len(nodes))
    print(sum(nodes) / len(nodes))
    print(sum(edges) / len(edges))
    print(sum([e / n for n, e in zip(nodes, edges)]) / len(nodes))
    for dataset_name in ["lin", "puc", "alue"]:
        nodes = []
        edges = []
        for g, _, sol, name in read_data.generate_steinlib_graphs(dataset_name):
            print(f"{g.name}")
            edges.append(len(g.get_edges()))
            nodes.append(len(g.structure.keys()))
        print(dataset_name)
        print(len(nodes))
        print(sum(nodes) / len(nodes))
        print(sum(edges) / len(edges))
        print(sum([e / n for n, e in zip(nodes, edges)]) / len(nodes))


if __name__ == "__main__":
    general_data()

    # To run experiments, uncomment lines below (can take several hours)
    # experiment_1()
    # experiment_2()
