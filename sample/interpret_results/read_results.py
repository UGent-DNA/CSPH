# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

import dataclasses
import os
from dataclasses import dataclass
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from definitions import OUTPUT_PATH
from sample.general import data_util as du
from sample.general import visualize_util as vu

matplotlib.rcParams.update({'font.size': 14})
plasma = cm.get_cmap("plasma").reversed()
label_mappings = {"dijkstra": "Dijkstra", "csph": "CSPH", "prim_improved": "SPH"}


@dataclass
class Result:
    """ Dataclass to capture data from experiments. """
    nodes: int
    terminals: int
    connected_terminals: int
    fiber: int
    trench: int
    average_capacity: float
    n_different_capacities: int
    n_resets: int
    seed: int
    time: float
    dataset_name: str
    name: str
    method: str
    terminal_ratio: int = None
    base_cap: int = None
    cap_structure: str = None

    def combine_list(self, list_of_results):
        """
        Combine a list of {Result} instances by averaging the numerical values.
        The other values should be the same.
        """
        if not list_of_results:
            raise ValueError("input should not be empty")
        fields = dataclasses.fields(self)
        new_result = {}
        for field in fields:
            name = field.name
            vals_initial = [r.__getattribute__(name) for r in list_of_results]
            vals = [v for v in vals_initial if v is not None]
            if field.type is int:
                new_result[name] = round(sum(vals) / len(vals)) if vals else None
            elif field.type is float:
                new_result[name] = sum(vals) / len(vals) if vals else None
            else:
                if len(set(vals)) > 1:
                    raise ValueError("Only combine Results with the same strings!")
                else:
                    new_result[name] = vals[0] if vals else None

        return Result(**new_result)

    # def avg(self, r, f):
    #     return round((f(self) + f(r)) / 2)


def read_results() -> List[Result]:
    """ Read all the results from the experiments in {csph.py} and put them in a list of Result instances."""
    results_initial: List[Result] = []
    with open(os.path.join(OUTPUT_PATH, "results_all.txt"), "r") as f:
        for line in f:
            line_no_spaces = line.replace(" ", "").strip()
            params = {}
            for word in line_no_spaces.split(';'):
                key_val = word.split('=')
                key = key_val[0]
                val = key_val[1]
                if key in {"average_capacity", "time"}:
                    val = float(val)
                elif key in {"dataset_name", "name", "method", "cap_structure"}:
                    pass
                else:
                    val = int(float(val))
                params[key] = val

            result = Result(**params)
            results_initial.append(result)
    return results_initial


def results_to_seed_batches(list_of_results: List[Result]) -> List[List[Result]]:
    """
    Averages all CSPH results with the same running parameters (except different seed).
    Filters out incomplete solutions (terminals != connected_terminals.

    The batching is to facilitate comparison when filtered out incomplete solution force list lengths for different
    settings.

    :return: All results, in batches. Each batch (inner list) contains at most 10 Result instances
    (number of experiments with different seeds).
    """
    batch_bins = []
    counter = 0
    name_start = 0
    start_seed = list_of_results[name_start].seed
    for _ in range(139):
        batch_size = 0
        while list_of_results[name_start + batch_size].seed == start_seed:
            batch_size += 1

        print(name_start, batch_size)
        j = name_start
        while list_of_results[j].seed == start_seed:
            res_bin = []
            for i in range(10):
                result = list_of_results[j + i * batch_size]
                # Remove incomplete CSPH results!
                if result.terminals != result.connected_terminals:
                    continue
                res_bin.append(result)
            # res_bins.append(res_bin[0].combine_list(res_bin))
            if not res_bin:
                print("**", list_of_results[j + 9 * batch_size])
                counter += 1
            else:
                batch_bins.append(res_bin)
                # print(res_bin[0].combine_list(res_bin))

            j += 1

        name_start += 10 * batch_size
    return batch_bins


def make_capacity_plot(name="G101", n_terminals=644, reset_divider=40):
    r_graph = [r for r in seed_batched_results if
               r[0].name == name and r[0].method == "csph" and r[0].terminals == n_terminals]
    r_graph = sorted(r_graph, key=lambda res: du.mean([r.average_capacity for r in res]))

    def element_mean(list_of_results, attribute, transform):
        return [transform([attribute(r) for r in res]) for res in list_of_results]

    combined_graph = [r[0].combine_list(r) for r in r_graph]

    vu.bar_plot_double(element_mean(r_graph, lambda r: r.time, du.mean),
                       element_mean(r_graph, lambda r: r.time, du.standard_deviation),
                       element_mean(r_graph, lambda r: r.n_resets / reset_divider, du.mean),
                       element_mean(r_graph, lambda r: r.n_resets / reset_divider, du.standard_deviation),
                       ylab=f"Time (s) and |resets| (x{reset_divider})",
                       xlab="Capacity structure and average capacity",
                       xtick_lab=[f"{r.cap_structure}\n{round(r.average_capacity):5d}" for r in combined_graph],
                       save_file=f"time_reset_variance_{name}",
                       title=f"Graph {name} with {n_terminals} terminals",
                       legend=["Time", "Number of resets"])


def generate_vienna_by_terminalratios_plot():
    r_vienna_I = [r for r in results if r.dataset_name == "vienna" and r.name[0] == "I" and r.method == "csph"]
    r_vienna_G = [r for r in results if r.dataset_name == "vienna" and r.name[0] == "G" and r.method == "csph"]
    legend_mapping = lambda i: f"{i}%"

    vu.get_attribute_plot_specific(r_vienna_I, lambda r: r.nodes / 1000, lambda r: r.terminal_ratio,
                                   lambda r: r.time, legend_mapping, "Terminal ratio",
                                   xlabel="Number of nodes (x1000)", ylabel="Time (s)",
                                   title="Vienna I",
                                   savename="time_by_node_sep_t_vI", plot_average=True)

    vu.get_attribute_plot_specific(r_vienna_G, lambda r: r.nodes / 1000, lambda r: r.terminal_ratio,
                                   lambda r: r.time, legend_mapping, "Terminal ratio",
                                   xlabel="Number of nodes  (x1000)", ylabel="Time (s)",
                                   title="Vienna G",
                                   savename="time_by_node_sep_t_vG", plot_average=True)

    # legend: ["1%","5%","10%","15%","20%","25%"], title="Terminal ratios"


def time_evolution_separated_by_method():
    for dataset_name in ["lin", "alue", "puc"]:
        vu.get_attribute_plot_scatter_line([r for r in results if r.dataset_name == dataset_name],
                                           lambda r: r.nodes / 1000,
                                           lambda r: r.method, lambda r: r.time, xlabel="Number of nodes (x1000)",
                                           ylabel="Time (s)", label_mapping=label_mappings, title=f"{dataset_name}",
                                           savename=f"time_by_node_sep_m_{dataset_name}",
                                           quadratic_factor=200 if dataset_name == "lin" else 100)

    vu.get_attribute_plot_scatter_line([r for r in results if r.dataset_name == "vienna" and r.name[0] == "I"],
                                       lambda r: r.nodes / 1000,
                                       lambda r: r.method, lambda r: r.time, xlabel="Number of nodes (x1000)",
                                       ylabel="Time (s)", label_mapping=label_mappings, title=f"Vienna I",
                                       savename=f"time_by_node_sep_m_vI", quadratic_factor=25)
    vu.get_attribute_plot_scatter_line([r for r in results if r.dataset_name == "vienna" and r.name[0] == "G"],
                                       lambda r: r.nodes / 1000,
                                       lambda r: r.method, lambda r: r.time, xlabel="Number of nodes (x1000)",
                                       ylabel="Time (s)", label_mapping=label_mappings, title=f"Vienna G",
                                       savename=f"time_by_node_sep_m_vG")


def get_cost_type_plot():
    for dataset_name in ["lin", "alue", "puc"]:
        r_data_G = [r for r in results if
                    r.dataset_name == dataset_name and (r.method != "csph" or r.n_resets < 10)]
        _get_cost_type_plot(r_data_G)
        plt.title(f"{dataset_name}")

        vu.save_current_plot(f"cost_type_{dataset_name}")

    graph_name = "lin37"

    r_data_G = [r for r in results if r.name == graph_name and (r.method != "csph" or r.n_resets < 10)]
    _get_cost_type_plot(r_data_G)
    plt.title(f"{graph_name}")
    vu.save_current_plot(f"cost_type_{graph_name}")

    r_vienna_G = [r for r in results if
                  r.dataset_name == "vienna" and r.name[0] == "G"]
    _get_cost_type_plot(r_vienna_G)
    plt.title("Vienna G")
    vu.save_current_plot("cost_type_vG")

    r_vienna_I = [r for r in results if
                  r.dataset_name == "vienna" and r.name[0] == "I"]
    _get_cost_type_plot(r_vienna_I)
    plt.title("Vienna I")
    vu.save_current_plot("cost_type_vI")


def _get_cost_type_plot(r_data_G):
    cost_types = {"fiber": lambda r: r.fiber, "trench": lambda r: r.trench, "total": lambda r: r.fiber + r.trench}

    res_by_method = du.dict_by_attribute(r_data_G, lambda r: r.method)
    cost_means = {k: [] for k in cost_types.keys()}

    res = res_by_method["csph"]
    valid_name_terminal_combinations = {}
    for cost_type in cost_types.keys():
        res_by_name = du.dict_by_attribute(res, lambda r: r.name)
        means = []
        for c_name, list_of_r in res_by_name.items():
            res_by_terminals = du.dict_by_attribute(list_of_r, lambda r: r.terminals)
            valid_name_terminal_combinations[c_name] = set(res_by_terminals.keys())
            means.append(
                du.mean([du.mean([cost_types[cost_type](r) for r in l_of_r]) for l_of_r in res_by_terminals.values()]))
        total_mean = du.mean(means)
        cost_means[cost_type].append(total_mean)

    methods = ["dijkstra", "prim_improved"]
    for method in methods:
        res = res_by_method[method]
        for cost_type in cost_types.keys():
            res_by_name = du.dict_by_attribute(res, lambda r: r.name)
            means = []
            for c_name, list_of_r in res_by_name.items():
                res_by_terminals = du.dict_by_attribute(list_of_r, lambda r: r.terminals)
                for terminal in list(res_by_terminals.keys()):
                    if terminal not in valid_name_terminal_combinations[c_name]:
                        del res_by_terminals[terminal]
                means.append(du.mean(
                    [du.mean([cost_types[cost_type](r) for r in l_of_r]) for l_of_r in res_by_terminals.values()]))
            total_mean = du.mean(means)
            cost_means[cost_type].append(total_mean)
    color_counter = 0
    methods = ["CSPH", "Dijkstra", "SPH"]
    for cost_type in cost_types.keys():
        plt.scatter(methods,
                    cost_means[cost_type],
                    label=f"{cost_type}",
                    color=plasma.colors[50 + color_counter * int(200 / len(cost_types))])

        plt.plot(methods,
                 cost_means[cost_type],
                 color=plasma.colors[50 + color_counter * int(200 / len(cost_types))])
        color_counter += 1
    plt.legend(title="Cost type")
    plt.ylabel("Average cost")
    plt.xticks([0, 1, 2], labels=methods)


if __name__ == "__main__":
    input_results = read_results()
    seed_batched_results = results_to_seed_batches(input_results)

    # Average the batched results and add Dijkstra and SPH results
    results = [b[0].combine_list(b) for b in seed_batched_results if b] + [r for r in input_results if
                                                                           r.method != "csph"]

    make_capacity_plot("G101", 644, 40)
    make_capacity_plot("G307", 7830, 5)
    make_capacity_plot("I003", 976, 40)

    generate_vienna_by_terminalratios_plot()

    time_evolution_separated_by_method()

    get_cost_type_plot()
