# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

import os
from typing import List, TypeVar, Callable, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

from definitions import OUTPUT_PATH
from sample.general.data_util import possible_vals, mean, dict_by_attribute, smooth, combine_over_att

T = TypeVar('T')
V = TypeVar('V')

matplotlib.rcParams.update({'font.size': 14})
plasma = cm.get_cmap("plasma").reversed()


def save_current_plot(name):
    """ Save the current plot to a png-format."""
    plt.tight_layout()
    path = f'{OUTPUT_PATH}/{name}.png'
    dirname = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    plt.savefig(path, format='png', dpi=400)
    plt.close()


def get_heatmap(custom_dataclass_list: List[T], row_attribute: Callable[[T], V],
                col_attribute: Callable[[T], V],
                result_func: Callable[[T], float],
                xlabel: str,
                ylabel: str,
                result_label: str,
                savename: str):
    row = possible_vals(custom_dataclass_list, row_attribute)
    column = possible_vals(custom_dataclass_list, col_attribute)
    mat = np.zeros((len(row), len(column)))
    for i, r in enumerate(row):
        for j, c in enumerate(column):
            mat[i][j] = mean(
                [result_func(t) for t in custom_dataclass_list if col_attribute(t) == c and row_attribute(t) == r])

    ax = sns.heatmap(mat,
                     linewidth=1,
                     xticklabels=column,
                     yticklabels=row,
                     cbar_kws={'label': result_label},
                     cmap=plasma)
    plt.yticks(rotation=0)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    save_current_plot(f"{savename}_heatmap")


def get_attribute_plot_specific(custom_dataclass_list: List[T], x_attribute: Callable[[T], V],
                       separator_attribute: Callable[[T], V],
                       result_attribute: Callable[[T], float],
                       legend_mapping,
                       legend_title,
                       xlabel='x',
                       ylabel='y', title="",
                       savename="trial",
                       plot_average=False,
                       smooth_factor=3):
    times_terminal = dict_by_attribute(custom_dataclass_list, separator_attribute)

    color_counter = 0
    if None in times_terminal.keys():
        print()
    for term in sorted(times_terminal.keys()):
        times = times_terminal[term]
        plt.plot(possible_vals(times, x_attribute),
                    smooth(combine_over_att(times, x_attribute, result_attribute), smooth_factor),
                    label=f"{legend_mapping(term)}",
                    color=plasma.colors[50 + color_counter * int(200 / len(times_terminal))])
        color_counter += 1

    if plot_average:
        plt.plot(possible_vals(custom_dataclass_list, x_attribute),
                 smooth(combine_over_att(custom_dataclass_list, x_attribute, result_attribute), 3),
                 color=(0.25390625, 0.0, 0.578125),
                 label="Average",
                 linestyle=':')

    plt.legend(title=legend_title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    # plt.xticks([0, 4000, 8000, 12000, 16000], [0, 4000, 8000, 12000, 16000])
    save_current_plot(savename)
    # plt.show()


def get_plot(custom_dataclass_list: List[T], x_attribute: Callable[[T], V],
             result_attribute: Callable[[T], float],
             xlabel='x',
             ylabel='y',
             savename="trial", smooth_factor=3):
    plt.plot(possible_vals(custom_dataclass_list, x_attribute), smooth(
        combine_over_att(custom_dataclass_list, x_attribute, result_attribute), smooth_factor))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_current_plot(savename)


def bar_plot_double(means, std, means2, std2, ylab, xlab, xtick_lab, title, save_file, legend):
    ind = np.arange(len(means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind - width / 2, means, width, color=plasma.colors[200], label=legend[0], yerr=std)
    ax.bar(ind + width / 2, means2, width, color=plasma.colors[50], label=legend[1], yerr=std2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(xtick_lab)
    ax.set_xlabel(xlab)
    # ax.set_ylim((0,))
    # ax.set_yticks(list(range(1, 5)))
    ax.legend()

    # fig.set_size_inches(7, 5)
    save_current_plot(save_file)


def get_attribute_plot_scatter_line(custom_dataclass_list: List[T], x_attribute: Callable[[T], V],
                                    separator_attribute: Callable[[T], V],
                                    result_attribute: Callable[[T], float],
                                    label_mapping,
                                    xlabel='x',
                                    ylabel='y',
                                    title="",
                                    savename="trial",
                                    plot_average=False,
                                    quadratic_factor= 100,
                                    smooth_factor=3):
    times_terminal = dict_by_attribute(custom_dataclass_list, separator_attribute)

    color_counter = 0
    for term in sorted(times_terminal.keys()):
        times = times_terminal[term]
        plt.scatter(possible_vals(times, x_attribute),
                    smooth(combine_over_att(times, x_attribute, result_attribute), 0),
                    label=f"{label_mapping[term]}",
                    color=plasma.colors[50 + color_counter * int(200 / len(times_terminal))])

        binned_x_vals = bin_list(possible_vals(times, x_attribute))
        # reversed_bins = {x_val: mean(vals) for vals in binned_x_vals for x_val in vals}
        smoothed_y_vals = smooth(combine_over_att(times, lambda r: x_attribute(r), result_attribute), 0)
        binned_y_vals = copy_bin(binned_x_vals, smoothed_y_vals)

        plt.plot([mean(vals) for vals in binned_x_vals],
                 [mean(vals) for vals in binned_y_vals],
                 color=plasma.colors[50 + color_counter * int(200 / len(times_terminal))])
        color_counter += 1

    if plot_average:
        plt.plot(possible_vals(custom_dataclass_list, x_attribute),
                 smooth(combine_over_att(custom_dataclass_list, x_attribute, result_attribute), 3),
                 color=(0.25390625, 0.0, 0.578125),
                 label="Average",
                 linestyle=':')

    vals = [i for i in range(1+int(max(possible_vals(custom_dataclass_list, x_attribute))))]
    plt.plot(vals,
             [v**2/quadratic_factor for v in vals],
             color=(0.25390625, 0.0, 0.578125),
             label=r'$t=\frac{{{}^2}}{{{:.1e}}}$'.format("n", 1000000*quadratic_factor),
             linestyle=':')

    plt.plot(vals,
             [10*v / quadratic_factor for v in vals],
             color=(0.25390625, 0.0, 0.578125),
             label=r'$t=\frac{{{}}}{{{:d}}}$'.format("n", 100 * quadratic_factor),
             linestyle='dashed')

    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    # plt.xticks([0, 4000, 8000, 12000, 16000], [0, 4000, 8000, 12000, 16000])
    save_current_plot(savename)
    # plt.show()


def bin_list(vals: List[float]):
    bins = []
    current_bin = [vals[0]]
    for i, v in enumerate(vals[1:]):
        relative_dist = (v - vals[i]) / v
        if relative_dist > 0.05:
            bins.append(current_bin)
            current_bin = [v]
        else:
            current_bin.append(v)

    bins.append(current_bin)
    return bins


def copy_bin(bin_structure: List[List[Any]], list_of_vals: List[float]):
    copied_bin_structure = []
    counter = 0
    for binned_list in bin_structure:
        new_bin = []
        for _ in binned_list:
            new_bin.append(list_of_vals[counter])
            counter += 1
        copied_bin_structure.append(new_bin)

    return copied_bin_structure
