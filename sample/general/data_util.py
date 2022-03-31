# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

import math
from typing import Dict, List, TypeVar, Callable

T = TypeVar('T')
V = TypeVar('V')


def dict_by_attribute(custom_dataclass_list: List[T], attribute: Callable[[T], V]) -> Dict[V, List[T]]:
    """ Get a dict keyed by the attribute values. The values are dataclass instances.

    :param custom_dataclass_list: List of dataclass instances
    :param attribute: An attribute of said dataclass
    :return: A dict with as keys the attribute values and as values the dataclass elements with the key attribute value
    """
    result_by_att = {}
    for t in custom_dataclass_list:
        att = attribute(t)
        if att not in result_by_att:
            result_by_att[att] = []
        result_by_att[att].append(t)
    return result_by_att


def possible_vals(custom_dataclass_list: List[T], attribute: Callable[[T], V]) -> List[V]:
    """ Find all the possible values of a dataclass attribute in a list dataclass instances.

    :param custom_dataclass_list: List of dataclass instances
    :param attribute: An attribute of said dataclass
    :return: Sorted list of possible attribute values
    """
    return sorted(list(set((attribute(t) for t in custom_dataclass_list))))


def mean(my_list: List[float]) -> float:
    if len(my_list) == 0:
        return 0
    return sum(my_list) / len(my_list)


def standard_deviation(my_list: List[float]) -> float:
    avg = mean(my_list)
    return math.sqrt(mean([(i - avg) ** 2 for i in my_list]))


def combine_over_att(custom_dataclass_list: List[T], combiner_attribute: Callable[[T], V],
                     result_attribute: Callable[[T], float]) -> List[float]:
    """ Calculate the average of {@code result_attribute} values, over all values of {@code combiner_attribute}.

    :param custom_dataclass_list: List of dataclass instances
    :param combiner_attribute: An attribute of said class, dataclass instances with the same value are combined
    :param result_attribute: An attribute of said class, this dataclass value is collected
    :return: A list of {@code result_attribute} values, averaged over all values of {@code combiner_attribute}
    """
    x_result = []

    x_y_dict: Dict[List[float]] = {}
    for t in custom_dataclass_list:
        if combiner_attribute(t) not in x_y_dict:
            x_y_dict[combiner_attribute(t)] = []
        x_y_dict[combiner_attribute(t)].append(result_attribute(t))

    for x_val in possible_vals(custom_dataclass_list, combiner_attribute):
        x_result.append(mean(x_y_dict[x_val]))
    return x_result


def smooth(my_list: List[float], smooth_factor=5) -> List[float]:
    """ Returns a smoothed list of the same length as {my_list}. """
    if smooth_factor == 0:
        return my_list
    res = []
    rolling_sum = 0
    for i in range(min(smooth_factor, len(my_list))):
        rolling_sum += my_list[i]
        res.append(rolling_sum / (i + 1))

    n = len(my_list)
    for i in range(smooth_factor, n):
        rolling_sum += my_list[i]
        rolling_sum -= my_list[i - smooth_factor]
        res.append(rolling_sum / smooth_factor)
    return res
