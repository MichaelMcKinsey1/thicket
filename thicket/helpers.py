# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import re
from difflib import SequenceMatcher

from more_itertools import powerset
import pandas as pd


def _are_synced(gh, df):
    """Check if node objects are equal in graph and dataframe id(graph_node) ==
    id(df_node).
    """
    for graph_node in gh.traverse():
        node_present = False
        for df_node in df.index.get_level_values("node"):
            if id(df_node) == id(graph_node):
                node_present = True
                continue
        if not node_present:
            return False
    return True


def _missing_nodes_to_list(a_df, b_df):
    """Get a list of node differences between two dataframes. Mainly used for "tree"
    function.

    Arguments:
        a_df (Dataframe): First pandas Dataframe
        b_df (Dataframe): Second pandas Dataframe

    Returns:
        (list): List of numbers in range (0, 1, 2). "0" means node is in both, "1" is
            only in "a", "2" is only in "b"
    """
    missing_nodes = []
    a_list = list(map(hash, list(a_df.index.get_level_values("node"))))
    b_list = list(map(hash, list(b_df.index.get_level_values("node"))))
    # Basic cases
    while a_list and b_list:
        a = a_list.pop(0)
        b = b_list.pop(0)
        while a > b and b_list:
            missing_nodes.append(2)
            b = b_list.pop(0)
        while b > a and a_list:
            missing_nodes.append(1)
            a = a_list.pop(0)
        if a == b:
            missing_nodes.append(0)
            continue
        elif a > b:  # Case where last two nodes and "a" is missing "b" then opposite
            missing_nodes.append(2)
            missing_nodes.append(1)
        elif b > a:  # Case where last two nodes and "b" is missing "a" then opposite
            missing_nodes.append(1)
            missing_nodes.append(2)
    while a_list:  # In case "a" has a lot more nodes than "b"
        missing_nodes.append(1)
        a = a_list.pop(0)
    while b_list:  # In case "b" has a lot more nodes than "a"
        missing_nodes.append(2)
        b = b_list.pop(0)
    return missing_nodes


def _new_statsframe_df(df, multiindex=False):
    """Generate new aggregated statistics table from a dataframe. This is most commonly
    needed when changes are made to the performance data table's index.

    Arguments:
        df (DataFrame): Input dataframe to generate the aggregated statistics table from
        multiindex (Bool, optional): Option to setup MultiIndex column structure. This
            is standard to do if performance data table is MultiIndex.

    Returns:
        (DataFrame): new aggregated statistics table
    """
    nodes = list(set(df.reset_index()["node"]))  # List of nodes
    names = [node.frame["name"] for node in nodes]  # List of names

    # Create new dataframe with "node" index and "name" data by default.
    new_df = pd.DataFrame(
        data={"node": nodes, "name": names},
    ).set_index("node")

    # Create MultiIndex structure if necessary.
    if multiindex:
        new_df.columns = pd.MultiIndex.from_tuples([("name", "")])

    return new_df


def _print_graph(graph):
    """Print the nodes in a hatchet graph"""
    i = 0
    for node in graph.traverse():
        print(f"{node} ({hash(node)}) ({id(node)})")
        i += 1
    return i


def _resolve_missing_indicies(th_list):
    """Resolve indices if at least 1 profile has an index that another doesn't.

    If at least one profile has an index that another doesn't, then issues will arise
    when unifying. Need to add this index to other thickets.

    Note that the value to use for the new index is set to '0' for ease-of-use, but
    something like 'NaN' may arguably provide more clarity.
    """
    # Create a set of all index possibilities
    idx_set = set({})
    for th in th_list:
        idx_set.update(th.dataframe.index.names)

    # Apply missing indicies to thickets
    for th in th_list:
        for idx in idx_set:
            if idx not in th.dataframe.index.names:
                print(f"Resolving '{idx}' in thicket: ({id(th)})")
                th.dataframe[idx] = 0
                th.dataframe.set_index(idx, append=True, inplace=True)


def _set_node_ordering(thickets):
    """Set node ordering for each thicket in a list. All thickets must have node ordering on, otherwise it will be set to False.

    Arguments:
        thickets (list): list of Thicket objects
    """
    node_order = all([tk.graph.node_ordering for tk in thickets])

    for tk in thickets:
        if tk.graph.node_ordering:
            tk.graph.node_ordering = node_order
            # Have to re-enumerate the traverse
            tk.graph.enumerate_traverse()


def _get_perf_columns(df):
    """Get list of performance dataframe columns that are numeric.

    Numeric columns can be used with thicket's statistical functions.

    Arguments:
        df (DataFrame): thicket dataframe object

    Returns:
        numeric_columns (list): list of numeric columns
    """
    numeric_types = ["int32", "int64", "float32", "float64"]

    numeric_columns = df.select_dtypes(include=numeric_types).columns.tolist()

    # thicket object without columnar index
    if df.columns.nlevels == 1:
        if "nid" in numeric_columns:
            numeric_columns.remove("nid")

        return numeric_columns
    # columnar joined thicket object
    else:
        return [x for x in numeric_columns if "nid" not in x]


def _powerset_from_tuple(tup):
    pset = [y for y in powerset(tup)]
    return {x[0] if len(x) == 1 else x for x in pset}


def _match_call_trace_regex(kernel_call_trace, demangled_kernel_name, debug, action=None):
    """Use the NCU call trace to regex match the kernel name from the demangled
    kernel string. Also modifies the demangled kernel name in certain cases. Returns
    the matched kernel string, if match is possible.

    Arguments:
        kernel_call_trace (list): List of strings from NCU representing the call trace
        demangled_kernel_name (str): Demangled kernel name from NCU
        debug (bool): Print debug statements
        action (ncu_report.IAction): NCU action object
    """
    # Call trace with last element removed
    # (last elem usually not useful for matching)
    temp_call_trace = kernel_call_trace[:-1]
    # Special case to match "cub" kernels
    if "cub" in demangled_kernel_name:
        call_trace_str = "cub"
        # Replace substrings that may cause mismatch
        demangled_kernel_name = demangled_kernel_name.replace("(bool)1", "true")
        demangled_kernel_name = demangled_kernel_name.replace("(bool)0", "false")
    else:
        call_trace_str = "::".join([s.lower() for s in temp_call_trace])
    if debug:
        print(f"\tKernel Call Trace: {kernel_call_trace}")
        print(f"\t{action.name()}")

    # Pattern ends with ":" if RAJA_CUDA, "<" if Base_CUDA
    kernel_pattern = rf"{call_trace_str}::(\w+)[<:]"
    kernel_match = re.search(kernel_pattern, demangled_kernel_name)
    # Found match
    if kernel_match:
        kernel_str = kernel_match.group(1)
    else:
        if debug:
            print(f"\tCould not match {demangled_kernel_name}")
        return None, None, None, True

    # RAJA_CUDA/Lambda_CUDA variant
    instance_pattern = r"instance (\d+)"
    instance_match = re.findall(instance_pattern, demangled_kernel_name)
    if instance_match:
        instance_num = instance_match[-1]
        instance_exists = True
    else:
        # Base_CUDA variant
        instance_num = None
        instance_exists = False

    return kernel_str, demangled_kernel_name, instance_num, instance_exists, False


def _match_kernel_str_to_cali(
    node_set, kernel_str, instance_num, raja_lambda_cuda, instance_exists
):
    """Given a set of nodes, node_set, from querying the Caliper call
    tree using the NCU call trace, match the kernel_str to one of the
    node names. Additionally, use the instance number, instance_num to
    match kernels with multiple instances, if applicable.

    Arguments:
        node_set (list): List of Hatchet nodes from querying the call tree
        kernel_str (str): Kernel name from _match_call_trace_regex
        instance_num (int): Instance number of kernel, if applicable
        raja_lambda_cuda (bool): True if RAJA_CUDA or Lambda_CUDA, False if Base_CUDA
        instance_exists (bool): True if instance number exists, False if not
    """
    return [
        n
        for n in node_set
        if kernel_str in n.frame["name"]
        and (
            f"#{instance_num}" in n.frame["name"]
            if raja_lambda_cuda and instance_exists
            else True
        )
    ]


def _multi_match_fallback_similarity(matched_nodes, demangled_kernel_name, debug):
    """If _match_kernel_str_to_cali has more than one match, attempt to match using sequence similarity.

    Arguments:
        matched_nodes (list): List of matched Hatchet nodes
        demangled_kernel_name (str): Demangled kernel name from _match_call_trace_regex
        debug (bool): Print debug statements

    Returns:
        matched_node (Hatchet.node): Hatchet node with highest similarity score
    """
    # Attempt to match using similarity
    match_dict = {}
    for node in matched_nodes:
        match_ratio = SequenceMatcher(
            None, node.frame["name"], demangled_kernel_name
        ).ratio()
        match_dict[match_ratio] = node
    # Get highest ratio
    highest_ratio = max(list(match_dict.keys()))
    matched_node = match_dict[highest_ratio]
    if debug:
        print(
            f"NOTICE: Multiple matches ({len(matched_nodes)}) found for kernel. Matching using string similarity..."
        )
    return matched_node
