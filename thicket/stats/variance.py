# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from ..utils import verify_thicket_structures


def variance(thicket, columns=None):
    """Calculate the variance for each node in the performance data table.

    Designed to take in a thicket, and append one or more columns to the aggregated
    statistics table for the variance calculation for each node.

    Variance will allow you to see the spread of data within a node and that nodes
    profiles.

    Arguments:
        thicket (thicket): Thicket object
        columns (list): List of hardware/timing metrics to perform variance calculation
            on. Note, if using a columnar_joined thicket a list of tuples must be passed
            in with the format (column index, column name).
    """
    if columns is None:
        raise ValueError(
            "To see a list of valid columns, please run Thicket.get_perf_columns()."
        )

    verify_thicket_structures(
        thicket.dataframe, index=["node", "profile"], columns=columns
    )

    # thicket object without columnar index
    if thicket.dataframe.columns.nlevels == 1:
        for column in columns:
            var = []
            for node in pd.unique(thicket.dataframe.reset_index()["node"].tolist()):
                var.append(np.var(thicket.dataframe.loc[node][column]))
            # check to see if exclusive metric
            if column in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.append(column + "_var")
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.append(column + "_var")

            thicket.statsframe.dataframe[column + "_var"] = var
    # columnar joined thicket object
    else:
        for idx, column in columns:
            var = []
            for node in pd.unique(thicket.dataframe.reset_index()["node"].tolist()):
                var.append(np.var(thicket.dataframe.loc[node][(idx, column)]))
            # check to see if exclusive metric
            if (idx, column) in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.append((idx, column + "_var"))
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.append((idx, column + "_var"))
            thicket.statsframe.dataframe[(idx, column + "_var")] = var

        # sort columns in index
        thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)