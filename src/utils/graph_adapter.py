"""Thin adapter for igraph with node-ID mapping.

The project is migrating from NetworkX to igraph for the primary graph
representation (30.5M nodes, 95.5M edges).  This module centralises the
ID-mapping bookkeeping so call-sites stay clean.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import igraph as ig

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------------

def build_igraph_from_df(
    df: pd.DataFrame,
    directed: bool = True,
    weight_col: str = "usd_value",
    aggregate: bool = True,
) -> ig.Graph:
    """Build an igraph Graph from an edge-list DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain ``source_id`` and ``target_id`` columns.  May also
        contain ``btc_value``, ``usd_value``, and ``count``.
    directed : bool
        Whether to create a directed graph.
    weight_col : str
        Column to use as the canonical ``weight`` edge attribute.
    aggregate : bool
        If *True*, duplicate ``(source_id, target_id)`` pairs are collapsed
        and numeric columns are summed.

    Returns
    -------
    ig.Graph
        Graph with ``vs['name']`` set to the original node IDs (sorted) and
        edge attributes ``weight``, ``btc_value``, ``usd_value``, ``count``
        where available.
    """
    weight_cols = [c for c in ("btc_value", "usd_value", "count") if c in df.columns]

    if aggregate and weight_cols:
        df = df.groupby(["source_id", "target_id"], sort=False)[weight_cols].sum().reset_index()

    # Sorted unique node list for deterministic indexing.
    nodes = np.union1d(df["source_id"].unique(), df["target_id"].unique())
    nodes.sort()
    name_to_idx_map: dict[object, int] = {n: i for i, n in enumerate(nodes)}

    src_idx = df["source_id"].map(name_to_idx_map).values
    tgt_idx = df["target_id"].map(name_to_idx_map).values

    g = ig.Graph(n=len(nodes), edges=list(zip(src_idx, tgt_idx)), directed=directed)
    g.vs["name"] = nodes.tolist()

    # Edge attributes
    if weight_col in df.columns:
        g.es["weight"] = df[weight_col].values.tolist()
    for col in weight_cols:
        g.es[col] = df[col].values.tolist()
    if "count" not in df.columns:
        g.es["count"] = [1] * g.ecount()

    # Cache the reverse mapping on the graph object.
    g["_name_to_idx"] = name_to_idx_map

    logger.info(
        "Built igraph (%s): %s nodes, %s edges",
        "directed" if directed else "undirected",
        f"{g.vcount():,}",
        f"{g.ecount():,}",
    )
    return g


def build_igraph_from_edges(
    edge_data: dict[tuple, list],
    directed: bool = False,
) -> ig.Graph:
    """Build an igraph Graph from the streaming dict produced by main.py.

    Parameters
    ----------
    edge_data : dict
        Mapping of ``(src, tgt) -> [usd_value, btc_value, count]``.
    directed : bool
        Whether to create a directed graph.

    Returns
    -------
    ig.Graph
    """
    # Collect unique node IDs.
    node_set: set = set()
    for src, tgt in edge_data:
        node_set.add(src)
        node_set.add(tgt)
    nodes = sorted(node_set)
    del node_set

    name_to_idx_map: dict[object, int] = {n: i for i, n in enumerate(nodes)}

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    btc_values: list[float] = []
    counts: list[int] = []

    for (src, tgt), d in edge_data.items():
        edges.append((name_to_idx_map[src], name_to_idx_map[tgt]))
        weights.append(d[0])
        btc_values.append(d[1])
        counts.append(int(d[2]))

    g = ig.Graph(n=len(nodes), edges=edges, directed=directed)
    g.vs["name"] = nodes
    g.es["weight"] = weights
    g.es["usd_value"] = weights
    g.es["btc_value"] = btc_values
    g.es["count"] = counts

    g["_name_to_idx"] = name_to_idx_map

    logger.info(
        "Built igraph from edge dict (%s): %s nodes, %s edges",
        "directed" if directed else "undirected",
        f"{g.vcount():,}",
        f"{g.ecount():,}",
    )
    return g


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def name_to_idx(g: ig.Graph) -> dict:
    """Return a ``{node_name: vertex_index}`` mapping, cached on the graph.

    The cache is stored as ``g['_name_to_idx']`` so repeated calls are O(1).
    """
    try:
        return g["_name_to_idx"]
    except KeyError:
        mapping = {name: idx for idx, name in enumerate(g.vs["name"])}
        g["_name_to_idx"] = mapping
        return mapping


def get_edge_attr(
    g: ig.Graph,
    src_idx: int,
    tgt_idx: int,
    attr: str = "weight",
) -> Optional[float]:
    """Get an edge attribute by source/target vertex indices.

    Returns *None* if no such edge exists.
    """
    try:
        eid = g.get_eid(src_idx, tgt_idx)
    except ig.InternalError:
        return None
    return g.es[eid][attr]


# ---------------------------------------------------------------------------
# NetworkX interop (lazy import)
# ---------------------------------------------------------------------------

def to_networkx(g: ig.Graph):
    """Convert an igraph graph to a NetworkX graph.

    NetworkX is imported lazily so it can remain an optional dependency.
    Edge attributes (``weight``, ``btc_value``, ``count``, etc.) are
    preserved.  Node IDs in the returned graph are the original names
    from ``g.vs['name']``.

    Returns
    -------
    nx.Graph or nx.DiGraph
    """
    import networkx as nx  # noqa: E402 — lazy import

    names = g.vs["name"]
    nxg = nx.DiGraph() if g.is_directed() else nx.Graph()
    nxg.add_nodes_from(names)

    attr_names = g.es.attributes()
    for e in g.es:
        src_name = names[e.source]
        tgt_name = names[e.target]
        attrs = {a: e[a] for a in attr_names}
        nxg.add_edge(src_name, tgt_name, **attrs)

    return nxg
