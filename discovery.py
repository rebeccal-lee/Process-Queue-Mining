from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class DFGViewConfig:
    min_edge_count: int = 1
    top_k_edges: Optional[int] = None

    view_mode: Literal["full", "ego"] = "full"
    focus_activity: Optional[str] = None
    ego_depth: int = 1

    rankdir: Literal["LR", "TB"] = "LR"
    show_edge_labels: bool = True
    node_label_with_count: bool = True
    max_nodes: Optional[int] = None


@dataclass(frozen=True)
class DFGResult:
    edges: pd.DataFrame # columns: ['source','target','count']
    nodes: pd.DataFrame # columns: ['activity','count']
    case_col: str
    act_col: str
    time_col: str

'''
Compute DFGs 
'''
def compute_dfg(
    event_log_df: pd.DataFrame,
    *,
    case_col: str = "case_id",
    act_col: str = "activity",
    time_col: str = "timestamp",
    drop_self_loops: bool = False,
) -> DFGResult:
    required = {case_col, act_col, time_col}
    missing = required - set(event_log_df.columns)
    if missing:
        raise KeyError(f"event_log_df missing required columns: {missing}. Found: {list(event_log_df.columns)}")

    df = event_log_df[[case_col, act_col, time_col]].copy()
    df = df.sort_values([case_col, time_col], kind="mergesort")
    df["next_activity"] = df.groupby(case_col)[act_col].shift(-1)

    edges = (
        df.dropna(subset=["next_activity"])
          .groupby([act_col, "next_activity"])
          .size()
          .reset_index(name="count")
          .rename(columns={act_col: "source", "next_activity": "target"})
          .sort_values("count", ascending=False)
          .reset_index(drop=True)
    )

    if drop_self_loops:
        edges = edges[edges["source"] != edges["target"]].reset_index(drop=True)

    nodes = (
        df.groupby(act_col)
          .size()
          .reset_index(name="count")
          .rename(columns={act_col: "activity"})
          .sort_values("count", ascending=False)
          .reset_index(drop=True)
    )

    return DFGResult(edges=edges, nodes=nodes, case_col=case_col, act_col=act_col, time_col=time_col)

def filter_dfg(
    dfg: DFGResult,
    *,
    min_edge_count: int = 1,
    top_k_edges: Optional[int] = None,
) -> DFGResult:
    edges = dfg.edges.copy()
    edges = edges[edges["count"] >= min_edge_count]

    if top_k_edges is not None:
        edges = edges.sort_values("count", ascending=False).head(top_k_edges)

    keep_acts = pd.unique(edges[["source", "target"]].values.ravel("K"))
    nodes = dfg.nodes[dfg.nodes["activity"].isin(keep_acts)].copy()

    return DFGResult(edges=edges.reset_index(drop=True),
                     nodes=nodes.reset_index(drop=True),
                     case_col=dfg.case_col, act_col=dfg.act_col, time_col=dfg.time_col)

def drilldown_ego(
    dfg: DFGResult,
    focus_activity: str,
    *,
    depth: int = 1,
    min_edge_count: int = 1,
) -> DFGResult:
    edges = dfg.edges.copy()
    edges = edges[edges["count"] >= min_edge_count].copy()

    frontier = {focus_activity}
    visited = {focus_activity}

    for _ in range(depth):
        sub = edges[edges["source"].isin(frontier) | edges["target"].isin(frontier)]
        new_nodes = set(sub["source"]).union(set(sub["target"]))
        frontier = new_nodes - visited
        visited |= new_nodes

    sub_edges = edges[edges["source"].isin(visited) & edges["target"].isin(visited)].copy()
    sub_nodes = dfg.nodes[dfg.nodes["activity"].isin(visited)].copy()

    return DFGResult(edges=sub_edges.reset_index(drop=True),
                     nodes=sub_nodes.reset_index(drop=True),
                     case_col=dfg.case_col, act_col=dfg.act_col, time_col=dfg.time_col)


def list_activities(dfg: DFGResult, *, top_n: Optional[int] = None) -> list[str]:
    acts = dfg.nodes.sort_values("count", ascending=False)["activity"].astype(str).tolist()
    return acts if top_n is None else acts[:top_n]

'''
Visualization via Plotly
'''
def build_plotly_dfg(
    dfg: DFGResult,
    *,
    rankdir: str = "LR",
    min_edge_count: int = 1,
    max_nodes: Optional[int] = None,
    show_edge_labels: bool = True,
    node_label_with_count: bool = True,
):
    """
    Plotly DFG using Graphviz dot diagram for layout coordinates
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("plotly is required. Install with: pip install plotly") from e

    try:
        from graphviz import Digraph
    except ImportError as e:
        raise ImportError("graphviz python package is required (for layout). Install with: pip install graphviz") from e

    # nodes
    nodes_df = dfg.nodes.copy()
    if max_nodes is not None:
        nodes_df = nodes_df.sort_values("count", ascending=False).head(max_nodes)

    nodes_df = nodes_df.copy()
    nodes_df["activity"] = nodes_df["activity"].astype(str)

    allowed = set(nodes_df["activity"].tolist())

    # edges
    edges_df = dfg.edges.copy()
    edges_df = edges_df[edges_df["count"] >= min_edge_count].copy()
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)

    if max_nodes is not None:
        edges_df = edges_df[edges_df["source"].isin(allowed) & edges_df["target"].isin(allowed)].copy()

    if nodes_df.empty:
        return go.Figure()

    activities = nodes_df["activity"].tolist()
    act_to_id = {act: f"n{i}" for i, act in enumerate(activities)}
    id_to_act = {v: k for k, v in act_to_id.items()}

    # generate graphviz layout
    dot = Digraph(engine="dot")
    dot.attr(rankdir=rankdir)

    # add nodes
    for act in activities:
        dot.node(act_to_id[act], label=act)

    # edges
    for _, r in edges_df.iterrows():
        s = act_to_id.get(r["source"])
        t = act_to_id.get(r["target"])
        if s is None or t is None:
            continue
        dot.edge(s, t)

    plain = dot.pipe(format="plain").decode("utf-8")

    pos: Dict[str, Tuple[float, float]] = {}
    for line in plain.splitlines():
        if line.startswith("node "):
            parts = line.split()
            safe_id = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            pos[safe_id] = (x, y)

    # counts
    node_counts = dict(zip(nodes_df["activity"], nodes_df["count"].astype(int)))
    max_cnt = max(node_counts.values()) if node_counts else 1

    def node_size(c: int) -> float:
        return 14 + 26 * (c / max_cnt) if max_cnt > 0 else 14

    # edges as segments
    edge_x, edge_y = [], []
    edge_mid_x, edge_mid_y, edge_mid_text = [], [], []

    for _, r in edges_df.iterrows():
        s_act = r["source"]
        t_act = r["target"]
        c = int(r["count"])

        s = act_to_id.get(s_act)
        t = act_to_id.get(t_act)
        if s is None or t is None or s not in pos or t not in pos:
            continue

        x0, y0 = pos[s]
        x1, y1 = pos[t]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        if show_edge_labels:
            edge_mid_x.append((x0 + x1) / 2.0)
            edge_mid_y.append((y0 + y1) / 2.0)
            edge_mid_text.append(str(c))

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none")
    edge_label_trace = go.Scatter(x=edge_mid_x, y=edge_mid_y, mode="text", text=edge_mid_text, hoverinfo="none")

    # nodes
    nx, ny, text, hover, sizes = [], [], [], [], []
    for act, c in node_counts.items():
        safe_id = act_to_id.get(act)
        if safe_id is None or safe_id not in pos:
            continue
        x, y = pos[safe_id]
        nx.append(x); ny.append(y)

        label = f"{act}\n({c})" if node_label_with_count else act
        text.append(label)
        hover.append(f"{act}<br>events: {c}")
        sizes.append(node_size(c))

    node_trace = go.Scatter(
        x=nx, y=ny,
        mode="markers+text",
        text=text,
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker=dict(size=sizes),
    )

    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig

def build_dfg_view(
    event_log_df: pd.DataFrame,
    *,
    cfg: Optional[DFGViewConfig] = None,
    case_col: str = "case_id",
    act_col: str = "activity",
    time_col: str = "timestamp",
    drop_self_loops: bool = False,
):
    """
    Plotly figure for visualization in streamlit UI
    """
    if cfg is None:
        cfg = DFGViewConfig()

    base = compute_dfg(
        event_log_df,
        case_col=case_col,
        act_col=act_col,
        time_col=time_col,
        drop_self_loops=drop_self_loops,
    )

    dfg_f = filter_dfg(base, min_edge_count=cfg.min_edge_count, top_k_edges=cfg.top_k_edges)

    dfg_out = dfg_f
    if cfg.view_mode == "ego":
        if not cfg.focus_activity:
            raise ValueError("cfg.focus_activity must be set when cfg.view_mode='ego'")
        dfg_out = drilldown_ego(
            dfg_f,
            focus_activity=cfg.focus_activity,
            depth=cfg.ego_depth,
            min_edge_count=cfg.min_edge_count,
        )

    fig = build_plotly_dfg(
        dfg_out,
        rankdir=cfg.rankdir,
        min_edge_count=cfg.min_edge_count,
        max_nodes=cfg.max_nodes,
        show_edge_labels=cfg.show_edge_labels,
        node_label_with_count=cfg.node_label_with_count,
    )

    return dfg_out, fig
