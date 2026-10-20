"""
Stage 2 - structural causal graph over the application variables.

Recipe (the standard 'discovery + knowledge' workflow):
  1. PC algorithm (causal-learn, Fisher-z) on a real-data subsample of numeric variables.
  2. Temporal-tier background knowledge: NO edge may point from a later tier to an earlier tier
     (a loan request cannot cause a person's credit age). This orients every cross-tier edge.
  3. Expert edges for definitional relations (e.g. lti = loan_amnt / annual_inc, revol_util = balance / limit,
     FICO components) and for the four clause variables -> default. Experts override PC orientation.
  4. Per-treatment adjustment sets:
        A_default : tier rule only (config.TREATMENTS[..].drop)
        A_graph   : additionally removes DIRECTED descendants of the treatment variable and its direct
                    undirected/bidirected neighbours (possible mediators / colliders = 'bad controls').
     NB04 fits CALF on both and reports the sensitivity.
Everything is stored in graph.json / adjustment_sets.json and rendered to graph.png.
"""
import json
import numpy as np
import pandas as pd
import networkx as nx

from . import paths
from .config import TIER_OF, TREATMENTS, OUTCOME

NUM_VARS = ["annual_inc_log", "emp_years", "credit_age_yrs", "is_rent", "is_verified", "fico", "delinq_2yrs",
            "inq_last_6mths", "open_acc", "pub_rec", "total_acc", "mort_acc", "revol_bal_log", "revol_util",
            "dti", "loan_amnt_log", "lti", OUTCOME]
NODE_TIER = dict(TIER_OF); NODE_TIER.update({"is_rent": 0, "is_verified": 1, "addr_state": 1, "purpose": 3})
NODE_TO_COLS = {"is_rent": ["home_ownership"], "is_verified": ["verification_status"]}   # graph node -> feature columns

# (src, dst, kind, why)
EXPERT_EDGES = [
    ("annual_inc_log", "dti", "mechanical", "income is the DTI denominator"),
    ("annual_inc_log", "lti", "mechanical", "income is the LTI denominator"),
    ("loan_amnt_log", "lti", "mechanical", "requested amount is the LTI numerator"),
    ("revol_bal_log", "revol_util", "mechanical", "utilization = balance / limit"),
    ("total_acc", "open_acc", "mechanical", "open accounts are a subset of all accounts ever opened"),
    ("credit_age_yrs", "fico", "expert", "length of history is a FICO component"),
    ("delinq_2yrs", "fico", "expert", "payment history is a FICO component"),
    ("inq_last_6mths", "fico", "expert", "new-credit inquiries are a FICO component"),
    ("pub_rec", "fico", "expert", "derogatory records lower the score"),
    ("revol_util", "fico", "expert", "utilization is a FICO component"),
    ("fico", OUTCOME, "expert", "clause variable -> outcome"),
    ("dti", OUTCOME, "expert", "clause variable -> outcome"),
    ("revol_util", OUTCOME, "expert", "clause variable -> outcome"),
    ("lti", OUTCOME, "expert", "clause variable -> outcome"),
    ("annual_inc_log", OUTCOME, "expert", "capacity to repay"),
    ("addr_state", OUTCOME, "expert", "regional economy (kept as an expert-only categorical node)"),
    ("purpose", OUTCOME, "expert", "loan purpose (kept as an expert-only categorical node)"),
]


def discovery_frame(d, max_n=30000, seed=0):
    x = pd.DataFrame({
        "annual_inc_log": np.log1p(d.annual_inc), "emp_years": d.emp_years, "credit_age_yrs": d.credit_age_yrs,
        "is_rent": (d.home_ownership == "RENT").astype(float),
        "is_verified": (d.verification_status != "Not Verified").astype(float),
        "fico": d.fico, "delinq_2yrs": d.delinq_2yrs, "inq_last_6mths": d.inq_last_6mths,
        "open_acc": d.open_acc, "pub_rec": d.pub_rec, "total_acc": d.total_acc, "mort_acc": d.mort_acc,
        "revol_bal_log": np.log1p(d.revol_bal), "revol_util": d.revol_util, "dti": d.dti,
        "loan_amnt_log": np.log1p(d.loan_amnt), "lti": d.loan_amnt / d.annual_inc, OUTCOME: d[OUTCOME]})
    x = x.fillna(x.median())
    if len(x) > max_n:
        x = x.sample(max_n, random_state=seed)
    x["lti"] = x.lti.clip(upper=x.lti.quantile(.995))
    return (x - x.mean()) / x.std().replace(0, 1)


def run_pc(X, alpha=0.001):
    """Returns list of (a, b, directed) with a->b when directed else a--b."""
    names = list(X.columns)
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        from causallearn.graph.GraphNode import GraphNode
        nodes = {n: GraphNode(n) for n in names}
        bk = BackgroundKnowledge()
        for a in names:
            for b in names:
                if NODE_TIER[a] > NODE_TIER[b]:
                    bk.add_forbidden_by_node(nodes[a], nodes[b])
        cg = pc(X.values, alpha=alpha, indep_test="fisherz", node_names=names,
                background_knowledge=bk, show_progress=False)
        G = cg.G.graph
        out = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if G[i, j] == 0 and G[j, i] == 0:
                    continue
                if G[j, i] == 1 and G[i, j] == -1:
                    out.append((names[i], names[j], True))
                elif G[i, j] == 1 and G[j, i] == -1:
                    out.append((names[j], names[i], True))
                else:
                    out.append((names[i], names[j], False))
        return out, "PC (causal-learn, fisherz)"
    except ImportError:
        P = np.linalg.pinv(np.cov(X.values.T))
        pcorr = -P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
        out = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(pcorr[i, j]) > 0.03:
                    a, b = names[i], names[j]
                    if NODE_TIER[a] == NODE_TIER[b]:
                        out.append((a, b, False))
                    else:
                        out.append(((a, b) if NODE_TIER[a] < NODE_TIER[b] else (b, a)) + (True,))
        return out, "partial-correlation skeleton (causal-learn missing; tier-oriented)"


def build_graph(d, alpha=0.001, max_n=30000):
    X = discovery_frame(d, max_n)
    pc_edges, method = run_pc(X, alpha)
    edges = {}
    for a, b, directed in pc_edges:
        if directed:
            edges[(a, b)] = dict(src=a, dst=b, directed=True, kind="discovered", why=method)
        else:
            k = tuple(sorted((a, b)))
            edges[k] = dict(src=k[0], dst=k[1], directed=False, kind="discovered", why=method)
    for a, b, kind, why in EXPERT_EDGES:                       # experts override discovered orientation
        edges.pop((a, b), None); edges.pop((b, a), None); edges.pop(tuple(sorted((a, b))), None)
        edges[(a, b)] = dict(src=a, dst=b, directed=True, kind=kind, why=why)
    nodes = [dict(name=n, tier=NODE_TIER[n], expert_only=n not in NUM_VARS) for n in
             sorted(set(NUM_VARS) | {"addr_state", "purpose"}, key=lambda n: (NODE_TIER[n], n))]
    g = dict(method=method, nodes=nodes, edges=list(edges.values()))
    (paths.ART / "graph.json").write_text(json.dumps(g, indent=1))
    return g


def to_networkx(g):
    G = nx.DiGraph()
    U = set()
    for n in g["nodes"]:
        G.add_node(n["name"], tier=n["tier"])
    for e in g["edges"]:
        if e["directed"]:
            G.add_edge(e["src"], e["dst"], kind=e["kind"])
        else:
            U.add(frozenset((e["src"], e["dst"])))
    return G, U


def adjustment_sets(g):
    G, U = to_networkx(g)
    out = {}
    for tk, spec in TREATMENTS.items():
        v = spec["var"]
        desc = set(nx.descendants(G, v)) - {OUTCOME}
        amb = {x for pair in U if v in pair for x in pair if x != v}
        extra = set()
        for node in desc | amb:
            extra |= set(NODE_TO_COLS.get(node, [node]))
        out[tk] = dict(var=v, parents=sorted(G.predecessors(v)), directed_descendants=sorted(desc),
                       ambiguous_neighbours=sorted(amb), default_drop=sorted(spec["drop"]),
                       graph_drop=sorted(set(spec["drop"]) | extra))
    (paths.ART / "adjustment_sets.json").write_text(json.dumps(out, indent=1))
    return out


def load_graph():
    return json.loads((paths.ART / "graph.json").read_text()), json.loads((paths.ART / "adjustment_sets.json").read_text())


# ----------------------------------------------------------------------------- drawing
TIER_COLORS = {0: "#c7d2fe", 1: "#bbf7d0", 2: "#fde68a", 3: "#fbcfe8", 4: "#fca5a5"}
TIER_NAMES = {0: "Roots", 1: "Soil", 2: "Leaves", 3: "Choice", 4: "Outcome"}


def layout(G):
    pos, cnt = {}, {}
    for n, t in sorted(G.nodes(data="tier"), key=lambda x: (x[1], x[0])):
        cnt[t] = cnt.get(t, 0) + 1
        pos[n] = (t * 3.0, -cnt[t] * 1.0)
    return pos


def draw_graph(g, path=None, highlight=(), edge_labels=None, edge_colors=None, title=None, figsize=(15, 9)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G, U = to_networkx(g) if "nodes" in g else g
    pos = layout(G)
    fig, ax = plt.subplots(figsize=figsize)
    for t, nm in TIER_NAMES.items():
        ax.text(t * 3.0, 1.0, nm, ha="center", fontsize=13, weight="bold", color="#334155")
    nx.draw_networkx_nodes(G, pos, node_color=[TIER_COLORS[G.nodes[n]["tier"]] for n in G],
                           node_size=[2400 if n in highlight else 1500 for n in G],
                           edgecolors=["#dc2626" if n in highlight else "#64748b" for n in G],
                           linewidths=[3 if n in highlight else 1 for n in G], ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ecol = [(edge_colors or {}).get(e, "#475569") for e in G.edges]
    nx.draw_networkx_edges(G, pos, edge_color=ecol, arrows=True, arrowsize=14, width=1.3,
                           connectionstyle="arc3,rad=0.12", node_size=1700, ax=ax)
    for pair in U:
        a, b = tuple(pair)
        ax.plot(*zip(pos[a], pos[b]), ls="--", c="#94a3b8", lw=1, zorder=0)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax,
                                     bbox=dict(boxstyle="round", fc="white", ec="#cbd5e1", alpha=.9), label_pos=.35)
    ax.set_title(title or "Structural causal graph (solid = directed, dashed = undirected)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig
