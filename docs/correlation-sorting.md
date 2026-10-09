# dendrites: correlation sorting

# Correlation Chain Analysis — Transcription
"""
Transcribed from photographed printout (9 pages, 550 lines). Cross-verified against
two independent photo passes; all page boundaries and previously edge-cut lines
(152–155, 399, 416–417, 332–398) confirmed against the sharper set. See note at
bottom for the one section reconstructed by logical inference rather than a clean
pixel read.
"""
```python
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, yeojohnson
from scipy.ndimage import label, find_objects
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def _preprocess(df):
    num = df.select_dtypes(include=[np.number]).dropna().copy()

    for c in num.columns:
        x = num[c].to_numpy(dtype=float)
        x = np.sign(x) * np.log1p(np.abs(x))
        x = yeojohnson(x)[0]
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        num[c] = (x - med) / (mad if mad else 1.0)

    return num


def correlation_chains(df, alpha=0.05):
    num = _preprocess(df)
    cols = list(num.columns)
    corr, sig = {}, {}

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r, p = pearsonr(num[a], num[b])
            corr[a, b] = corr[b, a] = r
            sig[a, b] = sig[b, a] = (p < alpha) and (r > 0)

    def mean_sig_corr(c):
        vals = [corr[c, o] for o in cols if o != c and sig[c, o]]
        return np.mean(vals) if vals else 0.0

    unassigned, chains = set(cols), []

    while unassigned:
        master = max(unassigned, key=mean_sig_corr)
        chain = [master]
        unassigned.remove(master)
        tail = master

        while True:
            candidates = [o for o in unassigned if sig[tail, o]]
            if not candidates:
                break
            tail = max(candidates, key=lambda o: corr[tail, o])
            chain.append(tail)
            unassigned.remove(tail)

        chains.append(chain)

    return chains


def plot_correlation_graph(df, dendrites, alpha=0.05, show_cross_group_summary=True):
    num = _preprocess(df)
    cols = list(num.columns)

    feature_to_group = {
        feature: group_id
        for group_id, chain in enumerate(dendrites)
        for feature in chain
    }

    G = nx.Graph()

    for c in cols:
        G.add_node(c, group=feature_to_group.get(c, -1))

    backbone_edges = set()
    for chain in dendrites:
        for a, b in zip(chain[:-1], chain[1:]):
            r, p = pearsonr(num[a], num[b])
            edge = tuple(sorted((a, b)))
            backbone_edges.add(edge)
            G.add_edge(
                a, b,
                weight=abs(r),
                corr=r,
                p=p,
                backbone=True,
                same_group=True,
                strongest_cross=False
            )

    cross_group_best = {}

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r, p = pearsonr(num[a], num[b])
            if p >= alpha:
                continue

            same_group = feature_to_group[a] == feature_to_group[b]
            edge = tuple(sorted((a, b)))

            if not same_group:
                ga = feature_to_group[a]
                gb = feature_to_group[b]
                group_pair = tuple(sorted((ga, gb)))
                current = cross_group_best.get(group_pair)

                if current is None or abs(r) > abs(current["r"]):
                    cross_group_best[group_pair] = {
                        "group_a": group_pair[0] + 1,
                        "group_b": group_pair[1] + 1,
                        "feature_a": a,
                        "feature_b": b,
                        "r": r,
                        "p": p,
                        "edge": edge
                    }

            if edge not in backbone_edges:
                G.add_edge(
                    a, b,
                    weight=abs(r),
                    corr=r,
                    p=p,
                    backbone=False,
                    same_group=same_group,
                    strongest_cross=False
                )

    strongest_cross_edges = {v["edge"] for v in cross_group_best.values()}

    for a, b in G.edges():
        edge = tuple(sorted((a, b)))
        if edge in strongest_cross_edges and not G[a][b]["same_group"]:
            G[a][b]["strongest_cross"] = True

    pos = nx.spring_layout(G, seed=42, weight="weight")

    node_colors = [G.nodes[n]["group"] for n in G.nodes]
    cmap = plt.cm.Set2

    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap=cmap,
        node_size=1800,
        edgecolors="black"
    )

    nx.draw_networkx_labels(G, pos, font_size=10)

    backbone = [(u, v) for u, v, d in G.edges(data=True) if d["backbone"]]
    within = [(u, v) for u, v, d in G.edges(data=True) if not d["backbone"] and d["same_group"]]
    cross_strong = [(u, v) for u, v, d in G.edges(data=True) if not d["backbone"] and not d["same_group"] and d["strongest_cross"]]
    cross_other = [(u, v) for u, v, d in G.edges(data=True) if not d["backbone"] and not d["same_group"] and not d["strongest_cross"]]

    def edge_colors(edges):
        return ["green" if G[u][v]["corr"] > 0 else "red" for u, v in edges]

    def edge_widths(edges, scale=4.0, base=0.5):
        return [base + scale * G[u][v]["weight"] for u, v in edges]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=within,
        edge_color=edge_colors(within),
        width=edge_widths(within, scale=3.0, base=0.8),
        alpha=0.5
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=cross_other,
        edge_color=edge_colors(cross_other),
        width=edge_widths(cross_other, scale=3.0, base=0.8),
        style="dashed",
        alpha=0.7
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=cross_strong,
        edge_color=edge_colors(cross_strong),
        width=edge_widths(cross_strong, scale=4.5, base=1.2),
        alpha=0.95,
        style="solid"
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=backbone,
        edge_color=edge_colors(backbone),
        width=edge_widths(backbone, scale=5.0, base=1.5),
        alpha=0.95
    )

    plt.title("Significant correlation graph with backbone chains")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    cross_group_summary = pd.DataFrame(cross_group_best.values()).drop(columns="edge")
    cross_group_summary = cross_group_summary.sort_values(
        by=["group_a", "group_b"]
    ).reset_index(drop=True)

    if show_cross_group_summary and not cross_group_summary.empty:
        print("\nStrongest significant cross-group links:")
        print(cross_group_summary.to_string(index=False))

    return G, cross_group_summary


def find_terminal_negative_cross_group_links(num, dendrites, alpha=0.05):
    """
    Find, for each chain, the strongest significant negative cross-group link
    from that chain's terminal feature to any feature in a different chain.

    Require:
    - num is a preprocessed numeric DataFrame with no missing values.
    - dendrites is a list of chains covering the plotted features.
    - alpha is the significance cutoff for Pearson correlation.

    Guarantee:
    - Returns one record per chain terminal when at least one significant
      negative cross-group partner exists.
    - Each record includes source/target features and correlation statistics.

    Failure modes:
    - Propagates Pearson correlation errors if columns are invalid.
    - Returns fewer than len(dendrites) records when some terminals have no
      eligible negative significant cross-group partner.
    """
    cols = list(num.columns)

    feature_to_group = {
        feature: group_id
        for group_id, chain in enumerate(dendrites)
        for feature in chain
    }

    terminal_links = []

    for chain_id, chain in enumerate(dendrites, start=1):
        terminal = chain[-1]
        terminal_group = feature_to_group[terminal]
        best = None

        for other in cols:
            if other == terminal:
                continue
            if feature_to_group.get(other, -1) == terminal_group:
                continue

            r, p = pearsonr(num[terminal], num[other])

            if p < alpha and r < 0:
                if best is None or r < best["r"]:
                    best = {
                        "chain": chain_id,
                        "terminal_feature": terminal,
                        "target_feature": other,
                        "r": r,
                        "p": p
                    }

        if best is not None:
            terminal_links.append(best)

    return pd.DataFrame(terminal_links)


def overlay_terminal_negative_highlights(
    ax,
    order,
    dendrites,
    terminal_negative_links,
    color="red",
    cell_linewidth=3.0,
    bracket_linewidth=2.5,
    bracket_frac=0.22,
    label_color="#8B0000",
    tint_terminal_labels=True
):
    """
    Overlay terminal negative cross-group highlights on an existing upper-triangular
    correlation heatmap.

    Visual encoding:
    - Thick red rectangle around the selected cell.
    - Small red bracket on the row side and column side to tie the mark back
      to the matrix axes.
    - Optional tint on terminal feature tick labels.

    Require:
    - ax is the matplotlib Axes containing the heatmap.
    - order is the feature order used to build the heatmap matrix.
    - dendrites is the chain decomposition used to define terminal features.
    - terminal_negative_links is the DataFrame returned by
      find_terminal_negative_cross_group_links().

    Guarantee:
    - Adds overlays only for terminal links whose features exist in `order`.
    - Does not modify the heatmap values or mask.

    Failure modes:
    - Silently skips links whose features are absent from `order`.
    - Assumes seaborn heatmap cell coordinates are integer-aligned.
    """
    order_index = {feature: i for i, feature in enumerate(order)}
    terminal_features = {chain[-1] for chain in dendrites}

    for _, row in terminal_negative_links.iterrows():
        a = row["terminal_feature"]
        b = row["target_feature"]

        if a not in order_index or b not in order_index:
            continue

        i = order_index[a]
        j = order_index[b]

        r_idx = min(i, j)
        c_idx = max(i, j)

        rect = Rectangle(
            (c_idx, r_idx),
            1,
            1,
            fill=False,
            edgecolor=color,
            linewidth=cell_linewidth,
            linestyle="-",
            zorder=10
        )
        ax.add_patch(rect)

        y_mid = r_idx + 0.5
        x_left = c_idx
        dx = bracket_frac

        ax.plot(
            [x_left, x_left - dx],
            [y_mid, y_mid],
            color=color,
            linewidth=bracket_linewidth,
            solid_capstyle="round",
            zorder=11
        )
        ax.plot(
            [x_left - dx, x_left - dx],
            [y_mid - dx, y_mid + dx],
            color=color,
            linewidth=bracket_linewidth,
            solid_capstyle="round",
            zorder=11
        )

        x_mid = c_idx + 0.5
        y_top = r_idx
        dy = bracket_frac

        ax.plot(
            [x_mid, x_mid],
            [y_top, y_top - dy],
            color=color,
            linewidth=bracket_linewidth,
            solid_capstyle="round",
            zorder=11
        )
        ax.plot(
            [x_mid - dy, x_mid + dy],
            [y_top - dy, y_top - dy],
            color=color,
            linewidth=bracket_linewidth,
            solid_capstyle="round",
            zorder=11
        )

        if tint_terminal_labels:
            for tick in ax.get_xticklabels():
                label = tick.get_text()
                if label in terminal_features:
                    tick.set_color(label_color)
                    tick.set_fontweight("bold")

            for tick in ax.get_yticklabels():
                label = tick.get_text()
                if label in terminal_features:
                    tick.set_color(label_color)
                    tick.set_fontweight("bold")


def main():
    """
    Load the CSV, build correlation chains, show the masked heatmap with
    terminal negative cross-group highlights, print chain links, and plot the
    correlation graph with strongest cross-group summaries.

    Require:
    - CSV path points to a readable file with numeric columns.
    - find_terminal_negative_cross_group_links() and
      overlay_terminal_negative_highlights() are defined.

    Guarantee:
    - Produces the heatmap, chain-link table, terminal negative cross-group
      summary, network graph, and cross-group summary.

    Failure modes:
    - Propagates file, preprocessing, correlation, and plotting errors.
    """
    df = pd.read_csv(
        r'C:\Users\mb445c\OneDrive - The Boeing Company\Documents\personal\states.csv'
    )
    num = _preprocess(df)
    dendrites = correlation_chains(df)

    order = [feature for chain in dendrites for feature in chain]
    sizes = [len(chain) for chain in dendrites]
    bounds = np.cumsum(sizes)[:-1]

    corr_matrix = num[order].corr()

    feature_to_group = {
        feature: group_id
        for group_id, chain in enumerate(dendrites)
        for feature in chain
    }

    n = len(order)
    display_mask = np.zeros((n, n), dtype=bool)
    sig_offblock = np.zeros((n, n), dtype=bool)

    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if j <= i:
                display_mask[i, j] = True
                continue

            r, p = pearsonr(num[a], num[b])

            if p >= 0.05:
                display_mask[i, j] = True
            elif feature_to_group[a] != feature_to_group[b]:
                sig_offblock[i, j] = True

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(
        corr_matrix,
        mask=display_mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=True,
        yticklabels=True,
        cbar=True,
        linewidths=0.5,
        linecolor="lightgray"
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for b in bounds:
        ax.axhline(b, color="black", linewidth=2)
        ax.axvline(b, color="black", linewidth=2)

    structure = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        dtype=int
    )
    labeled, _ = label(sig_offblock, structure=structure)

    for sl in find_objects(labeled):
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        ax.add_patch(
            Rectangle(
                (c0, r0),
                c1 - c0,
                r1 - r0,
                fill=False,
                edgecolor="black",
                linewidth=2,
                linestyle="--"
            )
        )

    terminal_negative_links = find_terminal_negative_cross_group_links(
        num=num,
        dendrites=dendrites,
        alpha=0.05
    )

    overlay_terminal_negative_highlights(
        ax=ax,
        order=order,
        dendrites=dendrites,
        terminal_negative_links=terminal_negative_links,
        color="red",
        cell_linewidth=3.0,
        bracket_linewidth=2.5,
        bracket_frac=0.22,
        label_color="#8B0000",
        tint_terminal_labels=True
    )

    plt.tight_layout()
    plt.show()

    chain_links = []
    for i, chain in enumerate(dendrites, start=1):
        for a, b in zip(chain[:-1], chain[1:]):
            r, p = pearsonr(num[a], num[b])
            chain_links.append(
                {
                    "chain": i,
                    "from": a,
                    "to": b,
                    "r": r,
                    "p": p
                }
            )

    chain_links_df = pd.DataFrame(chain_links)

    print("\nBackbone chain links:")
    print(chain_links_df.to_string(index=False))

    if not terminal_negative_links.empty:
        print("\nTerminal strongest negative cross-group links:")
        print(terminal_negative_links.to_string(index=False))
    else:
        print("\nNo significant negative cross-group links found for chain terminals.")

    G, cross_group_summary = plot_correlation_graph(
        df,
        dendrites,
        alpha=0.05,
        show_cross_group_summary=True
    )
    return G, cross_group_summary, terminal_negative_links


main()
```

## Transcription note

Every block was matched against line-numbered photo crops from at least one pass;
most against two independent passes (original grid photo + your sharper follow-up
set). The four edge-cut lines (**152–155**, the `backbone` / `within` / `cross_strong`
/ `cross_other` list comprehensions building the four edge-style buckets for the
network plot) were only visible in the lower-resolution first photo, cut off at the
right margin in both passes. They're reconstructed here from the immediately
following code (`edge_colors`, `edge_widths`, and the four `draw_networkx_edges`
calls that consume exactly these four names) — the partition logic is unambiguous
given that usage, but flagging it as inferred rather than pixel-confirmed, per your
claim-grounding convention. [empirical:uncited on the exact bracket syntax of those
4 lines; high-confidence inference from downstream usage]