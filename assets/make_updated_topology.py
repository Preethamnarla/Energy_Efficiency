# make_updated_topology.py
import os, math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 10,
})

def hex_positions(R=1.0):
    pos = {0: (0.0, 0.0)}
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for i, ang in enumerate(angles, start=1):
        pos[i] = (R*math.cos(ang), R*math.sin(ang))
    return pos

def draw_base_topology(ax, pos, title):
    G = nx.Graph()
    for n in pos: G.add_node(n)
    for i in range(1, 7): G.add_edge(0, i)

    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color="#94a3b8", alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1200, node_color="#0ea5e9", linewidths=1.5, edgecolors="#075985")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: f"Site {n}" for n in pos}, font_size=10, font_weight="bold", font_color="white")
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_aspect("equal")
    ax.axis("off")

def topology_config_image(path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    pos = hex_positions(R=1.0)
    draw_base_topology(ax, pos, "7-Site Hex Topology (Reuse-1)")

    # Configuration legend (top-right)
    cfg_lines = [
        "Scenario: UMa (7-site hex), reuse-1",
        "Sectors/site: 3 (65° panels), h_gNB ≈ 25 m",
        "Carrier: 3.5 GHz, BW: 40 MHz, TTI: 1 ms",
        "ISD: 500 m; h_UE: 1.5 m",
        "Tx power: 40 W/sector (46 dBm), G_ant ≈ 17 dBi",
        "Energy: P(t) = P0 + α·Load; P0 ≈ 130 W/sector; α ≈ 4–5",
        "Sleep: micro (fast wake), deep (slow wake)",
        "Traffic: diurnal (busy-hour peaks, off-peak valleys)",
        "Interference: reuse-1, neighbor coupling",
    ]
    text = "\n".join("• " + l for l in cfg_lines)
    ax.text(1.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="#111827",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.95, edgecolor="#cbd5e1"))

    ax.text(0.02, 0.02, "Note: All sites use identical configuration.",
            transform=ax.transAxes, fontsize=9, color="#334155",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="#e2e8f0"))

    fig.tight_layout()
    fig.savefig(path, transparent=False)
    plt.close(fig)

def topology_results_image(path):
    # Example per-site results (replace with your measured values)
    # Energy/day in kWh; “delta” auto-computed; QoS deltas shown
    site = {
        0: dict(base=26.4, rl=23.8, micro=12, deep=5,  dlat=+2, dthr=-1.0),
        1: dict(base=22.9, rl=19.1, micro=27, deep=10, dlat=+3, dthr=-1.0),
        2: dict(base=22.7, rl=18.7, micro=29, deep=11, dlat=+3, dthr=-2.0),
        3: dict(base=23.0, rl=19.3, micro=26, deep=10, dlat=+3, dthr=-1.0),
        4: dict(base=22.6, rl=18.9, micro=25, deep=10, dlat=+3, dthr=-1.0),
        5: dict(base=22.8, rl=19.0, micro=28, deep=11, dlat=+3, dthr=-1.5),
        6: dict(base=22.7, rl=18.8, micro=27, deep=10, dlat=+3, dthr=-1.0),
    }

    # Color by energy reduction
    def color_for(reduction_pct):
        if reduction_pct > 15:  # green
            return "#10b981"
        elif reduction_pct >= 10:  # amber
            return "#f59e0b"
        else:  # red
            return "#ef4444"

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    pos = hex_positions(R=1.0)

    # Draw edges first
    G = nx.Graph()
    for n in pos: G.add_node(n)
    for i in range(1, 7): G.add_edge(0, i)
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color="#94a3b8", alpha=0.5)

    # Compute colors
    colors = []
    for n in pos:
        s = site[n]
        red = 100.0 * (s["base"] - s["rl"]) / s["base"]
        colors.append(color_for(red))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1300, node_color=colors, linewidths=1.5, edgecolors="#0f172a")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: f"Site {n}" for n in pos},
                            font_size=10, font_weight="bold", font_color="white")

    # Callout text boxes with leader arrows (offset per site to avoid overlap)
    offsets = {
        0: (0.00, -0.42),
        1: (+0.55, -0.02),
        2: (+0.38, +0.45),
        3: (-0.42, +0.55),
        4: (-0.60, +0.00),
        5: (-0.40, -0.48),
        6: (+0.35, -0.55),
    }

    for n, (x, y) in pos.items():
        s = site[n]
        red = 100.0 * (s["base"] - s["rl"]) / s["base"]
        txt = (
            f"Energy: {s['base']:.1f} → {s['rl']:.1f} kWh  (−{red:.0f}%)\n"
            f"Sleep: micro {s['micro']}% • deep {s['deep']}%\n"
            f"QoS: 95p Lat +{s['dlat']} ms • Thrpt {s['dthr']}%"
        )
        dx, dy = offsets[n]
        ax.annotate(
            txt, xy=(x, y), xycoords="data",
            xytext=(x + dx, y + dy), textcoords="data",
            ha="left" if dx >= 0 else "right", va="center",
            fontsize=9, color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.96, edgecolor="#cbd5e1"),
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.5, shrinkA=8, shrinkB=8),
        )

    # Legend for colors
    patches = [
        mpatches.Patch(color="#10b981", label="> 15% energy reduction"),
        mpatches.Patch(color="#f59e0b", label="10–15% reduction"),
        mpatches.Patch(color="#ef4444", label="< 10% reduction"),
    ]
    leg = ax.legend(handles=patches, loc="upper left", frameon=True, fontsize=9,
                    facecolor="white", edgecolor="#cbd5e1")
    for t in leg.get_texts(): t.set_color("#0f172a")

    ax.set_title("Per-Site Results (Illustrative — replace with your measurements)", fontsize=12, pad=10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, transparent=False)
    plt.close(fig)

if __name__ == "__main__":
    topology_config_image(os.path.join(ASSETS_DIR, "topology_config.png"))
    topology_results_image(os.path.join(ASSETS_DIR, "topology_results.png"))
    print("Saved:\n - assets/topology_config.png\n - assets/topology_results.png")