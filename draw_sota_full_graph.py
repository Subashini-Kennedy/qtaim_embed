"""
draw_sota_full_graph.py
=======================
Draw the SOTA DGL graph as it is ACTUALLY constructed —
showing all edge types including the global node connections.

Edge types:
  a2b / b2a  — atom ↔ bond (QTAIM BCPs only)
  a2g / g2a  — atom ↔ global (ALL atoms)
  b2g / g2b  — bond ↔ global (ALL bonds)
  a2a        — atom self-loops
"""
import pandas as pd, json, ast, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

SOTA_PKL    = "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/data_suba/filtered_qtaim_fullqm9/train_43k.pkl"
JSON_FOLDER = "/lustre/fsn1/projects/rech/ihj/urb54jd/gnn/control_and_critical_points_GNNs/data/criticalpoints_jsonfiles"
OUTDIR      = "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/tensor_viz"

sota_df = pd.read_pickle(SOTA_PKL)
sota_df["gdb_num"] = sota_df["names"].str.extract(r"gdb_(\d+)\.xyz").astype(int)
row = sota_df[sota_df["gdb_num"] == 78284].iloc[0]

mol       = row["molecule"]
pm_syms   = [str(s.species.elements[0].symbol) for s in mol.sites]
pm_coords = np.array([[s.x, s.y, s.z] for s in mol.sites])
n_atoms   = len(pm_syms)

def parse_bonds(raw):
    s = str(raw).strip()
    if s.startswith("[["): s = s[1:-1]
    try: return [(int(a),int(b)) for a,b in ast.literal_eval(s)]
    except: return []

qtaim_all   = parse_bonds(row["extra_feat_bond_indices_qtaim"])
qtaim_bonds = [(i,j) for i,j in qtaim_all if i != j]  # no self-loops
mg_edges    = {(min(u,v),max(u,v)) for u,v in row["molecule_graph"].graph.edges()}

COLORS = {"C":"#444441","O":"#D85A30","N":"#185FA5","H":"#B4B2A9","F":"#1D9E75"}

# 2D layout: atoms at their x,y coords, global node at centre
pos_atoms  = {i: (pm_coords[i,0], pm_coords[i,1]) for i in range(n_atoms)}

# bond node positions = midpoint of their two atoms
bond_positions = []
for i,j in qtaim_bonds:
    mid = ((pm_coords[i,0]+pm_coords[j,0])/2,
           (pm_coords[i,1]+pm_coords[j,1])/2)
    bond_positions.append(mid)

# global node at centroid of heavy atoms
heavy = [i for i in range(n_atoms) if pm_syms[i] != "H"]
gx = np.mean([pm_coords[i,0] for i in heavy])
gy = np.mean([pm_coords[i,1] for i in heavy])
global_pos = (gx, gy)

# ── draw ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

def draw_panel(ax, show_global, title):
    ax.set_title(title, fontsize=10, pad=10)

    # draw a2b/b2a edges (QTAIM bonds)
    for bi, (i,j) in enumerate(qtaim_bonds):
        bpos = bond_positions[bi]
        color = "#1D9E75" if (min(i,j),max(i,j)) in mg_edges else "#D85A30"
        # atom i → bond node
        ax.plot([pos_atoms[i][0], bpos[0]],
                [pos_atoms[i][1], bpos[1]],
                color=color, linewidth=1.5, zorder=2, alpha=0.8)
        # atom j → bond node
        ax.plot([pos_atoms[j][0], bpos[0]],
                [pos_atoms[j][1], bpos[1]],
                color=color, linewidth=1.5, zorder=2, alpha=0.8)

    if show_global:
        # draw a2g edges (all atoms → global)
        for i in range(n_atoms):
            ax.plot([pos_atoms[i][0], global_pos[0]],
                    [pos_atoms[i][1], global_pos[1]],
                    color="#B5D4F4", linewidth=0.7,
                    linestyle="dotted", zorder=1, alpha=0.6)
        # draw b2g edges (all bonds → global)
        for bpos in bond_positions:
            ax.plot([bpos[0], global_pos[0]],
                    [bpos[1], global_pos[1]],
                    color="#FAC775", linewidth=0.6,
                    linestyle="dotted", zorder=1, alpha=0.5)
        # draw global node
        ax.scatter(*global_pos, c="#534AB7", s=1200, zorder=6, marker="s")
        ax.text(*global_pos, "G\nglobal", ha="center", va="center",
                fontsize=7, color="white", zorder=7, fontweight="bold")

    # draw bond nodes
    for bi, (i,j) in enumerate(qtaim_bonds):
        bpos = bond_positions[bi]
        color = "#1D9E75" if (min(i,j),max(i,j)) in mg_edges else "#D85A30"
        ax.plot(*bpos, "D", color=color, markersize=8, zorder=4)
        ax.text(bpos[0]+0.06, bpos[1]+0.06, f"B{bi}",
                fontsize=5, color="#412402", zorder=5)

    # draw atom nodes
    for i in range(n_atoms):
        c = COLORS.get(pm_syms[i], "#888780")
        s = 900 if pm_syms[i] != "H" else 600
        ax.scatter(*pos_atoms[i], c=c, s=s, zorder=5)
        ax.text(*pos_atoms[i], f"{i}\n{pm_syms[i]}",
                ha="center", va="center", fontsize=6,
                color="white", zorder=6, fontweight="bold")

    ax.axis("off")

draw_panel(axes[0], show_global=False,
           title="SOTA graph — atom-bond edges only (a2b/b2a)\n"
                 "green=real bond, red=non-bonded BCP, diamonds=bond nodes\n"
                 "atoms with no diamonds appear isolated here...")

draw_panel(axes[1], show_global=True,
           title="SOTA graph — FULL (a2b/b2a + a2g/g2a + b2g/g2b)\n"
                 "blue dotted = atom→global, gold dotted = bond→global\n"
                 "ALL atoms connect through the global node (purple square)")

leg = [
    mpatches.Patch(color="#444441", label="C atom"),
    mpatches.Patch(color="#D85A30", label="O atom"),
    mpatches.Patch(color="#185FA5", label="N atom"),
    mpatches.Patch(color="#B4B2A9", label="H atom"),
    mpatches.Patch(color="#1D9E75", label="bond node (real covalent BCP)"),
    mpatches.Patch(color="#D85A30", label="bond node (non-covalent BCP)"),
    mpatches.Patch(color="#534AB7", label="global node"),
    mpatches.Patch(color="#B5D4F4", label="atom→global edge"),
    mpatches.Patch(color="#FAC775", label="bond→global edge"),
]
fig.legend(handles=leg, loc="lower center", ncol=3,
           fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("gdb_78284 — SOTA DGL graph: what looks disconnected is connected via global node",
             fontsize=11)
fig.tight_layout()
out = f"{OUTDIR}/gdb_78284_sota_full_graph.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {out}")
print(f"\nKey insight:")
print(f"  Left panel: only shows a2b/b2a — atoms without QTAIM bonds look isolated")
print(f"  Right panel: shows ALL edges — every atom connects to global node")
print(f"  The global node aggregates info from ALL atoms and passes it back")
print(f"  So 'isolated' atoms still receive information via global→atom messages")