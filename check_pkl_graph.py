"""
check_pkl_graph.py
==================
Thoroughly inspect the SOTA PKL file and understand how the graph is built.
Checks:
1. All columns in the PKL
2. Atom ordering vs molecule_graph
3. What extra_feat_bond_indices_qtaim actually contains
4. Bond distances to verify if they are real BCPs
5. How MoleculeWrapper uses these to build the DGL graph
"""
import pandas as pd, ast, numpy as np, json, os

# ── use test PKL (smaller) ────────────────────────────────────────
PKL = "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/data_suba/test_qm9_qtaim_1205_labelled_corrected.pkl"

print(f"Loading {PKL}...")
df = pd.read_pickle(PKL)
print(f"Rows: {len(df)}")
print(f"\nAll columns:")
for c in df.columns:
    print(f"  {c}")

# ── pick first molecule ───────────────────────────────────────────
row = df.iloc[0]
mol_name = row["names"]
print(f"\n{'='*60}")
print(f"Molecule: {mol_name}")
print(f"{'='*60}")

mol       = row["molecule"]
pm_syms   = [str(s.species.elements[0].symbol) for s in mol.sites]
pm_coords = np.array([[s.x,s.y,s.z] for s in mol.sites])
n_atoms   = len(pm_syms)

# ── atom ordering ─────────────────────────────────────────────────
print(f"\n--- Atom ordering in PKL (pymatgen) ---")
for i,(sym,coord) in enumerate(zip(pm_syms, pm_coords)):
    print(f"  {i:2d}  {sym}  ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})")

# ── molecule_graph edges ──────────────────────────────────────────
mg = row["molecule_graph"]
mg_edges = {(min(u,v),max(u,v)) for u,v in mg.graph.edges()}
print(f"\n--- molecule_graph edges ({len(mg_edges)}) ---")
for i,j in sorted(mg_edges):
    d = np.linalg.norm(pm_coords[i]-pm_coords[j])
    print(f"  ({i:2d},{j:2d})  {pm_syms[i]}-{pm_syms[j]}  dist={d:.3f}Å")

# ── bonds column ──────────────────────────────────────────────────
def parse_bonds(raw):
    s = str(raw).strip()
    if s.startswith("[["): s = s[1:-1]
    try: return [(int(a),int(b)) for a,b in ast.literal_eval(s)]
    except: return []

bonds_col = parse_bonds(row["bonds"])
print(f"\n--- bonds column ({len(bonds_col)} including self-loops) ---")
for i,j in bonds_col:
    if i == j:
        print(f"  ({i:2d},{j:2d})  SELF-LOOP")
    else:
        d = np.linalg.norm(pm_coords[i]-pm_coords[j])
        in_mg = "✓ real" if (min(i,j),max(i,j)) in mg_edges else "✗ NOT real"
        print(f"  ({i:2d},{j:2d})  {pm_syms[i]}-{pm_syms[j]}  dist={d:.3f}Å  {in_mg}")

# ── extra_feat_bond_indices_qtaim ─────────────────────────────────
qtaim_bonds = parse_bonds(row["extra_feat_bond_indices_qtaim"])
print(f"\n--- extra_feat_bond_indices_qtaim ({len(qtaim_bonds)}) ---")
for i,j in qtaim_bonds:
    if i == j:
        print(f"  ({i:2d},{j:2d})  SELF-LOOP")
    else:
        d = np.linalg.norm(pm_coords[i]-pm_coords[j])
        in_mg = "✓ real" if (min(i,j),max(i,j)) in mg_edges else "✗ NOT real"
        print(f"  ({i:2d},{j:2d})  {pm_syms[i]}-{pm_syms[j]}  dist={d:.3f}Å  {in_mg}")

# ── are bonds == extra_feat_bond_indices_qtaim? ───────────────────
print(f"\n--- bonds == extra_feat_bond_indices_qtaim? ---")
print(f"  bonds col: {bonds_col}")
print(f"  qtaim col: {qtaim_bonds}")
print(f"  identical: {bonds_col == qtaim_bonds}")

# ── check bonds_original ─────────────────────────────────────────
if "bonds_original" in df.columns:
    orig = parse_bonds(row["bonds_original"])
    print(f"\n--- bonds_original ({len(orig)}) ---")
    for i,j in orig:
        if i!=j:
            d = np.linalg.norm(pm_coords[i]-pm_coords[j])
            in_mg = "✓" if (min(i,j),max(i,j)) in mg_edges else "✗"
            print(f"  ({i:2d},{j:2d}) {pm_syms[i]}-{pm_syms[j]} dist={d:.3f}Å {in_mg}")

# ── check distance distribution of all bonds ─────────────────────
print(f"\n--- Distance distribution across ALL bonds in PKL ---")
all_dists_real = []
all_dists_nonbond = []
for _, r in df.head(200).iterrows():
    try:
        m      = r["molecule"]
        syms   = [str(s.species.elements[0].symbol) for s in m.sites]
        coords = np.array([[s.x,s.y,s.z] for s in m.sites])
        mge    = {(min(u,v),max(u,v)) for u,v in r["molecule_graph"].graph.edges()}
        qb     = [(i,j) for i,j in parse_bonds(r["extra_feat_bond_indices_qtaim"]) if i!=j]
        for i,j in qb:
            if i < len(coords) and j < len(coords):
                d = np.linalg.norm(coords[i]-coords[j])
                if (min(i,j),max(i,j)) in mge:
                    all_dists_real.append(d)
                else:
                    all_dists_nonbond.append(d)
    except: pass

if all_dists_real:
    print(f"  QTAIM bonds matching real bonds:  "
          f"n={len(all_dists_real)} "
          f"mean={np.mean(all_dists_real):.3f}Å "
          f"min={np.min(all_dists_real):.3f}Å "
          f"max={np.max(all_dists_real):.3f}Å")
if all_dists_nonbond:
    print(f"  QTAIM bonds NOT in molecule_graph: "
          f"n={len(all_dists_nonbond)} "
          f"mean={np.mean(all_dists_nonbond):.3f}Å "
          f"min={np.min(all_dists_nonbond):.3f}Å "
          f"max={np.max(all_dists_nonbond):.3f}Å")

# ── check if bonds with dist ~1.5Å are missing ────────────────────
print(f"\n--- Coverage summary (first 200 molecules) ---")
coverages = []
for _, r in df.head(200).iterrows():
    try:
        m      = r["molecule"]
        coords = np.array([[s.x,s.y,s.z] for s in m.sites])
        mge    = {(min(u,v),max(u,v)) for u,v in r["molecule_graph"].graph.edges()}
        qb     = {(min(i,j),max(i,j)) for i,j in
                  parse_bonds(r["extra_feat_bond_indices_qtaim"]) if i!=j}
        cov = len([e for e in qb if e in mge]) / max(len(mge),1)
        coverages.append(cov)
    except: pass
print(f"  mean coverage: {np.mean(coverages)*100:.1f}%")
print(f"  min coverage:  {np.min(coverages)*100:.1f}%")
print(f"  max coverage:  {np.max(coverages)*100:.1f}%")
print(f"  >80% coverage: {100*np.mean(np.array(coverages)>0.8):.0f}% of molecules")