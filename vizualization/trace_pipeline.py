"""
trace_pipeline.py
-----------------
Step-by-step trace of the full molwrapper → featurizer → grapher pipeline
for a single molecule (dsgdb9nsd_021159).

Run on Jean-Zay:
    PYTHONPATH=/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private \
    /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/env/qtaim_embed/bin/python \
    trace_pipeline.py

Edit the constants below to match your paths.
"""

import sys
import numpy as np
import pandas as pd

# ── CONFIG ──────────
PKL_PATH   = "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/data_suba/filtered_qtaim_fullqm9/train_43k.pkl"
MOL_FILTER = "21159"   # substring of names column
BOND_KEY   = "bonds"
MAP_KEY    = "extra_feat_bond_indices_qtaim"

ATOM_KEYS = [
    "extra_feat_atom_Lagrangian_K",
    "extra_feat_atom_Hamiltonian_K",
    "extra_feat_atom_e_density",
    "extra_feat_atom_lap_e_density",
    "extra_feat_atom_e_loc_func",
    "extra_feat_atom_ave_loc_ion_E",
    "extra_feat_atom_delta_g_promolecular",
    "extra_feat_atom_delta_g_hirsh",
    "extra_feat_atom_esp_total",
    "extra_feat_atom_eta",
    "extra_feat_atom_lol",
]

BOND_KEYS = [
    "extra_feat_bond_Lagrangian_K",
    "extra_feat_bond_Hamiltonian_K",
    "extra_feat_bond_e_density",
    "extra_feat_bond_lap_e_density",
    "extra_feat_bond_e_loc_func",
    "extra_feat_bond_ave_loc_ion_E",
    "extra_feat_bond_delta_g_promolecular",
    "extra_feat_bond_delta_g_hirsh",
    "extra_feat_bond_esp_total",
    "extra_feat_bond_eta",
    "extra_feat_bond_lol",
]

ALLOWED_RING_SIZE = [3, 4, 5, 6, 7]
ELEMENT_SET = ["C", "F", "H", "N", "O"]

# ── HELPERS ──────────────────
SEP  = "=" * 72
SEP2 = "-" * 72

def hdr(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def sub(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — load raw PKL row
# ══════════════════════════════════════════════════════════════════════════════
hdr("STAGE 0 — Load raw PKL row")

print(f"Loading {PKL_PATH} ...")
df = pd.read_pickle(PKL_PATH)
row = df[df["names"].str.contains(MOL_FILTER)].iloc[0]
print(f"Found molecule: names={row['names']}  ids={row['ids']}")

# Ground-truth graph
mg  = row["molecule_graph"]
mol_pmg = row["molecule"]

# Species lookup — works regardless of pymatgen version
def get_species(site):
    for attr in ("specie", "species_string"):
        try:
            v = getattr(site, attr)
            if hasattr(v, "symbol"):
                return v.symbol
            return str(v).split(":")[0]
        except Exception:
            pass
    return str(list(site.species.as_dict().keys())[0]).split(":")[0]

syms = [get_species(s) for s in mol_pmg.sites]
coords = [s.coords.tolist() for s in mol_pmg.sites]
n_atoms = len(syms)

print(f"\nAtoms ({n_atoms}):")
for i, (sp, xyz) in enumerate(zip(syms, coords)):
    print(f"  {i:2d}  {sp:2s}  {[round(x,3) for x in xyz]}")

mg_edges = sorted({(min(u,v), max(u,v)) for u,v,_ in mg.graph.edges(data=True)})
print(f"\nmolecule_graph edges ({len(mg_edges)} — GROUND TRUTH):")
for u, v in mg_edges:
    print(f"  ({u:2d},{v:2d})  {syms[u]}-{syms[v]}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — mol_wrappers_from_df  (molwrapper.py)
# ══════════════════════════════════════════════════════════════════════════════
hdr("STAGE 1 — mol_wrappers_from_df  (molwrapper.py)")

# ── 1a. read bonds column ────────────────────────────────────────────────────
sub("1a. Read bonds column")
raw_bonds = row[BOND_KEY]
print(f"row['{BOND_KEY}'] type  : {type(raw_bonds)}")
print(f"row['{BOND_KEY}'] length: {len(raw_bonds)}")
print(f"row['{BOND_KEY}'] value : {raw_bonds}")

# ── 1b. unwrap outer list ────────────────────────────────────────────────────
sub("1b. Unwrap outer list  (if len == 1)")
if len(raw_bonds) == 1:
    bonds_unwrapped = raw_bonds[0]
    print("len == 1  →  bonds = raw_bonds[0]")
else:
    bonds_unwrapped = raw_bonds
    print("len > 1  →  bonds = raw_bonds (no unwrap)")
print(f"After unwrap ({len(bonds_unwrapped)} pairs): {bonds_unwrapped}")

# ── 1c. filter self-bonds ────────────────────────────────────────────────────
sub("1c. filter_self_bonds=True  →  remove pairs where b[0]==b[1]")
self_loops = [b for b in bonds_unwrapped if b[0] == b[1]]
bonds_filtered = {tuple(sorted(b)): None for b in bonds_unwrapped if b[0] != b[1]}
print(f"Self-loops removed: {self_loops}")
print(f"Bonds after filter ({len(bonds_filtered)}):")
for b in sorted(bonds_filtered.keys()):
    in_mg = "✓ real" if b in mg_edges else "✗ phantom/wrong-index"
    print(f"  {b}  {syms[b[0]]}-{syms[b[1]]}  {in_mg}")

missing_from_bonds = [e for e in mg_edges if e not in bonds_filtered]
print(f"\nREAL bonds MISSING from bonds_filtered ({len(missing_from_bonds)}):")
for u, v in missing_from_bonds:
    print(f"  ({u:2d},{v:2d})  {syms[u]}-{syms[v]}  ← ABSENT from QTAIM BCP list")

# ── 1d. get_bond_features ────────────────────────────────────────────────────
sub("1d. get_bond_features  — build feature dict keyed by bond tuple")
print(f"map_key = '{MAP_KEY}'  (parallel index list for bond feature arrays)")

# Parse QTAIM bond index list
qtaim_idx_raw = row[MAP_KEY]
# normalise to list of (i,j)
if isinstance(qtaim_idx_raw, list) and len(qtaim_idx_raw) > 0:
    first = qtaim_idx_raw[0]
    if isinstance(first, (list, tuple)) and len(first) == 2:
        qtaim_pairs = [tuple(x) for x in qtaim_idx_raw]
    else:
        qtaim_pairs = [tuple(x) for x in qtaim_idx_raw]
else:
    qtaim_pairs = list(qtaim_idx_raw)

print(f"extra_feat_bond_indices_qtaim ({len(qtaim_pairs)} pairs): {qtaim_pairs}")

# Show what feature lookup looks like for one bond key
sample_key = BOND_KEYS[0]  # e.g. extra_feat_bond_Lagrangian_K
raw_feat = row[sample_key]
if hasattr(raw_feat, "__iter__"):
    feat_arr = list(raw_feat)
else:
    feat_arr = [raw_feat]

print(f"\nExample: {sample_key}")
print(f"  array length: {len(feat_arr)}")
print(f"  value[0] → bond {qtaim_pairs[0]}: {feat_arr[0]:.4f}")
print(f"  value[1] → bond {qtaim_pairs[1]}: {feat_arr[1]:.4f}")

# Build bond_feats dict (simulating get_bond_features)
bond_feats = {}
for k, bpair in enumerate(qtaim_pairs):
    bkey = tuple(sorted(bpair))
    if bkey[0] == bkey[1]:
        print(f"  SKIP self-loop {bkey}")
        continue
    bond_feats[bkey] = {}
    for feat_col in BOND_KEYS:
        raw = row[feat_col]
        if hasattr(raw, "__iter__"):
            val = list(raw)[k]
        else:
            val = raw
        bond_feats[bkey][feat_col] = float(val)

print(f"\nbond_feats dict keys ({len(bond_feats)}):")
for b, fdict in sorted(bond_feats.items()):
    sample_val = list(fdict.values())[0]
    print(f"  {b}  {syms[b[0]]}-{syms[b[1]]}  first_feat={sample_val:.4f}")

# ── 1e. get_atom_feats ────────────────────────────────────────────────────────
sub("1e. get_atom_feats  — build feature dict keyed by atom index")
atom_feats = {}
for i in range(n_atoms):
    atom_feats[i] = {}
    for feat_col in ATOM_KEYS:
        raw = row[feat_col]
        if hasattr(raw, "__iter__"):
            val = list(raw)[i]
        else:
            val = raw
        atom_feats[i][feat_col] = float(val)

print(f"atom_feats dict keys (0..{n_atoms-1}), showing first feat per atom:")
for i in range(n_atoms):
    v = list(atom_feats[i].values())[0]
    print(f"  atom {i:2d} ({syms[i]:2s})  {ATOM_KEYS[0].replace('extra_feat_atom_','')}={v:.4f}")

# ── 1f. MoleculeWrapper construction ─────────────────────────────────────────
sub("1f. MoleculeWrapper construction")
print("MoleculeWrapper(")
print(f"  mol_graph       = molecule_graph  (pymatgen MoleculeGraph)")
print(f"  bonds           = {sorted(bonds_filtered.keys())}  ← {len(bonds_filtered)} pairs")
print(f"  non_metal_bonds = same as bonds")
print(f"  atom_features   = dict[int → dict]  ({n_atoms} atoms)")
print(f"  bond_features   = dict[tuple → dict]  ({len(bond_feats)} bonds)")
print(")")
print()
print("KEY POINT: mol.bonds is set to the FILTERED bonds_filtered dict.")
print(f"  → {len(missing_from_bonds)} real bonds are ABSENT from mol.bonds")
print(f"  Missing: {missing_from_bonds}")

# ── Simulate MoleculeWrapper attributes ──────────────────────────────────────
class FakeMolWrapper:
    def __init__(self):
        self.bonds = bonds_filtered
        self.bond_features = bond_feats
        self.atom_features = atom_feats
        self.coords = coords
        self.species = syms
        self.num_atoms = n_atoms
        self.composition_dict = {}
        for sp in syms:
            self.composition_dict[sp] = self.composition_dict.get(sp, 0) + 1
        self.id = f"123092_{row['names']}"
        self.global_features = {"charge": 0, "spin": 1}

mol_wrapper = FakeMolWrapper()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — HeteroCompleteGraphFromMolWrapper.build_graph  (grapher.py)
# ══════════════════════════════════════════════════════════════════════════════
hdr("STAGE 2 — HeteroCompleteGraphFromMolWrapper.build_graph  (grapher.py)")

bond_list = list(mol_wrapper.bonds.keys())
num_bonds = len(bond_list)
num_atoms_g = len(mol_wrapper.coords)

sub("2a. Bond and atom counts")
print(f"num_bonds = len(mol.bonds.keys()) = {num_bonds}")
print(f"num_atoms = len(mol.coords)       = {num_atoms_g}")
print()
print(f"bond_list = {bond_list}")

sub("2b. Build a2b / b2a edges (atom↔bond connections)")
a2b = []
b2a = []
for b_idx in range(num_bonds):
    u = bond_list[b_idx][0]
    v = bond_list[b_idx][1]
    b2a.extend([[b_idx, u], [b_idx, v]])
    a2b.extend([[u, b_idx], [v, b_idx]])

print(f"{'Bond idx':>8}  {'Pair':>10}  {'Species':>6}  a2b edges added        b2a edges added")
for b_idx in range(num_bonds):
    u, v = bond_list[b_idx]
    print(f"  b{b_idx:2d}      ({u:2d},{v:2d})     {syms[u]}-{syms[v]}   "
          f"[({u},{b_idx}),({v},{b_idx})]          [({b_idx},{u}),({b_idx},{v})]")

sub("2c. Check which atoms appear in a2b (have at least one bond node)")
atoms_in_a2b = set()
for atom_idx, _ in a2b:
    atoms_in_a2b.add(atom_idx)

print(f"Atoms appearing in a2b edges: {sorted(atoms_in_a2b)}")
disconnected = [i for i in range(num_atoms_g) if i not in atoms_in_a2b]
print(f"\nDISCONNECTED atoms (no bond node): {disconnected}")
for i in disconnected:
    # find their real bonds
    real_bonds_for_atom = [(u,v) for u,v in mg_edges if u==i or v==i]
    print(f"  atom {i} ({syms[i]}): real bonds = {real_bonds_for_atom}  → ALL MISSING from graph")

sub("2d. a2g / g2a edges (every atom connects to global node)")
a2g = [(a, 0) for a in range(num_atoms_g)]
g2a = [(0, a) for a in range(num_atoms_g)]
print(f"a2g: {a2g}")
print(f"NOTE: disconnected atoms still appear here — they connect to global,")
print(f"      but have NO bond node, so their local bonding info is lost.")

sub("2e. Full edges_dict summary")
b2g = [(b, 0) for b in range(num_bonds)]
g2b = [(0, b) for b in range(num_bonds)]
a2a = [(i, i) for i in range(num_atoms_g)]
b2b = [(i, i) for i in range(num_bonds)]
print(f"  ('atom','a2b','bond')   : {len(a2b)} edges")
print(f"  ('bond','b2a','atom')   : {len(b2a)} edges")
print(f"  ('atom','a2g','global') : {len(a2g)} edges")
print(f"  ('global','g2a','atom') : {len(g2a)} edges")
print(f"  ('bond','b2g','global') : {len(b2g)} edges")
print(f"  ('global','g2b','bond') : {len(g2b)} edges")
print(f"  ('atom','a2a','atom')   : {len(a2a)} self-loop edges")
print(f"  ('bond','b2b','bond')   : {len(b2b)} self-loop edges")
print(f"  ('global','g2g','global'): 1 self-loop edge")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — AtomFeaturizerGraphGeneral  (featurizer.py)
# ══════════════════════════════════════════════════════════════════════════════
hdr("STAGE 3 — AtomFeaturizerGraphGeneral  (featurizer.py)")

sub("3a. Inputs to featurizer")
print(f"mol.species   = {mol_wrapper.species}")
print(f"mol.bonds     = {sorted(mol_wrapper.bonds.keys())}  ← used for degree/h_count/rings")
print(f"mol.atom_features = dict[0..{n_atoms-1}]")
print(f"allowed_ring_size = {ALLOWED_RING_SIZE}")
print(f"element_set       = {ELEMENT_SET}")

sub("3b. h_count_and_degree  — per atom (simulated)")
print(f"{'Atom':>5}  {'Sp':>3}  {'degree':>6}  {'h_count':>7}  {'bond_list_entries':>40}  note")

def simulated_h_count_and_degree(atom_idx, bl, species):
    """Simulate qtaim_embed's h_count_and_degree."""
    neighbors = []
    for bond in bl:
        if bond[0] == atom_idx:
            neighbors.append(bond[1])
        elif bond[1] == atom_idx:
            neighbors.append(bond[0])
    degree = len(neighbors)
    h_count = sum(1 for n in neighbors if species[n] == "H")
    return h_count, degree

bl = [list(b) for b in bond_list]
for i in range(n_atoms):
    h_cnt, deg = simulated_h_count_and_degree(i, bl, syms)
    entries = [(b[0], b[1]) for b in bl if i in b]
    
    # What is the REAL degree from molecule_graph?
    real_neighbors = [v if u==i else u for u,v in mg_edges if u==i or v==i]
    real_deg = len(real_neighbors)
    real_h = sum(1 for n in real_neighbors if syms[n] == "H")
    
    wrong = "  ← WRONG" if deg != real_deg else ""
    print(f"  {i:3d}  {syms[i]:>3}  {deg:>6}  {h_cnt:>7}  {str(entries):>40}{wrong}")
    if wrong:
        print(f"        real_degree={real_deg}, real_h_count={real_h}")

sub("3c. Ring detection  — find_rings(num_atoms, bond_list, allowed_ring_size, edges=False)")
print("Ring detection uses ONLY the 11-bond bond_list (missing C2's bonds).")
print("Rings involving C2 will NOT be found.")
print()
print("SMILES: CNc1oncc1N  → has a 5-membered ring: N-C-C-N-O")
print("Ring atoms in molecule_graph: look for the isoxazole ring")
print()

# Find ring atoms from molecule_graph
# The ring is the 5-membered heterocycle: find it from mg_edges
from collections import defaultdict as ddict
adj = ddict(set)
for u, v in mg_edges:
    adj[u].add(v)
    adj[v].add(u)

# simple DFS cycle finder for small rings
def find_all_rings(adj, n):
    rings = []
    visited = [False]*n
    def dfs(start, curr, path, depth):
        for nb in adj[curr]:
            if nb == start and depth >= 2:
                rings.append(tuple(sorted(path)))
                return
            if not visited[nb] and nb > start:
                visited[nb] = True
                path.append(nb)
                dfs(start, nb, path, depth+1)
                path.pop()
                visited[nb] = False
    for s in range(n):
        visited[s] = True
        dfs(s, s, [s], 0)
    return list(set(rings))

real_rings = find_all_rings(adj, n_atoms)
real_rings_5 = [r for r in real_rings if len(r) == 5]
print(f"Real 5-membered rings from molecule_graph: {real_rings_5}")
for r in real_rings_5:
    print(f"  atoms {r}  species: {[syms[i] for i in r]}")

print()
print("Now checking if bond_list (11 bonds, missing C2) can find the same rings:")
adj_qtaim = ddict(set)
for b in bl:
    adj_qtaim[b[0]].add(b[1])
    adj_qtaim[b[1]].add(b[0])
qtaim_rings = find_all_rings(adj_qtaim, n_atoms)
qtaim_rings_5 = [r for r in qtaim_rings if len(r) == 5]
print(f"5-membered rings from QTAIM bond_list: {qtaim_rings_5}")
if not qtaim_rings_5:
    print("  → NONE FOUND. Ring features for ring atoms will be wrong (0s).")

sub("3d. Full atom feature vector  (for each atom)")
print(f"Feature layout per atom:")
print(f"  [0]     total_degree          (from bond_list)")
print(f"  [1]     total_H               (from bond_list)")
print(f"  [2]     is_in_ring            (from find_rings on bond_list)")
print(f"  [3-7]   ring_size_3/4/5/6/7   (from find_rings on bond_list)")
print(f"  [8-12]  one-hot: C/F/H/N/O")
print(f"  [13+]   QTAIM atom features   ({len(ATOM_KEYS)} keys)")
print(f"  Total feature dim: 2 + 1 + {len(ALLOWED_RING_SIZE)} + {len(ELEMENT_SET)} + {len(ATOM_KEYS)} = {2+1+len(ALLOWED_RING_SIZE)+len(ELEMENT_SET)+len(ATOM_KEYS)}")
print()

for i in range(n_atoms):
    h_cnt, deg = simulated_h_count_and_degree(i, bl, syms)
    one_hot = [1 if e == syms[i] else 0 for e in ELEMENT_SET]
    ring_inc = 0  # from qtaim rings — all 0 for our case
    ring_sizes = [0]*len(ALLOWED_RING_SIZE)
    # check qtaim rings
    for r in qtaim_rings:
        if i in r:
            ring_inc = 1
            sz = len(r)
            if sz in ALLOWED_RING_SIZE:
                ring_sizes[ALLOWED_RING_SIZE.index(sz)] = 1

    qtaim_vals = [round(list(atom_feats[i].values())[j], 4) for j in range(min(3, len(ATOM_KEYS)))]
    feat_vec = [deg, h_cnt, ring_inc] + ring_sizes + one_hot
    
    # Real values
    real_neighbors = [v if u==i else u for u,v in mg_edges if u==i or v==i]
    real_deg = len(real_neighbors)
    real_h = sum(1 for n in real_neighbors if syms[n] == "H")
    real_ring_inc = 1 if any(i in r for r in real_rings) else 0
    
    issues = []
    if deg != real_deg:        issues.append(f"degree:{deg}≠{real_deg}")
    if h_cnt != real_h:        issues.append(f"h_count:{h_cnt}≠{real_h}")
    if ring_inc != real_ring_inc: issues.append(f"ring_inc:{ring_inc}≠{real_ring_inc}")
    flag = "  ← WRONG: " + ", ".join(issues) if issues else "  ✓"
    
    print(f"  atom {i:2d} ({syms[i]:2s}):  feat={feat_vec}  qtaim[0:3]={qtaim_vals}{flag}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — BondAsNodeGraphFeaturizerGeneral  (featurizer.py)
# ══════════════════════════════════════════════════════════════════════════════
hdr("STAGE 4 — BondAsNodeGraphFeaturizerGeneral  (featurizer.py)")

sub("4a. Inputs to bond featurizer")
print(f"bond_list = {bond_list}")
print(f"mol.bond_features = dict keyed by {list(bond_feats.keys())}")

sub("4b. Ring features for bonds  — ring_features_for_bonds_full")
print("Same issue: find_rings uses 11-bond bond_list → wrong rings")

sub("4c. Full bond feature vector  (for each bond node)")
print(f"Feature layout per bond:")
print(f"  [0]     metal_bond            (0 for organic)")
print(f"  [1]     ring_inclusion        (from find_rings on bond_list)")
print(f"  [2-6]   ring_size_3/4/5/6/7   (from find_rings on bond_list)")
print(f"  [7]     bond_length           (from coords)")
print(f"  [8+]    QTAIM bond features   ({len(BOND_KEYS)} keys)")
print(f"  Total feature dim: 1 + 1 + {len(ALLOWED_RING_SIZE)} + 1 + {len(BOND_KEYS)} = {1+1+len(ALLOWED_RING_SIZE)+1+len(BOND_KEYS)}")
print()

for b_idx, bond in enumerate(bond_list):
    u, v = bond
    bl_vec = np.array(coords[u]) - np.array(coords[v])
    bond_len = float(np.sqrt(np.sum(bl_vec**2)))
    qtaim_vals = [round(list(bond_feats[bond].values())[j], 4) for j in range(min(3, len(BOND_KEYS)))]
    
    ring_inc = 0
    ring_sizes = [0]*len(ALLOWED_RING_SIZE)
    for r_edges in qtaim_rings:
        pass  # would check edge-based rings; simplified here
    
    feat_vec = [0, ring_inc] + ring_sizes + [round(bond_len, 4)]
    print(f"  b{b_idx:2d} ({u:2d},{v:2d}) {syms[u]}-{syms[v]}:  topo={feat_vec}  qtaim[0:3]={qtaim_vals}")

sub("4d. Missing bond nodes")
print(f"The following real bonds have NO bond node in the graph:")
for u, v in missing_from_bonds:
    print(f"  ({u:2d},{v:2d})  {syms[u]}-{syms[v]}  ← no node, no features, not in any edge")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Final DGL graph structure summary
# ══════════════════════════════════════════════════════════════════════════════
hdr("STAGE 5 — Final DGL graph summary")

print(f"Node types and counts:")
print(f"  atom   : {n_atoms} nodes  (correct)")
print(f"  bond   : {num_bonds} nodes  (should be {len(mg_edges)}  — missing {len(mg_edges)-num_bonds})")
print(f"  global : 1 node")
print()
print(f"Edge counts:")
print(f"  a2b : {len(a2b):3d}  (atom→bond)")
print(f"  b2a : {len(b2a):3d}  (bond→atom)")
print(f"  a2g : {len(a2g):3d}  (atom→global)")
print(f"  g2a : {len(g2a):3d}  (global→atom)")
print(f"  b2g : {len(b2g):3d}  (bond→global)")
print(f"  g2b : {len(g2b):3d}  (global→bond)")
print()
print(f"Disconnected atoms (in graph but no bond node):")
for i in disconnected:
    print(f"  atom {i} ({syms[i]}):  appears in a2g/g2a only")
    print(f"    real bonds: {[(u,v) for u,v in mg_edges if u==i or v==i]}")
print()
print(f"Phantom bond nodes (bond node exists but pair not in molecule_graph):")
phantoms = [b for b in bond_list if b not in mg_edges]
for b in phantoms:
    print(f"  {b}  {syms[b[0]]}-{syms[b[1]]}  ← not a real covalent bond")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — molecule_graph as topology
# ══════════════════════════════════════════════════════════════════════════════
hdr("STAGE 6 — The fix: use molecule_graph as bond topology")

print("""
The fix strategy:
  1. Bond TOPOLOGY  ← molecule_graph.edges()   (always complete, correct)
  2. Bond FEATURES  ← QTAIM BCP list           (fill zeros for missing bonds)

For each bond in molecule_graph:
  - if bond in QTAIM BCP list  → use QTAIM features (19 values)
  - if bond NOT in QTAIM list  → use zeros (19 values)
""")

fixed_bonds = {(min(u,v), max(u,v)): None for u,v in mg_edges}
fixed_bond_feats = {}
n_zero_bonds = 0
for u, v in mg_edges:
    key = (min(u,v), max(u,v))
    if key in bond_feats:
        fixed_bond_feats[key] = bond_feats[key]
        status = "✓ QTAIM features"
    else:
        fixed_bond_feats[key] = {k: 0.0 for k in BOND_KEYS}
        status = "  ZERO-filled"
        n_zero_bonds += 1
    print(f"  ({u:2d},{v:2d})  {syms[u]}-{syms[v]}:  {status}")

print(f"\nResult: {len(fixed_bonds)} bond nodes  ({n_zero_bonds} zero-filled, {len(fixed_bonds)-n_zero_bonds} have QTAIM features)")
print(f"All {n_atoms} atoms are now connected through their correct bonds.")
print()
print("To implement: in mol_wrappers_from_df, replace:")
print("  bonds = row[bond_key]")
print("with:")
print("  mg    = row['molecule_graph']")
print("  bonds = [(min(u,v),max(u,v)) for u,v,_ in mg.graph.edges(data=True)]")
print("  # then zero-fill bond_feats for bonds not in QTAIM BCP list")

print(f"\n{SEP}")
print("  DONE — full trace complete")
print(SEP)
