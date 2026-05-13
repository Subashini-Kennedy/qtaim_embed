"""
trace_pipeline.py
-----------------
Runs the real pipeline on one molecule and prints every intermediate value.
Uses ONLY the actual functions from molwrapper.py, grapher.py, featurizer.py.
No custom logic added.

Run:
    cd /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private
    PYTHONPATH=. env/qtaim_embed/bin/python trace_pipeline.py
"""

import pandas as pd

# ── real imports from the three scripts ──────────────────────────────────────
from qtaim_embed.core.molwrapper import MoleculeWrapper
from qtaim_embed.utils.descriptors import (
    get_atom_feats,
    get_bond_features,
    get_global_features,
    elements_from_pmg,
    h_count_and_degree,
    ring_features_from_atom_full,
    ring_features_for_bonds_full,
    find_rings,
    one_hot_encoding,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PKL_PATH  = "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/data_suba/filtered_qtaim_fullqm9/train_43k.pkl"
#MOL_NAME  = "21159" "01598"
MOL_NAME  = "01598" 
BOND_KEY  = "bonds"
MAP_KEY   = "extra_feat_bond_indices_qtaim"

ATOM_KEYS = [
    "extra_feat_atom_Lagrangian_K", "extra_feat_atom_Hamiltonian_K",
    "extra_feat_atom_e_density",    "extra_feat_atom_lap_e_density",
    "extra_feat_atom_e_loc_func",
]
BOND_KEYS = [
    "extra_feat_bond_Lagrangian_K", "extra_feat_bond_Hamiltonian_K",
    "extra_feat_bond_e_density",    "extra_feat_bond_lap_e_density",
    "extra_feat_bond_e_loc_func",
]
ALLOWED_RING_SIZE = [3, 4, 5, 6, 7]
ELEMENT_SET       = ["C", "F", "H", "N", "O"]

S  = "=" * 70
S2 = "-" * 70
def hdr(t): print(f"\n{S}\n  {t}\n{S}")
def sub(t): print(f"\n{S2}\n  {t}\n{S2}")

# ── load one row ──────────────────────────────────────────────────────────────
hdr("LOAD PKL ROW")
df  = pd.read_pickle(PKL_PATH)
df1 = df[df["names"].str.contains(MOL_NAME)].head(1)
row = df1.iloc[0]
print(f"names={row['names']}  ids={row['ids']}")

# ground-truth bonds for comparison
mg       = row["molecule_graph"]
mg_edges = sorted({(min(u,v), max(u,v)) for u,v,_ in mg.graph.edges(data=True)})
print(f"\nmolecule_graph real bonds ({len(mg_edges)}):")
for e in mg_edges:
    print(f"  {e}")


# ══════════════════════════════════════════════════════════════════════════════
hdr("mol_wrappers_from_df  [molwrapper.py]  — line by line")
# ══════════════════════════════════════════════════════════════════════════════

# line 45: bonds = row[bond_key]
sub("line 45: bonds = row[bond_key]")
bonds = row[BOND_KEY]
print(f"type={type(bonds)}  len={len(bonds)}")
print(f"value={bonds}")

# line 52: bonds = {tuple(sorted(b)): None for b in bonds}
sub("line 52: bonds = {tuple(sorted(b)): None for b in bonds}")
bonds = {tuple(sorted(b)): None for b in bonds}
print(f"bonds dict ({len(bonds)} keys): {list(bonds.keys())}")

# line 59: atom_feats = get_atom_feats(row, atom_keys)
sub("line 59: atom_feats = get_atom_feats(row, atom_keys)")
atom_feats = get_atom_feats(row, ATOM_KEYS)
print(f"type={type(atom_feats)}")
if atom_feats != -1:
    print(f"keys (atom indices): {list(atom_feats.keys())}")
    print(f"atom_feats[0] = {atom_feats[0]}")
    print(f"atom_feats[2] = {atom_feats[2]}")

# line 61: bond_feats = get_bond_features(row, map_key, bond_key, keys)
sub("line 61: bond_feats = get_bond_features(row, map_key=MAP_KEY, bond_key=BOND_KEY, keys=BOND_KEYS)")
bond_feats = get_bond_features(row, map_key=MAP_KEY, bond_key=BOND_KEY, keys=BOND_KEYS)
print(f"type={type(bond_feats)}")
if bond_feats != -1:
    print(f"bond_feats keys ({len(bond_feats)}): {list(bond_feats.keys())}")
    for k in list(bond_feats.keys())[:3]:
        print(f"  {k} -> {bond_feats[k]}")

# line 74-77: if len(row[bond_key]) == 1: bonds = row[bond_key][0]
sub("line 74-77: if len(row[bond_key]) == 1: bonds = row[bond_key][0]")
print(f"len(row[BOND_KEY]) = {len(row[BOND_KEY])}")
if len(row[BOND_KEY]) == 1:
    bonds = row[BOND_KEY][0]
    print("len==1 → bonds = row[BOND_KEY][0]")
else:
    bonds = row[BOND_KEY]
    print("len!=1 → bonds unchanged")
print(f"bonds ({len(bonds)} pairs): {bonds}")

# line 79-81: if filter_self_bonds
sub("line 79-81: filter_self_bonds=True")
print(f"before filter: {len(bonds)} pairs")
self_loops = [b for b in bonds if b[0] == b[1]]
print(f"self-loops found: {self_loops}")
bonds     = {tuple(sorted(b)): None for b in bonds if b[0] != b[1]}
bond_feats = {k: v for k, v in bond_feats.items() if k[0] != k[1]}
print(f"after filter: bonds={len(bonds)} pairs, bond_feats={len(bond_feats)} entries")
print(f"\nbonds dict keys:")
for k in sorted(bonds.keys()):
    flag = "✓ real" if k in mg_edges else "✗ NOT in molecule_graph"
    print(f"  {k}  {flag}")
print(f"\nReal bonds MISSING from bonds dict:")
for e in mg_edges:
    if e not in bonds:
        print(f"  {e}  ← missing")

# line 68-71: mol_graph, pmg_mol, elements
sub("line 68-71: mol_graph / pmg_mol / elements_from_pmg")
mol_graph = row.molecule_graph
pmg_mol   = row.molecule
elements  = elements_from_pmg(pmg_mol)
print(f"elements_from_pmg result: {elements}")

# line 84-96: MoleculeWrapper(...)
sub("line 84-96: MoleculeWrapper(...)")
id_combined = str(row.ids) + "_" + row.names
print(f"id_combined = {id_combined}")
print(f"bonds passed in       : {len(bonds)} pairs → {sorted(bonds.keys())}")
print(f"atom_features passed  : {len(atom_feats)} atoms")
print(f"bond_features passed  : {len(bond_feats)} bonds")

mol_wrapper = MoleculeWrapper(
    mol_graph,
    functional_group=None,
    free_energy=None,
    id=id_combined,
    bonds=bonds,
    non_metal_bonds=bonds,
    atom_features=atom_feats,
    bond_features=bond_feats,
    global_features={},
    original_atom_ind=None,
    original_bond_mapping=None,
)

sub("MoleculeWrapper object — key attributes")
print(f"mol_wrapper.id           = {mol_wrapper.id}")
print(f"mol_wrapper.bonds        = {list(mol_wrapper.bonds.keys())}")
print(f"mol_wrapper.num_atoms    = {mol_wrapper.num_atoms}")
print(f"mol_wrapper.coords       = {mol_wrapper.coords}")
print(f"mol_wrapper.species      = (skipped — pymatgen Site.is_ordered bug)")
print(f"mol_wrapper.atom_features keys = {list(mol_wrapper.atom_features.keys())}")
print(f"mol_wrapper.bond_features keys = {list(mol_wrapper.bond_features.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
hdr("HeteroCompleteGraphFromMolWrapper.build_graph  [grapher.py]  — line by line")
# ══════════════════════════════════════════════════════════════════════════════

mol = mol_wrapper   # alias matching the grapher code

# line 21: bonds = list(mol.bonds.keys())
sub("line 21: bonds = list(mol.bonds.keys())")
bonds_g   = list(mol.bonds.keys())
print(f"bonds ({len(bonds_g)}): {bonds_g}")

# line 23-24: num_bonds / num_atoms
sub("line 23-24: num_bonds = len(bonds) / num_atoms = len(mol.coords)")
num_bonds = len(bonds_g)
num_atoms = len(mol.coords)
print(f"num_bonds = {num_bonds}  (molecule_graph has {len(mg_edges)} real bonds)")
print(f"num_atoms = {num_atoms}")

# line 36-40: a2b / b2a loop
sub("line 36-40: for b in range(num_bonds): build a2b / b2a")
a2b, b2a = [], []
for b in range(num_bonds):
    u = bonds_g[b][0]
    v = bonds_g[b][1]
    b2a.extend([[b, u], [b, v]])
    a2b.extend([[u, b], [v, b]])
    print(f"  b={b}  bond=({u},{v})  b2a adds [{b},{u}],[{b},{v}]  a2b adds [{u},{b}],[{v},{b}]")

print(f"\na2b ({len(a2b)} entries): {a2b}")
print(f"b2a ({len(b2a)} entries): {b2a}")

# line 42-45: a2g / b2g etc
sub("line 42-45: a2g / g2a / b2g / g2b")
a2g = [(a, 0) for a in range(num_atoms)]
g2a = [(0, a) for a in range(num_atoms)]
b2g = [(b, 0) for b in range(num_bonds)]
g2b = [(0, b) for b in range(num_bonds)]
print(f"a2g ({len(a2g)}): {a2g}")
print(f"b2g ({len(b2g)}): {b2g}")

# which atoms have no a2b edge?
sub("atoms with no a2b edge  (disconnected from bond nodes)")
atoms_in_a2b = {pair[0] for pair in a2b}
print(f"atoms appearing in a2b: {sorted(atoms_in_a2b)}")
disconnected = sorted(i for i in range(num_atoms) if i not in atoms_in_a2b)
print(f"DISCONNECTED atoms    : {disconnected}")
for i in disconnected:
    real = [(u,v) for u,v in mg_edges if u==i or v==i]
    print(f"  atom {i}  real bonds in molecule_graph: {real}")


# ══════════════════════════════════════════════════════════════════════════════
hdr("AtomFeaturizerGraphGeneral.__call__  [featurizer.py]  — line by line")
# ══════════════════════════════════════════════════════════════════════════════

# line 261-267: setup
sub("line 261-267: setup variables")
features        = mol.atom_features
feats_atom      = []
bond_list       = []
num_atoms_f     = len(mol.coords)
# mol.species triggers pymatgen Site.is_ordered bug; read from pmg_mol directly
def _sp(site):
    try: return site.specie.symbol
    except: return str(list(site.species.as_dict().keys())[0]).split(":")[0]
species_sites   = [_sp(s) for s in pmg_mol.sites]
bond_list_tuple = list(mol.bonds.keys())
atom_num        = len(species_sites)
[bond_list.append(list(bond)) for bond in bond_list_tuple]
print(f"num_atoms      = {num_atoms_f}")
print(f"species_sites  = {species_sites}")
print(f"bond_list      = {bond_list}  ({len(bond_list)} bonds)")

# line 269-273: find_rings + ring_features_from_atom_full
sub("line 269-273: find_rings(atom_num, bond_list, edges=False)  +  ring_features_from_atom_full")
cycles    = find_rings(atom_num, bond_list, edges=False)
ring_info = ring_features_from_atom_full(num_atoms_f, cycles, ALLOWED_RING_SIZE)
print(f"cycles found: {cycles}")
print(f"ring_info type={type(ring_info)}")
print(f"ring_info value={ring_info}")
# ring_features_from_atom_full returns a dict {atom_idx: (ring_inclusion, ring_size_list)}
# or a list depending on version — handle both
if isinstance(ring_info, dict):
    for i in range(num_atoms_f):
        inc, szlist = ring_info[i]
        print(f"  atom {i:2d} ({species_sites[i]:2s}): ring_inclusion={inc}  ring_sizes={szlist}")
elif isinstance(ring_info, list) and len(ring_info) > 0 and isinstance(ring_info[0], tuple):
    for i, (inc, szlist) in enumerate(ring_info):
        print(f"  atom {i:2d} ({species_sites[i]:2s}): ring_inclusion={inc}  ring_sizes={szlist}")
else:
    print(f"  raw: {ring_info}")

# line 275-292: per-atom feature loop
sub("line 275-292: for atom_ind in range(num_atoms): build ft")
print(f"{'i':>3} {'sp':>3}  degree  h_count  ring_inc  ring_sizes           one_hot(C,F,H,N,O)   QTAIM[0]")
for atom_ind in range(num_atoms_f):
    ft = []
    atom_element            = species_sites[atom_ind]
    h_count, degree         = h_count_and_degree(atom_ind, bond_list, species_sites)
    if isinstance(ring_info, dict):
        ring_inclusion, ring_sz = ring_info[atom_ind]
    else:
        ring_inclusion, ring_sz = ring_info[atom_ind]

    ft.append(degree)
    ft.append(h_count)
    ft.append(ring_inclusion)
    ft += ring_sz
    ft += one_hot_encoding(atom_element, list(ELEMENT_SET))
    qtaim0 = features[atom_ind][ATOM_KEYS[0]]

    print(f"  {atom_ind:2d}  {atom_element:>2}  {degree:>6}  {h_count:>7}  {ring_inclusion:>8}  "
          f"{ring_sz}  {one_hot_encoding(atom_element, list(ELEMENT_SET))}  {qtaim0:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
hdr("BondAsNodeGraphFeaturizerGeneral.__call__  [featurizer.py]  — line by line")
# ══════════════════════════════════════════════════════════════════════════════

# line 125-129: setup
sub("line 125-129: setup variables")
bond_list_b  = list(mol.bonds)
num_bonds_f  = len(bond_list_b)
num_atoms_b  = int(mol.num_atoms)
features_b   = mol.bond_features
xyz          = mol.coords
print(f"bond_list  = {bond_list_b}  ({num_bonds_f} bonds)")
print(f"num_atoms  = {num_atoms_b}")

# line 147-155: find_rings + ring_features_for_bonds_full
sub("line 147-155: find_rings(num_atoms, bond_list, allowed_ring_size, edges=True)  +  ring_features_for_bonds_full")
cycles_b       = find_rings(num_atoms_b, bond_list_b, ALLOWED_RING_SIZE, edges=True)
no_metal_bin   = [1 for _ in range(num_bonds_f)]
ring_dict      = ring_features_for_bonds_full(bond_list_b, no_metal_bin, cycles_b, ALLOWED_RING_SIZE)
ring_dict_keys = list(ring_dict.keys())
print(f"cycles found   : {cycles_b}")
print(f"ring_dict keys : {ring_dict_keys}")
print(f"ring_dict      : {ring_dict}")

# line 157-194: per-bond feature loop
sub("line 157-194: for ind, bond in enumerate(bond_list): build ft")
print(f"{'b':>3} {'bond':>8}  metal  ring_inc  ring_sizes           QTAIM[0]")
for ind, bond in enumerate(bond_list_b):
    ft = []
    if tuple(bond) in ring_dict_keys:
        ft.append(ring_dict[tuple(bond)][0])   # metal
        ft.append(ring_dict[tuple(bond)][1])   # ring_inclusion
        ft += ring_dict[tuple(bond)][2]         # one-hot ring sizes
    else:
        ft += [0, 0]
        ft += [0 for _ in ALLOWED_RING_SIZE]

    qtaim0 = features_b[bond][BOND_KEYS[0]]
    ft.append(qtaim0)

    print(f"  {ind:2d}  {bond}  {ft[0]:>5}  {ft[1]:>8}  {ft[2:2+len(ALLOWED_RING_SIZE)]}  {qtaim0:.4f}")

sub("bonds that have NO bond node (missing from bond_list_b)")
for e in mg_edges:
    if e not in bond_list_b:
        print(f"  {e}  ← no bond node in graph")