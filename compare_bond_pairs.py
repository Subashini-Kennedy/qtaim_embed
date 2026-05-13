"""
compare_bond_pairs.py
Compare bond pairs in:
  1. CP graph (ELF valence CPs from JSON)
  2. QTAIM SOTA graph (bond_indices from PKL)
"""
import ast, json, sys
from pathlib import Path
import pandas as pd

ROOT_GNN  = Path("/lustre/fsn1/projects/rech/ihj/urb54jd/gnn/control_and_critical_points_GNNs")
ROOT_SOTA = Path("/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private")

elf_csv  = ROOT_GNN  / "data/qm9_43k_clean_with_val.csv"
json_dir = ROOT_GNN  / "data/criticalpoints_jsonfiles"
sota_csvs = [
    ROOT_SOTA / "data_suba/filtered_qtaim_fullqm9_corrected/train_43k.csv",
    ROOT_SOTA / "data_suba/filtered_qtaim_fullqm9_corrected/val_43k.csv",
    ROOT_SOTA / "data_suba/filtered_qtaim_fullqm9_corrected/test_43k.csv",
]

elf_df  = pd.read_csv(elf_csv)
sota_df = pd.concat([pd.read_csv(p) for p in sota_csvs], ignore_index=True)
sota_df["gdb_num"] = sota_df["names"].str.extract(r"gdb_(\d+)\.xyz").astype(int)
merged = elf_df.merge(sota_df, left_on="GDB_Index", right_on="gdb_num", how="inner")

Z_SYM = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F"}

def parse_bond_indices(raw):
    s = str(raw).strip()
    try:    return [(int(a), int(b)) for a, b in ast.literal_eval(s)]
    except: return []

def pair_to_str(pair_tuple, atom_z):
    i, j = pair_tuple
    si = Z_SYM.get(atom_z.get(i, 0), "?")
    sj = Z_SYM.get(atom_z.get(j, 0), "?")
    return f"({i},{j}) {si}-{sj}"

N = 5
count = 0
for _, row in merged.iterrows():
    if count >= N: break
    mol_id  = str(row["ID"])
    gdb_num = int(row["GDB_Index"])
    jf      = json_dir / f"integrated_aimel_{gdb_num:06d}.json"
    if not jf.exists(): continue

    with open(jf) as f: jdata = json.load(f)
    atoms = jdata.get("Atoms", {})
    cps   = jdata.get("Critical Points", {})

    atom_name_to_idx = {}
    atom_z = {}
    for idx, (_, info) in enumerate(atoms.items()):
        name = str(info.get("Atom list", "")).strip()
        atom_name_to_idx[name] = idx
        atom_z[idx] = int(info.get("z value", 0))

    # ELF valence basin pairs — stored as sorted tuples
    elf_pairs = set()
    for _, cp_info in cps.items():
        if cp_info.get("Type", "") != "valence": continue
        raw   = str(cp_info.get("Atom list", "")).strip()
        items = [a.strip() for a in raw.strip("()").split(",") if a.strip()]
        idxs  = [atom_name_to_idx[a] for a in items if a in atom_name_to_idx]
        if len(idxs) == 2:
            elf_pairs.add(tuple(sorted(idxs)))

    # QTAIM bond pairs — stored as sorted tuples
    qtaim_pairs = set()
    for i, j in parse_bond_indices(row["new_bond_indices"]):
        if i != j:                          # skip self-loops
            qtaim_pairs.add(tuple(sorted([i, j])))

    common     = elf_pairs & qtaim_pairs
    elf_only   = elf_pairs - qtaim_pairs
    qtaim_only = qtaim_pairs - elf_pairs

    print("=" * 65)
    print(f"Mol: {mol_id}  GDB: {gdb_num}  Atoms: {len(atoms)}")
    print(f"  ELF CP bonds:   {len(elf_pairs)}")
    print(f"  QTAIM bonds:    {len(qtaim_pairs)}")
    print(f"  Overlap:        {len(common)}")
    print(f"  ELF-only:       {len(elf_only)}  (real bonds missing from QTAIM)")
    print(f"  QTAIM-only:     {len(qtaim_only)}  (non-covalent BCPs absent in ELF)")
    print(f"  Common:         {sorted([pair_to_str(p,atom_z) for p in common])}")
    print(f"  ELF-only:       {sorted([pair_to_str(p,atom_z) for p in elf_only])}")
    print(f"  QTAIM-only:     {sorted([pair_to_str(p,atom_z) for p in qtaim_only])}")
    count += 1
