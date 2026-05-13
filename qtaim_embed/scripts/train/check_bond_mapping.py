import json, pandas as pd, ast, numpy as np

JSON_FOLDER = "/lustre/fsn1/projects/rech/ihj/urb54jd/gnn/control_and_critical_points_GNNs/data/criticalpoints_jsonfiles"
ELF_CSV = "/lustre/fsn1/projects/rech/ihj/urb54jd/gnn/control_and_critical_points_GNNs/data/qm9_43k_clean_with_val.csv"
SOTA_PKL = "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/data_suba/filtered_qtaim_fullqm9/train_43k.pkl"

# load the molecule from the verify output — it was gdb_78284 (19 atoms)
elf_df  = pd.read_csv(ELF_CSV)
sota_df = pd.read_pickle(SOTA_PKL)
sota_df["gdb_num"] = sota_df["names"].str.extract(r"gdb_(\d+)\.xyz").astype(int)

# find gdb_78284
row_sota = sota_df[sota_df["gdb_num"] == 78284].iloc[0]
row_elf  = elf_df[elf_df["GDB_Index"] == 78284].iloc[0]

jf = f"{JSON_FOLDER}/integrated_aimel_078284.json"
with open(jf) as f: jdata = json.load(f)

atoms = jdata["Atoms"]
cps   = jdata["Critical Points"]

print(f"Molecule: gdb_78284  ({len(atoms)} atoms)")
print(f"\nJSON atom list (idx → name → symbol → z):")
atom_name_to_idx = {}
atom_idx_to_name = {}
atom_idx_to_sym  = {}
for idx, (key, info) in enumerate(atoms.items()):
    name = str(info.get("Atom list","")).strip()
    z    = int(info.get("z value", 0))
    sym  = {6:"C",9:"F",1:"H",7:"N",8:"O"}.get(z,"?")
    atom_name_to_idx[name] = idx
    atom_idx_to_name[idx]  = name
    atom_idx_to_sym[idx]   = sym
    print(f"  idx={idx:2d}  name={name:<5}  symbol={sym}  z={z}")

print(f"\nSOTA bond indices (extra_feat_bond_indices_qtaim):")
def parse_bond_indices(raw):
    s = str(raw).strip()
    try: return [(int(a),int(b)) for a,b in ast.literal_eval(s)]
    except: return []

bond_indices = parse_bond_indices(row_sota["extra_feat_bond_indices_qtaim"])

# build ELF valence CP lookup
val_elf = {}
for k, cp_info in cps.items():
    if cp_info.get("Type") != "valence": continue
    raw   = str(cp_info.get("Atom list","")).strip()
    items = [a.strip() for a in raw.strip("()").split(",") if a.strip()]
    idxs  = [atom_name_to_idx[a] for a in items if a in atom_name_to_idx]
    if len(idxs) == 2:
        key = frozenset(idxs)
        if key not in val_elf: val_elf[key] = []
        val_elf[key].append(k)

print(f"\n  {'idx':<5} {'pair':<10} {'atom i':<8} {'atom j':<8} {'ELF CP':<20} {'status'}")
print(f"  {'-'*70}")
for bi, (i,j) in enumerate(bond_indices):
    sym_i = atom_idx_to_sym.get(i,"?")
    sym_j = atom_idx_to_sym.get(j,"?")
    name_i = atom_idx_to_name.get(i,"?")
    name_j = atom_idx_to_name.get(j,"?")
    if i == j:
        status = "SELF-LOOP"
        cps_found = "—"
    else:
        key = frozenset({i,j})
        cps_found_list = val_elf.get(key, [])
        cps_found = ",".join(f"CP{k}" for k in cps_found_list) if cps_found_list else "none"
        status = "✓ found" if cps_found_list else "✗ MISMATCH"
    print(f"  [{bi:2d}]  ({i:2d},{j:2d})  {sym_i}{name_i:<6}  {sym_j}{name_j:<6}  {cps_found:<20}  {status}")

print(f"\nAll valence CPs in JSON (2-atom):")
for k, cp_info in cps.items():
    if cp_info.get("Type") != "valence": continue
    raw   = str(cp_info.get("Atom list","")).strip()
    items = [a.strip() for a in raw.strip("()").split(",") if a.strip()]
    idxs  = [atom_name_to_idx[a] for a in items if a in atom_name_to_idx]
    if len(idxs) == 2:
        i,j = idxs
        sym_i = atom_idx_to_sym.get(i,"?")
        sym_j = atom_idx_to_sym.get(j,"?")
        print(f"  CP{k:<3} {raw:<20} idx=({i},{j})  {sym_i}-{sym_j}")