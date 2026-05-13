"""
add_elf_to_pkl.py  —  Option C only
=====================================
Add ELF basin features from JSON critical point files to the SOTA PKL files.

Option C: ELF on BOTH atom nodes AND bond nodes
  Atom nodes  (13 → 18):  core CP + lone-pair CPs summed → 5 ELF features
  Bond nodes  (7  → 12):  bonding CPs summed per pair    → 5 ELF features
                           zeros if SOTA bond has no ELF CP (bond topology mismatch)
                           zeros for self-loops

ELF feature columns added:
  Atom: elf_atom_group_value, elf_atom_volume, elf_atom_population,
        elf_atom_charge, elf_atom_value
  Bond: elf_bond_group_value, elf_bond_volume, elf_bond_population,
        elf_bond_charge, elf_bond_value

Justification:
  Core CP  (1 atom, type=core)    → belongs to atom (inner shell basin)
  Lone-pair CP (1 atom, type=valence) → belongs to atom (non-bonding basin)
  Bonding CP (2 atoms, type=valence)  → belongs to bond (shared bonding basin)
  SOTA bond with no ELF CP            → zeros (ELF topology does not detect this bond)
  Self-loop bonds                     → zeros

Modes:
  --mode verify   : print detailed per-molecule mapping for one sample molecule
  --mode process  : enrich all PKL splits and save

Usage:
  # Step 1 — verify mapping on 1 molecule (recommended first)
  python add_elf_to_pkl.py --mode verify \\
      --pkl_dir   /lustre/.../filtered_qtaim_fullqm9_corrected\\
      --json_dir  /lustre/.../criticalpoints_jsonfiles \\
      --elf_csv   /lustre/.../qm9_43k_clean_with_val.csv \\
      --outdir    /lustre/.../filtered_qtaim_fullqm9_elf_corrected

  # Step 2 — process all splits
  python add_elf_to_pkl.py --mode process \\
      --pkl_dir   /lustre/.../filtered_qtaim_fullqm9_corrected \\
      --json_dir  /lustre/.../criticalpoints_jsonfiles \\
      --elf_csv   /lustre/.../qm9_43k_clean_with_val.csv \\
      --outdir    /lustre/.../filtered_qtaim_fullqm9_elf_corrected
"""

from __future__ import annotations
import argparse, ast, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------------ #
# ELF feature column names                                            #
# ------------------------------------------------------------------ #
ATOM_ELF_COLS = [
    "elf_atom_group_value",   # 0=core/other, -3=valence attractor
    "elf_atom_volume",
    "elf_atom_population",
    "elf_atom_charge",
    "elf_atom_value",         # ELF η at attractor
]
BOND_ELF_COLS = [
    "elf_bond_group_value",
    "elf_bond_volume",
    "elf_bond_population",
    "elf_bond_charge",
    "elf_bond_value",
]
ELF_FEAT_NAMES = ["group_value", "volume", "population", "charge", "elf_value"]


# ------------------------------------------------------------------ #
# Parse helpers                                                       #
# ------------------------------------------------------------------ #
def parse_bond_indices(raw):
    s = str(raw).strip()
    try:    return [(int(a), int(b)) for a, b in ast.literal_eval(s)]
    except: return []


def cp_feat(cp_info):
    """Extract 5-dim ELF feature vector from one CP entry."""
    group = cp_info.get("group", 0)
    return np.array([
        0.0 if group == 0 else -3.0,
        float(cp_info.get("volume",     0) or 0),
        float(cp_info.get("population", 0) or 0),
        float(cp_info.get("charge",     0) or 0),
        float(cp_info.get("value",      0) or 0),
    ], dtype=float)


# ------------------------------------------------------------------ #
# Core extraction logic for one molecule                              #
# ------------------------------------------------------------------ #
def extract_elf_for_molecule(jdata, bond_indices, verbose=False):
    """
    Map ELF basin features from JSON onto SOTA atom/bond arrays.

    Returns:
      atom_elf : np.ndarray [n_atoms, 5]  — summed (core + lone-pair) per atom
      bond_elf : np.ndarray [n_bonds, 5]  — summed bonding CPs per bond, zeros if missing
    """
    atoms   = jdata.get("Atoms", {})
    cps     = jdata.get("Critical Points", {})
    n_atoms = len(atoms)
    n_bonds = len(bond_indices)

    # atom name → JSON index
    atom_name_to_idx = {}
    for idx, (_, info) in enumerate(atoms.items()):
        atom_name_to_idx[str(info.get("Atom list", "")).strip()] = idx

    # --- classify each CP ---
    atom_basins = {i: [] for i in range(n_atoms)}   # core + lone-pair per atom
    bond_basins = {}                                  # frozenset({i,j}) → list of feats

    for cp_key, cp_info in cps.items():
        raw   = str(cp_info.get("Atom list", "")).strip()
        items = [a.strip() for a in raw.strip("()").split(",") if a.strip()]
        idxs  = [atom_name_to_idx[a] for a in items if a in atom_name_to_idx]
        cp_type = cp_info.get("Type", "")
        feat    = cp_feat(cp_info)

        if len(idxs) == 1:
            # core CP or lone-pair CP → belongs to atom
            atom_basins[idxs[0]].append((cp_key, cp_type, feat))

        elif len(idxs) == 2:
            # bonding CP → belongs to bond
            key = frozenset(idxs)
            if key not in bond_basins:
                bond_basins[key] = []
            bond_basins[key].append((cp_key, feat))

    # --- sum atom basins ---
    atom_elf = np.zeros((n_atoms, 5), dtype=float)
    for ai, basins in atom_basins.items():
        if basins:
            atom_elf[ai] = np.sum([f for _, _, f in basins], axis=0)

    # --- sum bond basins, zeros for missing/self-loop ---
    bond_elf = np.zeros((n_bonds, 5), dtype=float)
    for bi, (i, j) in enumerate(bond_indices):
        if i == j:
            continue   # self-loop → zeros
        key    = frozenset({i, j})
        basins = bond_basins.get(key, [])
        if basins:
            bond_elf[bi] = np.sum([f for _, f in basins], axis=0)

    if verbose:
        _print_molecule_mapping(
            atoms, cps, atom_name_to_idx, atom_basins, bond_basins,
            bond_indices, atom_elf, bond_elf
        )

    return atom_elf, bond_elf


# ------------------------------------------------------------------ #
# Verbose mapping printer                                             #
# ------------------------------------------------------------------ #
def _print_molecule_mapping(atoms, cps, atom_name_to_idx,
                             atom_basins, bond_basins,
                             bond_indices, atom_elf, bond_elf):
    """Print detailed side-by-side mapping for one molecule."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  STEP 1 — JSON CRITICAL POINTS (raw)")
    print(sep)
    print(f"  {'CP key':<8} {'Type':<10} {'Atom list':<20} "
          f"{'Indices':<14} {'group_val':>10} {'volume':>10} "
          f"{'pop':>8} {'charge':>8} {'elf_val':>9}")
    print("  " + "-" * 80)
    for cp_key, cp_info in cps.items():
        raw   = str(cp_info.get("Atom list","")).strip()
        items = [a.strip() for a in raw.strip("()").split(",") if a.strip()]
        idxs  = [atom_name_to_idx[a] for a in items if a in atom_name_to_idx]
        cp_type = cp_info.get("Type","")
        f = cp_feat(cp_info)
        print(f"  CP {cp_key:<5} {cp_type:<10} {raw:<20} {str(idxs):<14} "
              f"{f[0]:>10.4f} {f[1]:>10.3f} {f[2]:>8.4f} "
              f"{f[3]:>8.4f} {f[4]:>9.5f}")

    print(f"\n{sep}")
    print("  STEP 2 — ATOM NODE MAPPING")
    print("  Rule: core CPs (type=core, 1 atom) + lone-pair CPs (type=valence, 1 atom)")
    print("        → summed into atom ELF feature vector")
    print(sep)
    atom_syms = [str(info.get("Atom list","")).rstrip("0123456789")
                 for _, info in atoms.items()]
    print(f"  {'Atom':<6} {'Symbol':<6} {'Basins assigned':<40} "
          f"{'→ summed ELF [grp, vol, pop, chg, eta]'}")
    print("  " + "-" * 80)
    for ai in range(len(atoms)):
        sym    = atom_syms[ai] if ai < len(atom_syms) else "?"
        basins = atom_basins[ai]
        if basins:
            desc = ", ".join(
                f"CP{k}({t[:4]})" for k, t, _ in basins
            )
            print(f"  {ai:<6} {sym:<6} {desc:<40} → {atom_elf[ai].round(4).tolist()}")
        else:
            print(f"  {ai:<6} {sym:<6} {'(no basin — H atom)':<40} → {atom_elf[ai].round(4).tolist()}")

    print(f"\n{sep}")
    print("  STEP 3 — SOTA BOND ORDER (from new_bond_indices)")
    print(sep)
    print(f"  {'Bond idx':<10} {'Pair':<10} {'ELF CPs found':<35} "
          f"{'→ summed ELF [grp, vol, pop, chg, eta]'}")
    print("  " + "-" * 80)
    for bi, (i, j) in enumerate(bond_indices):
        pair_str = f"({i},{j})"
        if i == j:
            print(f"  [{bi:2d}]       {pair_str:<10} {'SELF-LOOP → zeros':<35} "
                  f"→ {bond_elf[bi].round(4).tolist()}")
            continue
        key    = frozenset({i, j})
        basins = bond_basins.get(key, [])
        if basins:
            desc = ", ".join(f"CP{k}" for k, _ in basins)
            n    = len(basins)
            tag  = f"{n} CP(s): {desc}"
        else:
            tag = "NO ELF CP (topology mismatch) → zeros"
        print(f"  [{bi:2d}]       {pair_str:<10} {tag:<35} "
              f"→ {bond_elf[bi].round(4).tolist()}")

    print(f"\n{sep}")
    print("  STEP 4 — FINAL FEATURE DIMENSIONS")
    print(sep)
    print(f"  Atom features: 13 (SOTA topology) + 5 (ELF summed basins) = 18")
    print(f"  Bond features:  7 (SOTA topology) + 5 (ELF bonding CPs)   = 12")
    print(f"  Global features: 3 (unchanged)")
    print(f"\n  ELF feature names: {ELF_FEAT_NAMES}")
    print(f"  Atom ELF columns:  {ATOM_ELF_COLS}")
    print(f"  Bond ELF columns:  {BOND_ELF_COLS}")
    n_missing = sum(
        1 for (i,j) in bond_indices
        if i != j and frozenset({i,j}) not in bond_basins
    )
    n_self    = sum(1 for (i,j) in bond_indices if i == j)
    n_found   = len(bond_indices) - n_missing - n_self
    print(f"\n  Bond ELF coverage:")
    print(f"    Found:         {n_found}/{len(bond_indices)} bonds")
    print(f"    Missing (→ 0): {n_missing}/{len(bond_indices)} bonds (ELF topology mismatch)")
    print(f"    Self-loops:    {n_self}/{len(bond_indices)} bonds")
    print(sep)


# ------------------------------------------------------------------ #
# Verify mode — one molecule                                          #
# ------------------------------------------------------------------ #
def run_verify(args):
    JSON_DIR = Path(args.json_dir)
    elf_csv  = pd.read_csv(args.elf_csv, nrows=1)
    sota_csv = pd.read_csv(
        Path(args.pkl_dir) / "train_43k.csv"
        if (Path(args.pkl_dir) / "train_43k.csv").exists()
        else None
    ) if False else None

    # load first molecule from train PKL
    pkl_path = Path(args.pkl_dir) / "train_43k.pkl"
    df       = pd.read_pickle(pkl_path)
    row_pkl  = df.iloc[0]

    # get gdb_num from names
    mol_name = str(row_pkl["names"])
    gdb_num  = int(mol_name.replace("gdb_","").replace(".xyz",""))
    elf_row  = pd.read_csv(args.elf_csv)
    elf_row  = elf_row[elf_row["GDB_Index"] == gdb_num].iloc[0]
    mol_id   = str(elf_row["ID"])

    jf = JSON_DIR / f"integrated_aimel_{gdb_num:06d}.json"
    with open(jf) as f: jdata = json.load(f)

    bond_indices = parse_bond_indices(row_pkl["new_bond_indices"])
    #bond_indices = parse_bond_indices(row_pkl["extra_feat_bond_indices_qtaim"])

    print(f"\n{'#'*72}")
    print(f"  VERIFY MODE — molecule: {mol_id}  (gdb_{gdb_num})")
    print(f"  PKL row: {mol_name}")
    print(f"  JSON:    {jf}")
    print(f"  Atoms:   {len(jdata['Atoms'])}")
    print(f"  CPs:     {len(jdata['Critical Points'])}")
    print(f"  SOTA bonds: {len(bond_indices)}")
    print(f"{'#'*72}")

    atom_elf, bond_elf = extract_elf_for_molecule(
        jdata, bond_indices, verbose=True
    )

    print(f"\n✓ Verify complete. Ready to run --mode process")


# ------------------------------------------------------------------ #
# Process mode — enrich all PKL splits                                #
# ------------------------------------------------------------------ #
def run_process(args):
    PKL_DIR  = Path(args.pkl_dir)
    JSON_DIR = Path(args.json_dir)
    OUTDIR   = Path(args.outdir) / "option_c_atom_bond_elf"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    elf_csv_df = pd.read_csv(args.elf_csv)
    elf_csv_df["gdb_num"] = elf_csv_df["GDB_Index"].astype(int)
    name_to_gdb = {
        f"gdb_{int(r['gdb_num'])}.xyz": int(r["gdb_num"])
        for _, r in elf_csv_df.iterrows()
    }

    SPLITS = ["train_43k", "val_43k", "test_43k"]

    print(f"\n{'='*60}")
    print("Option C: adding ELF to atom + bond nodes")
    print(f"Output: {OUTDIR}")
    print(f"{'='*60}")

    for split in SPLITS:
        pkl_in  = PKL_DIR / f"{split}.pkl"
        pkl_out = OUTDIR  / f"{split}_elf.pkl"
        if not pkl_in.exists():
            print(f"  ⚠  {pkl_in} not found, skipping")
            continue

        print(f"\n--- {split} ---")
        df = pd.read_pickle(pkl_in)
        print(f"  {len(df)} molecules, {len(df.columns)} columns")

        atom_elf_list = []
        bond_elf_list = []
        n_skip = 0

        for _, row in tqdm(df.iterrows(), total=len(df)):
            mol_name = str(row["names"])
            gdb_num  = name_to_gdb.get(mol_name)
            if gdb_num is None:
                try: gdb_num = int(mol_name.replace("gdb_","").replace(".xyz",""))
                except:
                    n_skip += 1
                    n_atoms = len(row["molecule"])
                    n_bonds = len(parse_bond_indices(row["new_bond_indices"]))
                    atom_elf_list.append(np.zeros((n_atoms, 5)))
                    bond_elf_list.append(np.zeros((n_bonds, 5)))
                    continue

            jf = JSON_DIR / f"integrated_aimel_{gdb_num:06d}.json"
            if not jf.exists():
                n_skip += 1
                n_atoms = len(row["molecule"])
                n_bonds = len(parse_bond_indices(row["new_bond_indices"]))
                atom_elf_list.append(np.zeros((n_atoms, 5)))
                bond_elf_list.append(np.zeros((n_bonds, 5)))
                continue

            with open(jf) as f: jdata = json.load(f)
            bond_indices = parse_bond_indices(row["new_bond_indices"])
            atom_elf, bond_elf = extract_elf_for_molecule(jdata, bond_indices)
            atom_elf_list.append(atom_elf)
            bond_elf_list.append(bond_elf)

        if n_skip: print(f"  ⚠  {n_skip} molecules skipped")

        # add atom ELF columns
        for fi, col in enumerate(ATOM_ELF_COLS):
            df[col] = [arr[:, fi] for arr in atom_elf_list]

        # add bond ELF columns
        for fi, col in enumerate(BOND_ELF_COLS):
            df[col] = [arr[:, fi] for arr in bond_elf_list]

        df.to_pickle(pkl_out)
        print(f"  ✓ Saved: {pkl_out}  ({len(df.columns)} columns)")
        print(f"  New atom cols: {ATOM_ELF_COLS}")
        print(f"  New bond cols: {BOND_ELF_COLS}")

    print(f"\n{'='*60}")
    print("Done. Enriched PKLs saved to:")
    print(f"  {OUTDIR}/train_43k_elf.pkl")
    print(f"  {OUTDIR}/val_43k_elf.pkl")
    print(f"  {OUTDIR}/test_43k_elf.pkl")
    print(f"\nNext steps:")
    print(f"  1. Copy config: settings_qm9_nonlmdb_43k_withval_baseline_elf_atom_bond.json")
    print(f"  2. sbatch submit_elf_atom_bond.sh")
    print(f"{'='*60}")


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser("add_elf_to_pkl.py")
    p.add_argument("--mode",     required=True, choices=["verify","process"],
                   help="verify=print mapping for 1 molecule | process=enrich all PKLs")
    p.add_argument("--pkl_dir",  required=True)
    p.add_argument("--json_dir", required=True)
    p.add_argument("--elf_csv",  required=True)
    p.add_argument("--outdir",   required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "verify":
        run_verify(args)
    elif args.mode == "process":
        run_process(args)


if __name__ == "__main__":
    main()
