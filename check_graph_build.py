"""
check_graph_build.py
====================
Traces exactly how the SOTA DGL graph is built from the PKL row,
following the code path:
  dataset.py → molwrapper.py → grapher.py → HeteroCompleteGraphFromMolWrapper
"""
import subprocess

# ── check HeteroCompleteGraphFromMolWrapper ───────────────────────
files_to_check = [
    "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/qtaim_embed/data/grapher.py",
    "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/qtaim_embed/data/featurizer.py",
    "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/qtaim_embed/data/molwrapper.py",
]

keywords = [
    "bond_key", "map_key", "manual_bond", "bonds", "atom_order",
    "reorder", "heavy", "sort", "HeteroComplete", "build_graph",
    "extra_feat_bond", "bond_index", "bond_list"
]

for fpath in files_to_check:
    if not __import__("os").path.exists(fpath):
        print(f"NOT FOUND: {fpath}")
        continue
    result = subprocess.run(
        ["grep", "-n"] + keywords + [fpath],
        capture_output=True, text=True)
    print(f"\n{'='*60}")
    print(f"FILE: {fpath}")
    print(f"{'='*60}")
    print(result.stdout[:4000])

# ── find and check HeteroCompleteGraphFromMolWrapper ─────────────
print(f"\n{'='*60}")
print("Finding HeteroCompleteGraphFromMolWrapper definition...")
result = subprocess.run(
    ["grep", "-rn", "class HeteroComplete\|def build\|def __call__\|bond_key\|manual_bond",
     "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/qtaim_embed/data/grapher.py"],
    capture_output=True, text=True)
print(result.stdout[:5000])

# ── check dataset.py for how MoleculeWrapper is instantiated ──────
print(f"\n{'='*60}")
print("dataset.py — MoleculeWrapper instantiation:")
result = subprocess.run(
    ["grep", "-n", "-A", "10",
     "MoleculeWrapper\|create_wrapper\|bonds=\|bond_key",
     "/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/qtaim_embed/core/dataset.py"],
    capture_output=True, text=True)
print(result.stdout[:5000])