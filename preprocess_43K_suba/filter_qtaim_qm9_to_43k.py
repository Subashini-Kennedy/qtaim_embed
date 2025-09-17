# File: scripts/filter_qtaim_qm9_to_43k.py
"""
Filter QM9-QTAIM pickled datasets so both train and test contain only your target molecules.

- Works with pandas DataFrame or list-of-dicts pickles.
- Auto-detects ID column among common names; or use --id-field (recommended: 'names').
- Preserves split semantics: filters each split independently; keeps row order.

Examples
--------
# TXT whitelist where lines are like: gdb_1234.xyz
python scripts/filter_qtaim_qm9_to_43k.py \
  --train-pkl train_qm9_qtaim_1205_labelled_corrected.pkl \
  --test-pkl  test_qm9_qtaim_1205_labelled_corrected.pkl \
  --whitelist scripts/my_43K.txt \
  --id-field names \
  --suffix _my43K --write-csv

# CSV whitelist with a specific column (e.g., GDB_Index -> names)
python scripts/filter_qtaim_qm9_to_43k.py \
  --train-pkl train_qm9_qtaim_1205_labelled_corrected.pkl \
  --test-pkl  test_qm9_qtaim_1205_labelled_corrected.pkl \
  --whitelist scripts/43.csv --whitelist-col GDB_Index \
  --id-field names --suffix _my43K
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Set, Tuple, Union

import pandas as pd

# Put 'names' first so it auto-detects for your dataset
COMMON_ID_FIELDS: Tuple[str, ...] = (
    "names",
    "smiles", "SMILES", "canonical_smiles", "cano_smiles",
    "mol_id", "molid", "molecule_id", "id",
    "inchi", "InChI", "inchi_key", "InChIKey",
    "cid", "index",
)


def load_whitelist(paths: Sequence[Union[str, Path]], column: Optional[str]) -> Set[str]:
    """Load whitelist from one or more files (txt/csv/tsv)."""
    ids: Set[str] = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Whitelist file not found: {p}")
        if p.suffix.lower() in {".csv", ".tsv"}:
            sep = "," if p.suffix.lower() == ".csv" else "\t"
            df = pd.read_csv(p, sep=sep)
            col = column
            if col is None:
                for cand in COMMON_ID_FIELDS:
                    if cand in df.columns:
                        col = cand
                        break
            if col is None:
                raise ValueError(
                    f"Could not auto-detect whitelist column in {p}. "
                    f"Provide --whitelist-col. Columns: {list(df.columns)}"
                )
            vals = df[col].astype(str).str.strip()
            ids.update(v for v in vals if v)
        else:
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    val = line.strip()
                    if val:
                        ids.add(val)
    if not ids:
        raise ValueError("Whitelist is empty after loading.")
    return ids


def load_pickle_any(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pickle file not found: {p}")
    try:
        return pd.read_pickle(p)
    except Exception:
        with p.open("rb") as fh:
            return pickle.load(fh)


def to_dataframe(data: Any) -> tuple[pd.DataFrame, str, str]:
    """Return (df, kind, id_field) where kind ∈ {'dataframe','list_of_dicts'}."""
    if isinstance(data, pd.DataFrame):
        id_field = detect_id_field(data.columns)
        return data.copy(), "dataframe", id_field or ""
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        df = pd.DataFrame(data)
        id_field = detect_id_field(df.columns)
        return df, "list_of_dicts", id_field or ""
    raise TypeError("Unsupported pickle structure. Expected pandas.DataFrame or list[dict].")


def detect_id_field(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    for cand in COMMON_ID_FIELDS:
        if cand in cols:
            return cand
    return None


def normalize_whitelist_for_names(ids: Set[str]) -> Set[str]:
    """Normalize whitelist when matching against the 'names' column (gdb_####[.xyz])."""
    import re

    out: Set[str] = set()
    for raw in ids:
        s = str(raw).strip()
        if not s:
            continue
        s_noext = s[:-4] if s.lower().endswith(".xyz") else s

        # Numeric only: '123'
        if re.fullmatch(r"\d+", s_noext):
            num = int(s_noext)
            out.update({f"gdb_{num}", f"gdb_{num}.xyz"})
            continue

        # gdb forms with/without underscore: gdb_123 / gdb123
        m_gdb = re.fullmatch(r"gdb_?(\d+)", s_noext)
        if m_gdb:
            num = int(m_gdb.group(1))
            out.update({f"gdb_{num}", f"gdb_{num}.xyz"})
            continue

        # dsgdb9nsd_000123 → take trailing digits
        m_dsgdb = re.search(r"(\d+)$", s_noext)
        if m_dsgdb:
            num = int(m_dsgdb.group(1))
            out.update({f"gdb_{num}", f"gdb_{num}.xyz"})
            continue

        # Fallback: keep both base and base.xyz
        out.update({s_noext, f"{s_noext}.xyz"})
    return out


def filter_df_to_ids(df: pd.DataFrame, ids: Set[str], id_field: str) -> pd.DataFrame:
    if id_field not in df.columns:
        raise KeyError(f"Identifier column '{id_field}' not found. Available: {list(df.columns)}")
    series = df[id_field].astype(str)
    mask = series.isin(ids)
    return df.loc[mask].copy()


def restore_to_original_type(df: pd.DataFrame, kind: str) -> Any:
    if kind == "dataframe":
        return df
    if kind == "list_of_dicts":
        return df.to_dict(orient="records")
    raise ValueError(f"Unknown kind: {kind}")


def add_suffix_to_filename(path: Union[str, Path], suffix: str, outdir: Optional[Union[str, Path]] = None) -> Path:
    p = Path(path)
    stem = p.stem
    suffixes = "".join(p.suffixes)  # keep .pkl or .pkl.gz
    parent = Path(outdir) if outdir else p.parent
    return parent / f"{stem}{suffix}{suffixes}"


def summarize_split(name: str, df: pd.DataFrame, id_field: str, ids: Set[str]) -> str:
    n_rows = len(df)
    n_unique = df[id_field].astype(str).nunique() if id_field in df.columns else 0
    cov = (n_unique / max(1, len(ids))) * 100.0
    return f"{name}: rows={n_rows}, unique_ids={n_unique}, whitelist_size={len(ids)}, coverage={cov:.2f}%"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Filter QM9-QTAIM train/test pickles to your whitelist of molecule IDs.")
    parser.add_argument("--train-pkl", required=True, help="Path to train .pkl file")
    parser.add_argument("--test-pkl", required=True, help="Path to test .pkl file")
    parser.add_argument("--whitelist", required=True, nargs="+", help="One or more whitelist files")
    parser.add_argument("--whitelist-col", default=None, help="Column in CSV/TSV with IDs")
    parser.add_argument("--id-field", default=None, help="Identifier column in pickles (e.g., 'names')")
    parser.add_argument("--suffix", default="_my43k", help="Suffix for output filenames")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--write-csv", action="store_true", help="Also save CSV copies")
    parser.add_argument("--dry-run", action="store_true", help="Compute only; write nothing")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    ids = load_whitelist(args.whitelist, args.whitelist_col)

    # Load pickles
    train_raw = load_pickle_any(args.train_pkl)
    test_raw = load_pickle_any(args.test_pkl)

    train_df, train_kind, train_auto = to_dataframe(train_raw)
    test_df, test_kind, test_auto = to_dataframe(test_raw)

    # Resolve id_field
    id_field = args.id_field or train_auto or test_auto
    if not id_field:
        raise ValueError("Could not auto-detect an identifier column. Use --id-field (e.g., --id-field names).")

    # Normalize whitelist for 'names'
    if id_field == "names":
        ids = normalize_whitelist_for_names(ids)

    logging.info(f"Using identifier column: '{id_field}' (train kind={train_kind}, test kind={test_kind})")

    # Pre-stats
    if id_field not in train_df.columns or id_field not in test_df.columns:
        raise KeyError(
            f"'{id_field}' not found in columns. "
            f"Train columns: {list(train_df.columns)} | Test columns: {list(test_df.columns)}"
        )

    logging.info(summarize_split("train(before)", train_df, id_field, ids))
    logging.info(summarize_split("test(before)", test_df, id_field, ids))

    # Filter
    train_f = filter_df_to_ids(train_df, ids, id_field)
    test_f = filter_df_to_ids(test_df, ids, id_field)

    logging.info(summarize_split("train(after)", train_f, id_field, ids))
    logging.info(summarize_split("test(after)", test_f, id_field, ids))

    if args.dry_run:
        logging.info("Dry-run: no files written.")
        return

    # Write outputs
    out_train_pkl = add_suffix_to_filename(args.train_pkl, args.suffix, args.outdir)
    out_test_pkl = add_suffix_to_filename(args.test_pkl, args.suffix, args.outdir)

    train_out = restore_to_original_type(train_f, train_kind)
    test_out = restore_to_original_type(test_f, test_kind)

    if isinstance(train_out, pd.DataFrame):
        train_out.to_pickle(out_train_pkl)
    else:
        with open(out_train_pkl, "wb") as fh:
            pickle.dump(train_out, fh)

    if isinstance(test_out, pd.DataFrame):
        test_out.to_pickle(out_test_pkl)
    else:
        with open(out_test_pkl, "wb") as fh:
            pickle.dump(test_out, fh)

    logging.info(f"Wrote: {out_train_pkl}")
    logging.info(f"Wrote: {out_test_pkl}")

    if args.write_csv:
        train_csv = out_train_pkl.with_suffix("")
        train_csv = train_csv.with_name(train_csv.name + ".csv")
        test_csv = out_test_pkl.with_suffix("")
        test_csv = test_csv.with_name(test_csv.name + ".csv")
        train_f.to_csv(train_csv, index=False)
        test_f.to_csv(test_csv, index=False)
        logging.info(f"Wrote: {train_csv}")
        logging.info(f"Wrote: {test_csv}")


if __name__ == "__main__":
    main()
