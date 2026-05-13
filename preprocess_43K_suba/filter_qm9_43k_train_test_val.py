from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import pandas as pd

"""
python filter_qm9_43k_train_test_val.py \
  --train-pkl train_qm9_qtaim_1205_labelled_corrected.pkl \
  --test-pkl test_qm9_qtaim_1205_labelled_corrected.pkl \
  --split-csv qm9_43k_clean_with_val.csv \
  --pickle-id-field names \
  --csv-id-field names \
  --split-field split \
  --outdir filtered_qtaim \
  --suffix _43k \
  --write-csv
"""
COMMON_ID_FIELDS: Tuple[str, ...] = (
    "names",
    "smiles", "SMILES", "canonical_smiles", "cano_smiles",
    "mol_id", "molid", "molecule_id", "id",
    "inchi", "InChI", "inchi_key", "InChIKey",
    "cid", "index", "GDB_Index",
)

COMMON_SPLIT_FIELDS: Tuple[str, ...] = (
    "split", "Split", "set", "subset"
)


def load_pickle_any(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pickle file not found: {p}")
    try:
        return pd.read_pickle(p)
    except Exception:
        with p.open("rb") as fh:
            return pickle.load(fh)


def to_dataframe(data: Any) -> tuple[pd.DataFrame, str]:
    if isinstance(data, pd.DataFrame):
        return data.copy(), "dataframe"
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return pd.DataFrame(data), "list_of_dicts"
    raise TypeError("Unsupported pickle structure. Expected pandas.DataFrame or list[dict].")


def restore_to_original_type(df: pd.DataFrame, kind: str) -> Any:
    if kind == "dataframe":
        return df
    if kind == "list_of_dicts":
        return df.to_dict(orient="records")
    raise ValueError(f"Unknown kind: {kind}")


def detect_field(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def normalize_names_value(raw: object) -> str:
    import re

    s = str(raw).strip()
    if not s:
        return s

    s_noext = s[:-4] if s.lower().endswith(".xyz") else s

    if re.fullmatch(r"\d+", s_noext):
        return f"gdb_{int(s_noext)}.xyz"

    m_gdb = re.fullmatch(r"gdb_?(\d+)", s_noext)
    if m_gdb:
        return f"gdb_{int(m_gdb.group(1))}.xyz"

    m_dsgdb = re.search(r"(\d+)$", s_noext)
    if m_dsgdb and s_noext.lower().startswith("dsgdb"):
        return f"gdb_{int(m_dsgdb.group(1))}.xyz"

    return f"{s_noext}.xyz"


def normalize_id_series(series: pd.Series, id_field: str) -> pd.Series:
    s = series.astype(str).str.strip()

    # Normalize fields that may refer to QM9 molecule IDs in different formats
    # into a common "gdb_<n>.xyz" form for matching.
    if id_field in {"names", "ID", "GDB_Index", "index"}:
        return s.map(normalize_names_value)

    return s


def save_pickle(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    if isinstance(obj, pd.DataFrame):
        obj.to_pickle(path)
    else:
        with path.open("wb") as fh:
            pickle.dump(obj, fh)


def make_csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_stringify(x):
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        try:
            return repr(x)
        except Exception:
            return f"<UNSERIALIZABLE:{type(x).__name__}>"

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(safe_stringify)

    return df



def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create train/test/val QTAIM pickle files matching qm9_43k_clean_with_val.csv split column."
    )
    parser.add_argument("--train-pkl", required=True, help="Original QTAIM train pickle")
    parser.add_argument("--test-pkl", required=True, help="Original QTAIM test pickle")
    parser.add_argument("--split-csv", required=True, help="CSV containing split column")
    parser.add_argument("--pickle-id-field", default=None, help="ID field in QTAIM pickles, e.g. names")
    parser.add_argument("--csv-id-field", default=None, help="ID field in split CSV, e.g. names")
    parser.add_argument("--split-field", default=None, help="Split column in CSV, e.g. split")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--suffix", default="_43k", help="Suffix to append before .pkl")
    parser.add_argument("--write-csv", action="store_true", help="Also write CSV copies")
    parser.add_argument("--dry-run", action="store_true", help="Preview only; do not write files")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load split CSV
    split_df = pd.read_csv(args.split_csv)
    csv_id_field = args.csv_id_field or detect_field(split_df.columns, COMMON_ID_FIELDS)
    split_field = args.split_field or detect_field(split_df.columns, COMMON_SPLIT_FIELDS)

    if not csv_id_field:
        raise ValueError(
            f"Could not detect ID field in split CSV. Columns: {list(split_df.columns)}. "
            f"Use --csv-id-field."
        )
    if not split_field:
        raise ValueError(
            f"Could not detect split field in split CSV. Columns: {list(split_df.columns)}. "
            f"Use --split-field."
        )

    logging.info(f"Using CSV ID field: '{csv_id_field}'")
    logging.info(f"Using CSV split field: '{split_field}'")

    if csv_id_field not in split_df.columns:
        raise KeyError(
            f"CSV ID field '{csv_id_field}' not found in split CSV. "
            f"Available columns: {list(split_df.columns)}"
        )

    if split_field not in split_df.columns:
        raise KeyError(
            f"Split field '{split_field}' not found in split CSV. "
            f"Available columns: {list(split_df.columns)}"
        )
    # Normalize CSV ids
    split_df = split_df.copy()
    split_df[csv_id_field] = normalize_id_series(split_df[csv_id_field], csv_id_field)
    split_df[split_field] = split_df[split_field].astype(str).str.strip().str.lower()

    valid_splits = {"train", "test", "val", "valid", "validation"}
    observed_splits = set(split_df[split_field].unique())
    if not observed_splits.intersection(valid_splits):
        raise ValueError(f"No expected splits found in '{split_field}'. Observed: {sorted(observed_splits)}")

    split_df[split_field] = split_df[split_field].replace({
        "valid": "val",
        "validation": "val",
    })

    # Build ID sets from CSV
    train_ids = set(split_df.loc[split_df[split_field] == "train", csv_id_field])
    test_ids = set(split_df.loc[split_df[split_field] == "test", csv_id_field])
    val_ids = set(split_df.loc[split_df[split_field] == "val", csv_id_field])

    logging.info(f"CSV split sizes: train={len(train_ids)}, test={len(test_ids)}, val={len(val_ids)}")

    # Load QTAIM pickles
    train_raw = load_pickle_any(args.train_pkl)
    test_raw = load_pickle_any(args.test_pkl)

    train_df, train_kind = to_dataframe(train_raw)
    test_df, test_kind = to_dataframe(test_raw)

    if train_kind != test_kind:
        logging.warning(f"Train/test pickle kinds differ: train={train_kind}, test={test_kind}")

    pickle_id_field = args.pickle_id_field or detect_field(train_df.columns, COMMON_ID_FIELDS) or detect_field(test_df.columns, COMMON_ID_FIELDS)
    if not pickle_id_field:
        raise ValueError(
            f"Could not detect ID field in pickle files. "
            f"Train columns: {list(train_df.columns)} | Test columns: {list(test_df.columns)}. "
            f"Use --pickle-id-field."
        )

    if pickle_id_field not in train_df.columns or pickle_id_field not in test_df.columns:
        raise KeyError(
            f"Pickle ID field '{pickle_id_field}' must exist in both pickles. "
            f"Train columns: {list(train_df.columns)} | Test columns: {list(test_df.columns)}"
        )

    logging.info(f"Using pickle ID field: '{pickle_id_field}'")

    # Combine both QTAIM pickles
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True).copy()
    combined_df[pickle_id_field] = normalize_id_series(combined_df[pickle_id_field], pickle_id_field)

    before_rows = len(combined_df)
    before_unique = combined_df[pickle_id_field].nunique()
    logging.info(f"Combined QTAIM rows={before_rows}, unique_ids={before_unique}")

    # Optional duplicate handling
    dup_mask = combined_df.duplicated(subset=[pickle_id_field], keep="first")
    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        logging.warning(f"Found {n_dups} duplicate IDs across combined pickles. Keeping first occurrence.")
        combined_df = combined_df.loc[~dup_mask].copy()

    after_unique = combined_df[pickle_id_field].nunique()
    logging.info(f"Combined unique IDs after deduplication={after_unique}")

    # Filter into new splits
    train_43k_df = combined_df.loc[combined_df[pickle_id_field].isin(train_ids)].copy()
    test_43k_df = combined_df.loc[combined_df[pickle_id_field].isin(test_ids)].copy()
    val_43k_df = combined_df.loc[combined_df[pickle_id_field].isin(val_ids)].copy()

    # Coverage report
    train_found = set(train_43k_df[pickle_id_field])
    test_found = set(test_43k_df[pickle_id_field])
    val_found = set(val_43k_df[pickle_id_field])

    logging.info(f"train_43k: rows={len(train_43k_df)}, matched_ids={len(train_found)}/{len(train_ids)}")
    logging.info(f"test_43k:  rows={len(test_43k_df)}, matched_ids={len(test_found)}/{len(test_ids)}")
    logging.info(f"val_43k:   rows={len(val_43k_df)}, matched_ids={len(val_found)}/{len(val_ids)}")

    missing_train = train_ids - train_found
    missing_test = test_ids - test_found
    missing_val = val_ids - val_found

    if missing_train:
        logging.warning(f"Missing train IDs in QTAIM data: {len(missing_train)}")
    if missing_test:
        logging.warning(f"Missing test IDs in QTAIM data: {len(missing_test)}")
    if missing_val:
        logging.warning(f"Missing val IDs in QTAIM data: {len(missing_val)}")

    if args.dry_run:
        logging.info("Dry-run: no files written.")
        return

    out_train = outdir / f"train{args.suffix}.pkl"
    out_test = outdir / f"test{args.suffix}.pkl"
    out_val = outdir / f"val{args.suffix}.pkl"

    # Preserve original type if both input pickles used same structure; otherwise default to DataFrame
    out_kind = train_kind if train_kind == test_kind else "dataframe"

    train_out = restore_to_original_type(train_43k_df, out_kind)
    test_out = restore_to_original_type(test_43k_df, out_kind)
    val_out = restore_to_original_type(val_43k_df, out_kind)

    save_pickle(out_train, train_out)
    save_pickle(out_test, test_out)
    save_pickle(out_val, val_out)

    logging.info(f"Wrote: {out_train}")
    logging.info(f"Wrote: {out_test}")
    logging.info(f"Wrote: {out_val}")

    if args.write_csv:
        train_csv_df = make_csv_safe(train_43k_df)
        test_csv_df = make_csv_safe(test_43k_df)
        val_csv_df = make_csv_safe(val_43k_df)

        train_csv_df.to_csv(outdir / f"train{args.suffix}.csv", index=False)
        test_csv_df.to_csv(outdir / f"test{args.suffix}.csv", index=False)
        val_csv_df.to_csv(outdir / f"val{args.suffix}.csv", index=False)

        logging.info(f"Wrote CSV copies in: {outdir}")

if __name__ == "__main__":
    main()