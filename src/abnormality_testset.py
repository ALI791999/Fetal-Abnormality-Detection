import os
import glob
import traceback
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd

from abnormality import hc_abnormality_from_image_exact
import configs

RACE_ASSUMPTION = "Non-Hispanic White"   # per your request


def _iter_hc18_image_paths(split: str) -> List[str]:
    """
    Collect HC18 image paths for a split.
      - train/validation: '*HC.png'
      - test: '*.png'
    """
    split = split.lower().strip()
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of {'train','validation','test'}")
    base = os.path.join(configs.MAIN_DIR, f"{split}_set")
    pattern = "*HC.png" if split in {"train", "validation"} else "*.png"
    paths = sorted(glob.glob(os.path.join(base, pattern)))
    if not paths:
        raise ValueError(f"No images found for split='{split}' at {base!r} with pattern {pattern!r}")
    return paths


def _flatten_row(res: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten single-image abnormality result into a one-row record."""
    row = {
        "image_path": res.get("image_path"),
        "race": res.get("race"),
        "pixel_size_mm": res.get("pixel_size_mm"),
        "hc_mm": res.get("hc_mm"),
        "ga_weeks_from_hc": res.get("ga_weeks_from_hc"),
        "ga_used_from_table": res.get("ga_used_from_table"),
        "percentile_band": res.get("percentile_band"),
        "classification": res.get("classification"),
    }
    cuts = res.get("cutoffs_mm_at_ga", {}) or {}
    for c in configs.PCT_COLS:
        row[c] = cuts.get(c, float("nan"))
    return row


def batch_hc18_split_to_excel(
    split: str = "validation",
    *,
    excel_output_path: Optional[str] = None,
    race: str = RACE_ASSUMPTION,
    pixel_size_mm: Optional[float] = None,
    model_path: Optional[str] = None,
    excel_path_growth: Optional[str] = None,
    sheet_name_growth: Optional[str] = None,
    log_every: int = 20,
) -> pd.DataFrame:
    """
    Run abnormality detection on the entire HC18 split and write results to Excel.

    Sheets:
      - 'results' : per-image measurements and labels
      - 'summary' : counts and simple stats
      - 'errors'  : images that failed to process
    """
    img_paths = _iter_hc18_image_paths(split)

    pixel_size_mm = float(pixel_size_mm if pixel_size_mm is not None else configs.PIXEL_SIZE_MM)
    model_path = model_path or configs.BEST_MODEL_SAVE_PATH
    excel_path_growth = excel_path_growth or configs.PERCENTILE_RANGE_PATH
    sheet_name_growth = sheet_name_growth or configs.SHEET_NAME
    out_path = excel_output_path or f"abnormality_results_{split}.xlsx"

    rows: List[Dict[str, Any]] = []
    errs: List[Dict[str, Any]] = []

    for i, p in enumerate(img_paths, 1):
        try:
            res = hc_abnormality_from_image_exact(
                image_path=p,
                pixel_size_mm=pixel_size_mm,
                model_path=model_path,
                race=race,
                excel_path=excel_path_growth,
                sheet_name=sheet_name_growth,
            )
            rows.append(_flatten_row(res))
        except Exception as e:
            errs.append(
                {
                    "image_path": p,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=2),
                }
            )
        if log_every and i % log_every == 0:
            print(f"[batch:{split}] processed {i}/{len(img_paths)}")

    df_results = pd.DataFrame(rows)
    df_errors = pd.DataFrame(errs)

    # Build a compact summary sheet
    summary_parts = []
    if not df_results.empty:
        # label distributions
        summary_parts.append(df_results["classification"].value_counts().rename_axis("classification").reset_index(name="count"))
        summary_parts.append(df_results["percentile_band"].value_counts().rename_axis("percentile_band").reset_index(name="count"))
        # simple stats
        stats = df_results[["hc_mm", "ga_weeks_from_hc"]].describe().T.reset_index().rename(columns={"index": "metric"})
        summary_parts.append(stats)

    # Save to Excel
    with pd.ExcelWriter(out_path) as writer:
        df_results.to_excel(writer, index=False, sheet_name="results")
        if summary_parts:
            # write summary tables stacked vertically
            start_row = 0
            for tbl in summary_parts:
                tbl.to_excel(writer, index=False, sheet_name="summary", startrow=start_row)
                start_row += len(tbl) + 3
        if not df_errors.empty:
            df_errors.to_excel(writer, index=False, sheet_name="errors")

    print(
        f"[batch:{split}] done → results: {len(df_results)} "
        + (f"| errors: {len(df_errors)} " if len(df_errors) else "")
        + f"| wrote: {out_path}"
    )
    return df_results


if __name__ == "__main__":
    # Example: run on the validation split with the Non-Hispanic White assumption
    batch_hc18_split_to_excel(
        split="validation",
        race=RACE_ASSUMPTION,
        excel_output_path="abnormality_results_validation.xlsx",
        pixel_size_mm=configs.PIXEL_SIZE_MM,  # override if needed
        model_path=configs.BEST_MODEL_SAVE_PATH,  # override if needed
        excel_path_growth=configs.PERCENTILE_RANGE_PATH,
        sheet_name_growth=configs.SHEET_NAME,
    )
