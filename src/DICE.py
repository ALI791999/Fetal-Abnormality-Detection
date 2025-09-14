import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
from datetime import datetime

import configs
from model import SegModel
from learnable_resizer import LearnableResizer
from postprocess_functions import (
    logits_to_prob, tta_mean_prob_on_resized,
    largest_component, morph_clean,
    densecrf_refine,
    fit_ellipse_with_contours,
    to_uint8_from_01
)
from data_load import HC18Data  # uses your existing class

# ======= Feature toggles (same defaults as your visualize script) =======
USE_TTA     = True
USE_LCC     = True
USE_MORPH   = True
USE_CRF     = True
USE_ELLIPSE = True

THRESHOLD = getattr(configs, "THRESHOLD", 0.5)  # fallback if missing


def dice_score_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """pred, gt: HxW arrays with {0,1}"""
    pred = pred.astype(np.uint8)
    gt   = gt.astype(np.uint8)
    inter = (pred & gt).sum(dtype=np.int64)
    denom = pred.sum(dtype=np.int64) + gt.sum(dtype=np.int64)
    if denom == 0:
        return 1.0  # both empty → perfect match
    return (2.0 * inter + eps) / (denom + eps)


def build_model_and_resizer(device):
    ckpt = torch.load(configs.BEST_MODEL_SAVE_PATH, map_location=device)
    model = SegModel()
    resizer = LearnableResizer(
        in_ch=configs.MODEL_INPUT_CHANNELS,
        out_ch=configs.MODEL_INPUT_CHANNELS,
        out_size=(configs.IMAGE_SIZE, configs.IMAGE_SIZE)
    )
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "resizer" in ckpt and ckpt.get("resizer") is not None:
            resizer.load_state_dict(ckpt["resizer"])
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()
    resizer.to(device).eval()
    return model, resizer


def compute_dice_for_split(folder_path: str, split: str = "validation", excel_path: str | None = None):
    """
    Uses your HC18Data loader (train/validation/test). Writes an Excel file with:
      - Sheet 'per_image': filename, dice, pred_pixels, gt_pixels, intersection
      - Sheet 'summary'  : N, mean, median, std, min, max, global_dice
      - Sheet 'meta'     : run settings (threshold, toggles, model path, split, etc.)
    """
    device = configs.DEVICE

    # Ensure MAIN_DIR matches provided folder path (with trailing slash)
    configs.MAIN_DIR = folder_path if folder_path.endswith(os.sep) else folder_path + os.sep

    # Build model + resizer
    model, resizer = build_model_and_resizer(device)

    # Dataset / Loader (uses your HC18Data exactly as-is)
    ds = HC18Data(split)
    batch_size = 1  # safe for TTA/CRF/ellipse
    num_workers = getattr(configs, "NUM_WORKERS", 0)
    pin_mem = torch.cuda.is_available()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    per_image_rows = []
    dices = []
    total_inter = 0
    total_pred  = 0
    total_gt    = 0

    S = (configs.IMAGE_SIZE, configs.IMAGE_SIZE)
    seen = 0  # to align filenames with dataset order

    with torch.no_grad():
        for batch in loader:
            if split == "test":
                # No masks in test split → nothing to score
                continue

            x, y = batch  # x: [B,1,H1,W1] float in [0,1], y: [B,H,W] int {0,1}
            B = x.shape[0]
            x = x.to(device, non_blocking=True)

            # Resizer + model
            x_resized = resizer(x)  # [B,1,S,S]
            if USE_TTA:
                prob_t = tta_mean_prob_on_resized(model, x_resized)  # [B,1,S,S]
            else:
                logits = model(x_resized)                             # [B,C,S,S]
                prob_t = logits_to_prob(logits)                       # [B,1,S,S]

            prob_np_all = prob_t.squeeze(1).detach().cpu().numpy()     # [B,H,W]
            y_np_all    = y.detach().cpu().numpy().astype(np.uint8)    # [B,H,W]
            x_res_np_all = x_resized.squeeze(1).detach().cpu().numpy() # [B,H,W]

            for i in range(B):
                prob_np = prob_np_all[i]                                # HxW float
                pred_np = (prob_np >= THRESHOLD).astype(np.uint8)       # HxW {0,1}
                y_np    = y_np_all[i]                                   # HxW {0,1}

                # CRF guidance image
                x_res_np = x_res_np_all[i]
                if x_res_np.min() < 0 or x_res_np.max() > 1:
                    mn, mx = x_res_np.min(), x_res_np.max()
                    x_res_np = (x_res_np - mn) / (mx - mn + 1e-8)
                gray_u8 = to_uint8_from_01(x_res_np)

                # Post-processing
                if USE_LCC:
                    pred_np = largest_component(pred_np)
                if USE_MORPH:
                    pred_np = morph_clean(pred_np, min_size=200, close=3, open_=2, fill_holes=True)
                if USE_CRF:
                    pred_np = densecrf_refine(gray_u8, prob_np, iters=5, sxy=50, srgb=3, compat=4)
                    pred_np = (pred_np > 0).astype(np.uint8)
                if USE_ELLIPSE:
                    pred_np = fit_ellipse_with_contours(pred_np).astype(np.uint8)

                # Enforce SxS just in case
                if pred_np.shape != S:
                    pred_np = np.asarray(
                        Image.fromarray((pred_np * 255).astype(np.uint8)).resize(S, Image.Resampling.NEAREST)
                    ) > 127
                    pred_np = pred_np.astype(np.uint8)

                # Dice + aggregates
                d = dice_score_binary(pred_np, y_np)
                inter = int((pred_np & y_np).sum())
                p_sum = int(pred_np.sum())
                g_sum = int(y_np.sum())

                dices.append(float(d))
                total_inter += inter
                total_pred  += p_sum
                total_gt    += g_sum

                fname = Path(ds.x_data[seen + i]).name if hasattr(ds, "x_data") else f"index_{seen + i}"
                per_image_rows.append({
                    "filename": fname,
                    "dice": float(d),
                    "pred_pixels": p_sum,
                    "gt_pixels": g_sum,
                    "intersection": inter
                })

            seen += B

    # Build DataFrames
    if len(dices) == 0:
        print("No masks available to score (are you pointing at the correct split/folder?).")
        return

    df_per = pd.DataFrame(per_image_rows)
    dices_np = np.array(dices, dtype=np.float64)
    global_dice = (2.0 * total_inter) / max(1, (total_pred + total_gt))

    df_summary = pd.DataFrame([{
        "N": len(dices),
        "mean_dice": float(dices_np.mean()),
        "median_dice": float(np.median(dices_np)),
        "std_dice": float(dices_np.std(ddof=0)),
        "min_dice": float(dices_np.min()),
        "max_dice": float(dices_np.max()),
        "global_dice": float(global_dice),
    }])

    df_meta = pd.DataFrame([
        {"key": "timestamp", "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "folder_path", "value": os.path.abspath(folder_path)},
        {"key": "split", "value": split},
        {"key": "model_ckpt", "value": getattr(configs, "BEST_MODEL_SAVE_PATH", "")},
        {"key": "threshold", "value": THRESHOLD},
        {"key": "use_tta", "value": USE_TTA},
        {"key": "use_lcc", "value": USE_LCC},
        {"key": "use_morph", "value": USE_MORPH},
        {"key": "use_crf", "value": USE_CRF},
        {"key": "use_ellipse", "value": USE_ELLIPSE},
        {"key": "image_size", "value": getattr(configs, "IMAGE_SIZE", None)},
        {"key": "pre_resizer_size", "value": getattr(configs, "IMAGE_RESIZE_PRE_LEARANBLE_RESIZER", None)},
        {"key": "device", "value": str(configs.DEVICE)},
    ])

    # Determine Excel path
    if excel_path is None:
        excel_path = os.path.join(folder_path, f"dice_{split}_results.xlsx")

    # Write Excel
    try:
        with pd.ExcelWriter(excel_path) as writer:
            df_per.to_excel(writer, index=False, sheet_name="per_image")
            df_summary.to_excel(writer, index=False, sheet_name="summary")
            df_meta.to_excel(writer, index=False, sheet_name="meta")
        print(f"\nSaved results to: {excel_path}")
    except Exception as e:
        # Optional CSV fallback if Excel engine isn't available
        base = os.path.splitext(excel_path)[0]
        df_per.to_csv(base + "_per_image.csv", index=False)
        df_summary.to_csv(base + "_summary.csv", index=False)
        df_meta.to_csv(base + "_meta.csv", index=False)
        print(f"\n[warn] Excel write failed ({e}). Wrote CSVs to '{base}_*.csv' instead.")

    # Also print a quick summary to stdout
    print("\n=== Summary ===")
    print(f"N = {len(dices)}")
    print(f"Mean Dice   : {dices_np.mean():.4f}")
    print(f"Median Dice : {np.median(dices_np):.4f}")
    print(f"Std Dice    : {dices_np.std(ddof=0):.4f}")
    print(f"Min / Max   : {dices_np.min():.4f} / {dices_np.max():.4f}")
    print(f"Global Dice : {global_dice:.4f}")


# ============================= Minimal call site ===========================
if __name__ == "__main__":
    # Set the root folder that contains {training_set, validation_set, test_set}
    FOLDER_PATH = "data/"          # e.g., "data/" or "/abs/path/to/HC18/"
    SPLIT = "validation"           # "train" | "validation" | "test"
    EXCEL_PATH = None              # or e.g., "results/dice_validation.xlsx"
    compute_dice_for_split(FOLDER_PATH, SPLIT, EXCEL_PATH)
