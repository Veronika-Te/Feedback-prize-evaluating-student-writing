import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch

DISCOURSE_TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]

COLORS = {
    "Lead": "#ffeaa7",
    "Position": "#55efc4",
    "Claim": "#81ecec",
    "Counterclaim": "#fab1a0",
    "Rebuttal": "#ff7675",
    "Evidence": "#74b9ff",
    "Concluding Statement": "#a29bfe",
}


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_kaggle_dataset_paths() -> Tuple[Path, Path]:
    csv_candidates = [
        Path("../data/raw/train.csv"),
        Path("data/raw/train.csv"),
        Path("data/train.csv"),
        Path("train.csv"),
        Path("/content/data/raw/train.csv"),
        Path("/content/train.csv"),
    ]
    txt_dir_candidates = [
        Path("../data/raw/train"),
        Path("data/raw/train"),
        Path("data/train"),
        Path("train"),
        Path("/content/data/raw/train"),
        Path("/content/train"),
    ]
    found_csv = next((p.resolve() for p in csv_candidates if p.is_file()), None)
    found_txt_dir = next((p.resolve() for p in txt_dir_candidates if p.is_dir() and len(os.listdir(p)) > 0), None)

    if found_csv is None or found_txt_dir is None:
        raise FileNotFoundError("train.csv or train/ directory not found.")
    return found_csv, found_txt_dir


def compute_overlap_ratio(pred_span: Tuple[int, int], true_span: Tuple[int, int]) -> float:
    p_start, p_end = pred_span
    t_start, t_end = true_span
    intersection = max(0, min(p_end, t_end) - max(p_start, t_start) + 1)
    union = max(p_end, t_end) - min(p_start, t_start) + 1
    return intersection / union if union > 0 else 0.0


def score_feedback_comp(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    overlap_threshold: float = 0.5,
) -> Tuple[float, pd.DataFrame]:
    class_metrics = []
    f1_scores = []

    for discourse_type in DISCOURSE_TYPES:
        pred_type_df = pred_df[pred_df["discourse_type"] == discourse_type] if not pred_df.empty else pd.DataFrame()
        gt_type_df = gt_df[gt_df["discourse_type"] == discourse_type] if not gt_df.empty else pd.DataFrame()

        pred_grouped = pred_type_df.groupby("id") if not pred_type_df.empty else None
        gt_grouped = gt_type_df.groupby("id") if not gt_type_df.empty else None

        all_essay_ids = set(gt_df["id"].unique()).union(set(pred_df["id"].unique()) if not pred_df.empty else set())
        tp, fp, fn = 0, 0, 0

        for essay_id in all_essay_ids:
            preds = pred_grouped.get_group(essay_id) if (pred_grouped is not None and essay_id in pred_grouped.groups) else pd.DataFrame()
            gts = gt_grouped.get_group(essay_id) if (gt_grouped is not None and essay_id in gt_grouped.groups) else pd.DataFrame()

            pred_spans = [
                (int(str(ps).split()[0]), int(str(ps).split()[-1]))
                for ps in preds["predictionstring"].values
                if len(str(ps).split()) > 0
            ] if not preds.empty else []

            gt_spans = [
                (int(str(ps).split()[0]), int(str(ps).split()[-1]))
                for ps in gts["predictionstring"].values
                if len(str(ps).split()) > 0
            ] if not gts.empty else []

            matched_gt = set()
            for p_span in pred_spans:
                best_iou = 0.0
                best_gt_idx = -1
                for idx, g_span in enumerate(gt_spans):
                    if idx in matched_gt:
                        continue
                    iou = compute_overlap_ratio(p_span, g_span)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                if best_iou >= overlap_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            fn += len(gt_spans) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        class_metrics.append({"Discourse Type": discourse_type, "Precision": precision, "Recall": recall, "F1-Score": f1})

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return macro_f1, pd.DataFrame(class_metrics)


def render_discourse_html(
    text: str,
    spans: List[Dict[str, Any]],
    title: str = "Discourse Elements",
    badge_color: str = "#2d3436",
) -> str:
    words = text.split()
    word_meta = [None] * len(words)

    for sp in spans:
        dtype = sp.get("type", sp.get("discourse_type"))
        s_idx = sp["start_word"]
        e_idx = sp["end_word"]
        conf = sp.get("neural_conf", None)
        tag_text = f"{dtype} ({conf:.0f}%)" if conf is not None else dtype
        for i in range(s_idx, min(e_idx + 1, len(words))):
            word_meta[i] = (dtype, tag_text)

    legend_items = [
        f"<span style='background-color: {col}; color: #2d3436; padding: 3px 8px; margin: 2px; border-radius: 6px; font-weight: bold; font-size: 11px; display: inline-block;'>{dt}</span>"
        for dt, col in COLORS.items()
    ]
    legend_html = "".join(legend_items)

    html_chunks = [
        "<div style='border: 2px solid #dfe6e9; border-radius: 12px; padding: 18px; background: #ffffff; margin-bottom: 20px; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>",
        "<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #edf2f7; padding-bottom: 10px; margin-bottom: 12px;'>",
        f"<h3 style='margin: 0; color: {badge_color}; font-size: 16px;'>{title}</h3>",
        f"<div>{legend_html}</div>",
        "</div>",
        "<div style='line-height: 2.4; font-size: 15px; color: #2d3436; text-align: justify;'>",
    ]

    cur_type = None
    cur_tag_text = None
    for w, meta in zip(words, word_meta):
        dt = meta[0] if meta else None
        tag_t = meta[1] if meta else None
        if dt != cur_type:
            if cur_type is not None:
                html_chunks.append(f" <b style='font-size: 10px; color: #2d3436; background: rgba(255,255,255,0.95); padding: 1px 5px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.15);'>[{cur_tag_text}]</b></span> ")
            if dt is not None:
                color = COLORS.get(dt, "#dfe6e9")
                html_chunks.append(f"<span style='background-color: {color}; padding: 3px 6px; border-radius: 5px; margin: 0 1px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>")
            cur_type = dt
            cur_tag_text = tag_t
        html_chunks.append(w + " ")

    if cur_type is not None:
        html_chunks.append(f" <b style='font-size: 10px; color: #2d3436; background: rgba(255,255,255,0.95); padding: 1px 5px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.15);'>[{cur_tag_text}]</b></span>")

    html_chunks.append("</div></div>")
    return "".join(html_chunks)