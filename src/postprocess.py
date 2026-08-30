from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

DISCOURSE_TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]

LABELS = ["O"]
for dt in DISCOURSE_TYPES:
    LABELS.extend([f"B-{dt}", f"I-{dt}"])

LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

FINAL_OPTIMAL_THRESHOLDS = {
    "O": 0.64,
    "Evidence": 0.56,
    "Lead": 0.28,
    "Concluding Statement": 0.32,
    "Position": 0.28,
    "Claim": 0.36,
    "Counterclaim": 0.26,
    "Rebuttal": 0.20,
}

EDA_MIN_LENGTHS = {
    "Evidence": 21,
    "Concluding Statement": 14,
    "Lead": 14,
    "Position": 6,
    "Claim": 4,
    "Counterclaim": 10,
    "Rebuttal": 4,
}
FINAL_OPTIMAL_MIN_LENS = EDA_MIN_LENGTHS

RELAXED_CANDIDATE_THRESHOLDS = {
    "O": 0.50,
    "Evidence": 0.30,
    "Lead": 0.25,
    "Concluding Statement": 0.25,
    "Position": 0.25,
    "Claim": 0.25,
    "Counterclaim": 0.20,
    "Rebuttal": 0.15,
}

NEURAL_KEEP_THRESHOLDS = {
    "Lead": 0.40,
    "Position": 0.35,
    "Claim": 0.28,
    "Counterclaim": 0.22,
    "Rebuttal": 0.16,
    "Evidence": 0.38,
    "Concluding Statement": 0.40,
}


def merge_nearby_spans(
    spans: List[Dict[str, Any]],
    words: Optional[List[str]] = None,
    max_gap: int = 3,
) -> List[Dict[str, Any]]:
    if not spans:
        return []

    merge_allowed = {"Evidence", "Concluding Statement"}
    sorted_spans = sorted(spans, key=lambda s: (s["start_word"], s["end_word"]))
    merged = [dict(sorted_spans[0])]

    for next_span in sorted_spans[1:]:
        prev = merged[-1]
        gap = next_span["start_word"] - prev["end_word"] - 1

        if (next_span["type"] == prev["type"]) and (prev["type"] in merge_allowed) and (0 <= gap <= max_gap):
            prev["end_word"] = max(prev["end_word"], next_span["end_word"])
            if words:
                prev["text"] = " ".join(words[prev["start_word"]:prev["end_word"] + 1])
            prev["predictionstring"] = " ".join(str(x) for x in range(prev["start_word"], prev["end_word"] + 1))
        else:
            merged.append(dict(next_span))
    return merged


def snap_to_sentence_delimiters(
    spans: List[Dict[str, Any]],
    words: List[str],
) -> List[Dict[str, Any]]:
    delimiters = (".", "?", "!")
    snapped = []
    for sp in spans:
        s_idx, e_idx = sp["start_word"], sp["end_word"]
        best_end = e_idx
        for i in range(e_idx - 1, max(s_idx, e_idx - 3) - 1, -1):
            if any(words[i].endswith(ch) for ch in delimiters):
                best_end = i
                break

        sp_copy = dict(sp)
        sp_copy["end_word"] = best_end
        if words:
            sp_copy["text"] = " ".join(words[s_idx:best_end + 1])
        sp_copy["predictionstring"] = " ".join(str(x) for x in range(s_idx, best_end + 1))
        snapped.append(sp_copy)
    return snapped


def extract_candidate_spans(
    words: List[str],
    word_probs: np.ndarray,
    class_probs: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    min_lengths: Optional[Dict[str, int]] = None,
    max_merge_gap: int = 3,
) -> List[Dict[str, Any]]:
    if thresholds is None:
        thresholds = RELAXED_CANDIDATE_THRESHOLDS
    if min_lengths is None:
        min_lengths = {}

    n_words = len(words)
    if n_words == 0:
        return []

    pred_tags = ["O"] * n_words
    for i in range(n_words):
        if word_probs[i, 0] < thresholds.get("O", 0.50):
            best_c = int(np.argmax(class_probs[i]))
            best_dt = DISCOURSE_TYPES[best_c]
            if class_probs[i, best_c] >= thresholds.get(best_dt, 0.25):
                b_idx = LABEL2ID[f"B-{best_dt}"]
                if word_probs[i, b_idx] > 0.28 or (i > 0 and not pred_tags[i - 1].endswith(best_dt)):
                    pred_tags[i] = f"B-{best_dt}"
                else:
                    pred_tags[i] = f"I-{best_dt}"

    spans = []
    curr_type, curr_start = None, None

    for i, tag in enumerate(pred_tags):
        if tag.startswith("B-"):
            if curr_type is not None:
                spans.append({"type": curr_type, "start_word": curr_start, "end_word": i - 1})
            curr_type = tag[2:]
            curr_start = i
        elif tag.startswith("I-"):
            if curr_type != tag[2:]:
                if curr_type is not None:
                    spans.append({"type": curr_type, "start_word": curr_start, "end_word": i - 1})
                curr_type = tag[2:]
                curr_start = i
        else:
            if curr_type is not None:
                spans.append({"type": curr_type, "start_word": curr_start, "end_word": i - 1})
                curr_type = None

    if curr_type is not None:
        spans.append({"type": curr_type, "start_word": curr_start, "end_word": n_words - 1})

    for sp in spans:
        sp["text"] = " ".join(words[sp["start_word"]:sp["end_word"] + 1])
        sp["predictionstring"] = " ".join(str(x) for x in range(sp["start_word"], sp["end_word"] + 1))

    if min_lengths:
        spans = [sp for sp in spans if (sp["end_word"] - sp["start_word"] + 1) >= min_lengths.get(sp["type"], 1)]

    if max_merge_gap > 0 and spans:
        spans = merge_nearby_spans(spans, words, max_gap=max_merge_gap)

    return spans


get_spans_from_cached = extract_candidate_spans


def extract_smart_features(
    words_or_item: Any,
    candidate_spans_or_probs: Any,
    class_probs: Optional[np.ndarray] = None,
    candidate_spans: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    if isinstance(words_or_item, dict):
        words = words_or_item["words"]
        word_probs = words_or_item["word_probs"]
        c_probs = words_or_item["class_probs"]
        spans = candidate_spans_or_probs
    else:
        words = words_or_item
        word_probs = candidate_spans_or_probs
        c_probs = class_probs
        spans = candidate_spans

    if not spans:
        return np.empty((0, 33), dtype=np.float32)

    n_words = len(words)
    total_spans = len(spans)
    features_list = []

    for sp_idx, sp in enumerate(spans):
        s_idx = sp["start_word"]
        e_idx = sp["end_word"]
        c_idx = DISCOURSE_TYPES.index(sp["type"])
        span_len = e_idx - s_idx + 1

        span_c_probs = c_probs[s_idx:e_idx + 1, c_idx]
        mean_p = float(np.mean(span_c_probs))
        std_p = float(np.std(span_c_probs))
        max_p = float(np.max(span_c_probs))
        min_p = float(np.min(span_c_probs))
        median_p = float(np.median(span_c_probs))
        start_p = float(span_c_probs[0])
        end_p = float(span_c_probs[-1])

        mean_all_classes = np.mean(c_probs[s_idx:e_idx + 1], axis=0)
        sorted_c = np.sort(mean_all_classes)
        margin_top1_top2 = float(sorted_c[-1] - sorted_c[-2]) if len(sorted_c) > 1 else 1.0
        entropy = float(-np.sum(mean_all_classes * np.log(mean_all_classes + 1e-7)))

        prev_3_o = float(np.mean(word_probs[max(0, s_idx - 3):s_idx, 0])) if s_idx > 0 else 1.0
        next_3_o = float(np.mean(word_probs[e_idx + 1:min(n_words, e_idx + 4), 0])) if e_idx < n_words - 1 else 1.0
        prev_3_class = float(np.mean(c_probs[max(0, s_idx - 3):s_idx, c_idx])) if s_idx > 0 else 0.0
        next_3_class = float(np.mean(c_probs[e_idx + 1:min(n_words, e_idx + 4), c_idx])) if e_idx < n_words - 1 else 0.0
        first_word_o = float(word_probs[s_idx, 0])
        last_word_o = float(word_probs[e_idx, 0])

        ends_with_punct = 1.0 if any(words[e_idx].endswith(ch) for ch in (".", "?", "!")) else 0.0
        starts_with_capital = 1.0 if words[s_idx][0].isupper() else 0.0
        has_comma_inside = 1.0 if any("," in words[i] for i in range(s_idx, e_idx + 1)) else 0.0
        has_quotes = 1.0 if any('"' in words[i] or "'" in words[i] for i in range(s_idx, e_idx + 1)) else 0.0

        start_ratio = s_idx / max(1, n_words)
        end_ratio = e_idx / max(1, n_words)
        len_ratio = span_len / max(1, n_words)
        span_order_ratio = sp_idx / max(1, total_spans)
        log_len = float(np.log1p(span_len))
        sqrt_len = float(np.sqrt(span_len))

        one_hot = [0.0] * len(DISCOURSE_TYPES)
        one_hot[c_idx] = 1.0

        feat_vec = [
            mean_p, std_p, max_p, min_p, median_p, start_p, end_p,
            margin_top1_top2, entropy,
            prev_3_o, next_3_o, prev_3_class, next_3_class, first_word_o, last_word_o,
            ends_with_punct, starts_with_capital, has_comma_inside, has_quotes,
            start_ratio, end_ratio, len_ratio, span_order_ratio, span_len, log_len, sqrt_len,
        ] + one_hot

        features_list.append(feat_vec)

    return np.array(features_list, dtype=np.float32)


class SmartSpanResMLP(nn.Module):
    def __init__(self, input_dim: int = 33, hidden_dim: int = 128):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )
        self.downsample = nn.Linear(hidden_dim, 64)

        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4]])
        self.out_head = nn.Linear(64, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.act(x + self.res1(x))

        identity = self.downsample(x)
        x = self.act(identity + self.res2(x))

        logits = torch.mean(torch.stack([self.out_head(d(x)) for d in self.dropouts], dim=0), dim=0)
        return torch.sigmoid(logits).squeeze(-1)


def postprocess_with_thresholds(
    words: List[str],
    word_probs: np.ndarray,
    class_probs: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    min_lengths: Optional[Dict[str, int]] = None,
    max_merge_gap: int = 3,
) -> List[Dict[str, Any]]:
    if thresholds is None:
        thresholds = FINAL_OPTIMAL_THRESHOLDS
    if min_lengths is None:
        min_lengths = FINAL_OPTIMAL_MIN_LENS

    spans = extract_candidate_spans(
        words,
        word_probs,
        class_probs,
        thresholds=thresholds,
        min_lengths=min_lengths,
        max_merge_gap=max_merge_gap,
    )

    if spans:
        spans = snap_to_sentence_delimiters(spans, words)

    n_words = len(words)
    valid_spans = []
    for sp in spans:
        if sp["type"] == "Lead" and (sp["start_word"] / max(1, n_words)) > 0.50:
            continue
        if sp["type"] == "Concluding Statement" and (sp["end_word"] / max(1, n_words)) < 0.30:
            continue
        valid_spans.append(sp)

    return valid_spans


def postprocess_with_resmlp(
    words: List[str],
    word_probs: np.ndarray,
    class_probs: np.ndarray,
    resmlp_model: Optional[SmartSpanResMLP] = None,
    device: Optional[torch.device] = None,
    keep_thresholds: Optional[Dict[str, float]] = None,
    relaxed_thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if keep_thresholds is None:
        keep_thresholds = NEURAL_KEEP_THRESHOLDS
    if relaxed_thresholds is None:
        relaxed_thresholds = RELAXED_CANDIDATE_THRESHOLDS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_spans = extract_candidate_spans(
        words,
        word_probs,
        class_probs,
        thresholds=relaxed_thresholds,
        min_lengths={},
        max_merge_gap=3,
    )

    if not raw_spans:
        return []

    if resmlp_model is None:
        return postprocess_with_thresholds(words, word_probs, class_probs)

    features = extract_smart_features(words, word_probs, class_probs, raw_spans)
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    resmlp_model.eval()
    with torch.no_grad():
        conf_scores = resmlp_model(features_tensor)
        if conf_scores.ndim == 0:
            conf_scores = conf_scores.unsqueeze(0)
        conf_scores = conf_scores.cpu().numpy()

    filtered_spans = []
    for sp, score in zip(raw_spans, conf_scores):
        score_val = float(score)
        req_thresh = keep_thresholds.get(sp["type"], 0.30)
        if score_val >= req_thresh:
            sp_copy = dict(sp)
            sp_copy["neural_conf"] = score_val * 100.0
            filtered_spans.append(sp_copy)

    if filtered_spans:
        filtered_spans = snap_to_sentence_delimiters(filtered_spans, words)

    n_words = len(words)
    final_spans = []
    for sp in filtered_spans:
        if sp["type"] == "Lead" and (sp["start_word"] / max(1, n_words)) > 0.50:
            continue
        if sp["type"] == "Concluding Statement" and (sp["end_word"] / max(1, n_words)) < 0.30:
            continue
        final_spans.append(sp)

    return final_spans