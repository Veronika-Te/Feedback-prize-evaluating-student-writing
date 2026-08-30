import os
import sys
import argparse
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoTokenizer

from src.model import FeedbackDebertaModel
from src.dataset import NUM_LABELS, LABEL2ID, DISCOURSE_TYPES
from src.postprocess import (
    SmartSpanResMLP,
    postprocess_with_resmlp,
    postprocess_with_thresholds,
)


def resolve_default_model_path() -> str:
    candidates = [
        "output/best_deberta_v3_base.pt",
        "output/best_deberta_v3_base_f1_06255.pt",
        "outputs_deberta_optimized/best_deberta_v3_base.pt",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Predict Discourse Elements in Essay using DeBERTa & Smart ResMLP")
    parser.add_argument("--text", type=str, required=True, help="Raw Essay Text")
    parser.add_argument("--model_path", type=str, default=resolve_default_model_path(), help="Path to DeBERTa .pt weights")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="Pretrained model identifier")
    parser.add_argument("--resmlp_path", type=str, default="output/best_resmlp.pt", help="Path to Smart ResMLP .pt weights (optional)")
    parser.add_argument("--method", type=str, choices=["resmlp", "thresholds"], default="resmlp", help="Postprocessing method: 'resmlp' or 'thresholds'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer and DeBERTa model
    print(f"Loading backbone '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = FeedbackDebertaModel(args.model_name, NUM_LABELS).to(device)

    if os.path.exists(args.model_path) and os.path.getsize(args.model_path) > 0:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded DeBERTa weights from: '{args.model_path}'")
    else:
        print(f"'{args.model_path}' not found or empty. Running with base pretrained weights.")

    model.eval()

    # Load Smart ResMLP model if selected and checkpoint exists
    resmlp_model = None
    if args.method == "resmlp":
        if os.path.exists(args.resmlp_path) and os.path.getsize(args.resmlp_path) > 0:
            resmlp_model = SmartSpanResMLP(input_dim=33, hidden_dim=128).to(device)
            resmlp_model.load_state_dict(torch.load(args.resmlp_path, map_location=device))
            resmlp_model.eval()
            print(f"Loaded Smart ResMLP ranker weights from: '{args.resmlp_path}'")
        else:
            print(f"ResMLP checkpoint '{args.resmlp_path}' not found or empty. Using threshold-based postprocessing.")

    words = args.text.split()
    if not words:
        print("Error: Text is empty.")
        return

    # Tokenization
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    word_ids = tokenized.word_ids(batch_index=0)

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)[0]
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

    # Subword mean pooling
    word_prob_accum = {w_idx: [] for w_idx in range(len(words))}
    for token_idx, w_id in enumerate(word_ids):
        if w_id is not None and w_id < len(words):
            word_prob_accum[w_id].append(probs[token_idx])

    word_probs = np.zeros((len(words), NUM_LABELS), dtype=np.float32)
    for w_idx in range(len(words)):
        word_probs[w_idx] = np.mean(word_prob_accum[w_idx], axis=0) if len(word_prob_accum[w_idx]) > 0 else np.array([1.0] + [0.0] * (NUM_LABELS - 1))

    class_probs = np.zeros((len(words), len(DISCOURSE_TYPES)), dtype=np.float32)
    for c_idx, dt in enumerate(DISCOURSE_TYPES):
        class_probs[:, c_idx] = word_probs[:, LABEL2ID[f"B-{dt}"]] + word_probs[:, LABEL2ID[f"I-{dt}"]]

    # Post-processing
    if args.method == "resmlp" and resmlp_model is not None:
        print("Applying Smart ResMLP Neural Span Ranking...")
        spans = postprocess_with_resmlp(words, word_probs, class_probs, resmlp_model=resmlp_model, device=device)
    else:
        print("Applying Thresholds Post-Processing...")
        spans = postprocess_with_thresholds(words, word_probs, class_probs)

    # Output results
    print("\n" + "=" * 80)
    print(f"PREDICTED DISCOURSE SPANS ({len(spans)} Spans Found):")
    print("=" * 80)
    if not spans:
        print("No specific discourse spans detected above thresholds.")
    for sp in spans:
        span_text = " ".join(words[sp["start_word"]:sp["end_word"] + 1])
        conf_str = f" [Conf: {sp['neural_conf']:.1f}%]" if "neural_conf" in sp else ""
        print(f"[{sp['type']:<20}]{conf_str} (Words {sp['start_word']:>3}-{sp['end_word']:>3}): \"{span_text}\"")


if __name__ == "__main__":
    main()