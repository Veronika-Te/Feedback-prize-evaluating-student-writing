import os
import sys
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr
from transformers import AutoTokenizer

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

from src.model import FeedbackDebertaModel
from src.dataset import NUM_LABELS, LABEL2ID, DISCOURSE_TYPES
from src.postprocess import (
    SmartSpanResMLP,
    postprocess_with_resmlp,
    postprocess_with_thresholds,
)
from src.utils import COLORS


def resolve_model_path() -> str:
    candidates = [
        "output/best_deberta_v3_base.pt",
        "outputs_deberta_optimized/best_deberta_v3_base.pt",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return candidates[0]


MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_PATH = resolve_model_path()
RESMLP_PATH = "output/best_resmlp.pt"

print("Initializing models into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = FeedbackDebertaModel(MODEL_NAME, NUM_LABELS)
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print(f"Loaded DeBERTa weights: {MODEL_PATH}")
else:
    print(f"DeBERTa checkpoint not found at {MODEL_PATH}. Running with base pretrained weights.")
model.eval()

resmlp_model = SmartSpanResMLP(input_dim=33, hidden_dim=128)
has_resmlp_weights = False
if os.path.exists(RESMLP_PATH) and os.path.getsize(RESMLP_PATH) > 0:
    resmlp_model.load_state_dict(torch.load(RESMLP_PATH, map_location="cpu"))
    has_resmlp_weights = True
    print(f"Loaded Smart ResMLP weights: {RESMLP_PATH}")
else:
    print("ResMLP checkpoint not found or empty.")
resmlp_model.eval()


def generate_html_view(text, spans):
    words = text.split()
    word_meta = [None] * len(words)

    for sp in spans:
        dtype = sp.get("type", sp.get("discourse_type"))
        s_idx = sp["start_word"]
        e_idx = sp["end_word"]
        conf = sp.get("neural_conf", None)
        badge_text = f"{dtype} ({conf:.0f}%)" if conf is not None else dtype
        for i in range(s_idx, min(e_idx + 1, len(words))):
            word_meta[i] = (dtype, badge_text)

    legend_items = [
        f"<span style='background-color: {col}; color: #2d3436; padding: 4px 10px; margin: 3px; border-radius: 6px; font-weight: bold; font-size: 12px; display: inline-block;'>{dt}</span>"
        for dt, col in COLORS.items()
    ]
    legend_html = "".join(legend_items)

    html_chunks = [
        "<div style='border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; background: #ffffff; font-family: -apple-system, BlinkMacSystemFont, sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>",
        f"<div style='border-bottom: 1px solid #eee; padding-bottom: 12px; margin-bottom: 15px;'>{legend_html}</div>",
        "<div style='line-height: 2.5; font-size: 16px; color: #2d3436; text-align: justify;'>",
    ]

    cur_type = None
    cur_badge = None
    for w, meta in zip(words, word_meta):
        dt = meta[0] if meta else None
        badge = meta[1] if meta else None
        if dt != cur_type:
            if cur_type is not None:
                html_chunks.append(f" <b style='font-size: 11px; color: #2d3436; background: rgba(255,255,255,0.95); padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.15);'>[{cur_badge}]</b></span> ")
            if dt is not None:
                color = COLORS.get(dt, "#dfe6e9")
                html_chunks.append(f"<span style='background-color: {color}; padding: 4px 8px; border-radius: 6px; margin: 0 1px;'>")
            cur_type = dt
            cur_badge = badge
        html_chunks.append(w + " ")

    if cur_type is not None:
        html_chunks.append(f" <b style='font-size: 11px; color: #2d3436; background: rgba(255,255,255,0.95); padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.15);'>[{cur_badge}]</b></span>")

    html_chunks.append("</div></div>")
    return "".join(html_chunks)


if USING_SPACES:
    @spaces.GPU(duration=60)
    def analyze_essay(text, method="Smart ResMLP (Best)"):
        return _run_inference(text, method)
else:
    def analyze_essay(text, method="Smart ResMLP (Best)"):
        return _run_inference(text, method)


def _run_inference(text, method):
    if not text.strip():
        return "<p style='color: red;'>Please enter essay text.</p>", ""

    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(curr_device)
    resmlp_model.to(curr_device)

    words = text.split()
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=1024,
        padding=False,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(curr_device)
    attention_mask = tokenized["attention_mask"].to(curr_device)
    word_ids = tokenized.word_ids(batch_index=0)

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
    if "ResMLP" in method and has_resmlp_weights:
        spans = postprocess_with_resmlp(words, word_probs, class_probs, resmlp_model=resmlp_model, device=curr_device)
    else:
        spans = postprocess_with_thresholds(words, word_probs, class_probs)

    html_output = generate_html_view(text, spans)

    summary_text = f"Total Words: {len(words)} | Elements Found: {len(spans)} | Method: {method}\n"
    for sp in spans:
        conf_str = f" [Confidence: {sp['neural_conf']:.1f}%]" if "neural_conf" in sp else ""
        summary_text += f"\n- [{sp['type']}]{conf_str} (Words {sp['start_word']}-{sp['end_word']})"

    return html_output, summary_text


SAMPLE_ESSAY = """In recent years, distance learning has become increasingly common around the world. While traditional classrooms have their merits, online education is the most effective approach for modern students. First, online platforms offer unmatched flexibility, allowing students to study at their own pace and balance work with education. According to a recent study, 72% of online learners reported higher productivity when managing their own schedules. Furthermore, virtual classes save hours of daily commuting time. Although critics argue that online learning reduces face-to-face social interaction, digital discussion forums and video calls adequately bridge this gap. Ultimately, educational institutions should expand their online degree options to meet the needs of today's digital world."""

with gr.Blocks(title="Discourse AI Analyzer") as demo:
    gr.Markdown("# Feedback Prize — Student Essay Discourse Analyzer")
    gr.Markdown("Identify argumentative discourse elements (**Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement**) using fine-tuned **DeBERTa-v3** and **Smart ResMLP Neural Ranker**.")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="Paste Essay Here", value=SAMPLE_ESSAY, lines=12, placeholder="Type or paste essay text...")
            method_choice = gr.Radio(
                choices=["Smart ResMLP (Best)", "Thresholds Baseline"],
                value="Smart ResMLP (Best)",
                label="Post-Processing Method",
            )
            submit_btn = gr.Button("Analyze Essay Discourse", variant="primary")
            summary_box = gr.Textbox(label="Extraction Summary", lines=6, interactive=False)

        with gr.Column(scale=1):
            html_display = gr.HTML(label="Visual Discourse Elements")

    submit_btn.click(fn=analyze_essay, inputs=[input_text, method_choice], outputs=[html_display, summary_box])

if __name__ == "__main__":
    demo.launch()