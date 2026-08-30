import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

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


def get_essay_text(essay_id: str, txt_dir: Union[str, Path]) -> str:
    file_path = Path(txt_dir) / f"{essay_id}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def build_word_labels(essay_id: str, text: str, grouped_df: dict) -> Tuple[List[str], List[str]]:
    words = text.split()
    word_labels = ["O"] * len(words)
    if essay_id in grouped_df:
        for _, row in grouped_df[essay_id].iterrows():
            dtype = row["discourse_type"]
            w_indices = [int(x) for x in str(row["predictionstring"]).split() if x.isdigit()]
            if not w_indices:
                continue
            if w_indices[0] < len(word_labels):
                word_labels[w_indices[0]] = f"B-{dtype}"
            for idx in w_indices[1:]:
                if idx < len(word_labels):
                    word_labels[idx] = f"I-{dtype}"
    return words, word_labels


def prepare_all_features(
    essay_ids: List[str],
    df: pd.DataFrame,
    tokenizer: Any,
    txt_dir: Union[str, Path],
    max_len: int = 1024,
) -> List[Dict[str, Any]]:
    features = []
    g_df = dict(tuple(df[df["id"].isin(set(essay_ids))].groupby("id")))

    for eid in essay_ids:
        text = get_essay_text(eid, txt_dir)
        words, word_labels = build_word_labels(eid, text, g_df)

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        for w_id in word_ids:
            if w_id is None:
                aligned_labels.append(-100)
            else:
                label_name = word_labels[w_id] if w_id < len(word_labels) else "O"
                aligned_labels.append(LABEL2ID[label_name])

        features.append({
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "length": len(encoding["input_ids"]),
        })
    return features


class FastDynamicCollator:
    def __init__(self, pad_token_id: int = 0, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=self.label_pad_id),
        }