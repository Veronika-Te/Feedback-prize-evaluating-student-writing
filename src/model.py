import math
from typing import List, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoModel


class FeedbackDebertaModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 15,
        lstm_hidden: int = 512,
        dropouts: Optional[List[float]] = None,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
        })
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p) for p in (dropouts or [0.1, 0.2, 0.3, 0.4, 0.5])]
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
        self._init_weights(self.classifier)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0).float()
        lstm_out, _ = self.bilstm(hidden_states)
        logits = torch.mean(
            torch.stack([self.classifier(d(lstm_out)) for d in self.dropouts], dim=0),
            dim=0,
        )
        return logits


FeedbackCustomDeberta = FeedbackDebertaModel


def get_optimizer_grouped_parameters(
    model: nn.Module,
    encoder_lr: float = 2.0e-5,
    head_lr: float = 1.0e-4,
    weight_decay: float = 0.01,
    llrd_decay: float = 0.90,
) -> List[dict]:
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_parameters = []

    head_params = list(model.bilstm.named_parameters()) + list(model.classifier.named_parameters())
    optimizer_parameters.extend([
        {"params": [p for n, p in head_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": head_lr},
        {"params": [p for n, p in head_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": head_lr},
    ])

    embed_params = list(model.transformer.embeddings.named_parameters())
    optimizer_parameters.extend([
        {"params": [p for n, p in embed_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": encoder_lr * (llrd_decay ** 12)},
        {"params": [p for n, p in embed_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": encoder_lr * (llrd_decay ** 12)},
    ])

    layers = model.transformer.encoder.layer
    num_layers = len(layers)
    for i, layer in enumerate(layers):
        layer_lr = encoder_lr * (llrd_decay ** (num_layers - 1 - i))
        layer_params = list(layer.named_parameters())
        optimizer_parameters.extend([
            {"params": [p for n, p in layer_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": layer_lr},
            {"params": [p for n, p in layer_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": layer_lr},
        ])

    return optimizer_parameters


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)