"""Paper-aligned PeakST training operations for local experiment archives.

The public entry point executes one already chosen model configuration.  The
paper's source-side validation, blocked target-reference selection, and
multi-site aggregation are implemented separately in :mod:`peakst.selection`
and :mod:`peakst.reporting` so that the later test labels cannot enter model
choice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn

from .model import (
    PeakLoss,
    blend_small_cells,
    boundary_continuous_return,
)


@dataclass
class Transform:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Transform":
        values = np.asarray(features, dtype=float)
        median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(values), values, median)
        return cls(
            median=median,
            mean=filled.mean(axis=0),
            scale=np.maximum(filled.std(axis=0), 1e-6),
        )

    def __call__(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        missing = ~np.isfinite(values)
        scaled = (
            np.where(missing, self.median, values) - self.mean
        ) / self.scale
        return np.column_stack([np.clip(scaled, -20.0, 20.0), missing]).astype(
            "float32"
        )


class MLP(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        blocks: int,
        width: int,
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(blocks):
            layers.extend(
                [nn.Linear(n_inputs, width), nn.ReLU(), nn.Dropout(dropout)]
            )
            n_inputs = width
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(n_inputs, n_outputs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.backbone(features))[:, None, :]


def site_equal_weights(site: np.ndarray) -> np.ndarray:
    site = np.asarray(site)
    result = np.zeros(len(site), dtype="float32")
    unique, counts = np.unique(site, return_counts=True)
    for value, count in zip(unique, counts):
        result[site == value] = len(site) / (len(unique) * count)
    return result


def site_and_class_equal_weights(
    site: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Give sites equal total weight and classes equal weight within the pool."""

    site = np.asarray(site)
    labels = np.asarray(labels, dtype=bool)
    result = np.zeros(len(site), dtype=float)
    sites = np.unique(site)
    for site_value in sites:
        for class_value in (False, True):
            selected = (site == site_value) & (labels == class_value)
            count = int(selected.sum())
            if count == 0:
                raise ValueError("each source site must contain both high-state classes")
            result[selected] = len(site) / (len(sites) * 2.0 * count)
    return result.astype("float32")


def grouped_high_state(
    small_concentration: np.ndarray,
    site: np.ndarray,
    year: np.ndarray,
    quantile: float,
) -> np.ndarray:
    labels = np.zeros(len(small_concentration), dtype=bool)
    for site_value in np.unique(site):
        for year_value in np.unique(year[site == site_value]):
            selected = (site == site_value) & (year == year_value)
            threshold = np.quantile(small_concentration[selected], quantile)
            labels[selected] = small_concentration[selected] >= threshold
    return labels


def make_model(n_inputs: int, config: dict, seed: int, device: torch.device):
    torch.manual_seed(seed)
    backbone = str(config.get("backbone", "mlp")).lower()
    if backbone == "mlp":
        model = MLP(
            n_inputs=n_inputs,
            n_outputs=64,
            blocks=int(config["hidden_blocks"]),
            width=int(config["hidden_width"]),
            dropout=float(config["dropout"]),
        )
    elif backbone == "tabm":
        try:
            from tabm import TabM
            import rtdl_num_embeddings as embeddings
        except ImportError as error:
            raise ImportError(
                "TabM requires `python -m pip install -e '.[tabm]'`"
            ) from error
        embedding_name = str(config.get("embedding", "periodic"))
        if embedding_name == "periodic":
            embedding = embeddings.PeriodicEmbeddings(
                n_inputs,
                d_embedding=int(config.get("embedding_width", 16)),
                n_frequencies=int(config.get("embedding_frequencies", 16)),
                frequency_init_scale=float(config.get("frequency_scale", 0.1)),
                lite=False,
            )
        elif embedding_name == "linear_relu":
            embedding = embeddings.LinearReLUEmbeddings(
                n_inputs, d_embedding=int(config.get("embedding_width", 16))
            )
        elif embedding_name == "none":
            embedding = None
        else:
            raise ValueError(f"unsupported TabM embedding: {embedding_name}")
        model = TabM.make(
            n_num_features=n_inputs,
            d_out=64,
            num_embeddings=embedding,
            n_blocks=int(config["hidden_blocks"]),
            d_block=int(config["hidden_width"]),
            dropout=float(config["dropout"]),
            k=int(config.get("ensemble_members", 32)),
        )
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    return model.to(device)


def fit_model(
    model: nn.Module,
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    high_state: np.ndarray,
    additional_small_weight: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> None:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    weight = torch.as_tensor(sample_weight, dtype=torch.float32, device=device)
    high = torch.as_tensor(high_state.astype("float32"), device=device)
    loss_function = PeakLoss(additional_weight=additional_small_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator, device=device)
        for index in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[index])
            squared = (prediction - y[index, None, :]).square()
            full = squared.mean(dim=(1, 2))
            small = squared[:, :, :11].mean(dim=(1, 2))
            sample_loss = full + additional_small_weight * high[index] * small
            loss = (sample_loss * weight[index]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("training produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


@torch.no_grad()
def predict(model: nn.Module, features: np.ndarray, device: torch.device):
    model.eval()
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    return model(x).mean(dim=1).cpu().numpy()


def run_peakst(data: dict[str, np.ndarray], config: dict, seed: int, device: str):
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_config = config["model"]
    peak_config = config["peak_objective"]
    gate_config = config["gate"]

    transform = Transform.fit(data["source_x"])
    source_x = transform(data["source_x"])
    reference_x = transform(data["reference_x"])
    test_x = transform(data["test_x"])
    source_log = np.log1p(data["source_y"])
    mean = source_log.mean(axis=0)
    scale = np.maximum(source_log.std(axis=0), 1e-6)
    source_target = (source_log - mean) / scale
    source_small = data["source_y"][:, :11].sum(axis=1)
    source_high = grouped_high_state(
        source_small,
        data["source_site"],
        data["source_year"],
        float(peak_config["source_quantile"]),
    )
    weights = site_equal_weights(data["source_site"])
    n_inputs = source_x.shape[1]

    ordinary = make_model(n_inputs, model_config, seed, device_object)
    peak = make_model(n_inputs, model_config, seed, device_object)
    fit_arguments = dict(
        features=source_x,
        target=source_target,
        sample_weight=weights,
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        batch_size=int(model_config["batch_size"]),
        seed=seed,
        device=device_object,
    )
    fit_model(
        ordinary,
        high_state=np.zeros(len(source_x), dtype=bool),
        additional_small_weight=0.0,
        epochs=int(
            model_config.get(
                "ordinary_source_epochs", model_config.get("source_epochs", 40)
            )
        ),
        **fit_arguments,
    )
    fit_model(
        peak,
        high_state=source_high,
        additional_small_weight=float(peak_config["additional_small_cell_loss"]),
        epochs=int(
            model_config.get(
                "peak_source_epochs", model_config.get("source_epochs", 40)
            )
        ),
        **fit_arguments,
    )

    reference_log = np.log1p(data["reference_y"])
    reference_target = (reference_log - mean) / scale
    reference_small = data["reference_y"][:, :11].sum(axis=1)
    reference_high = reference_small >= np.quantile(
        reference_small, float(peak_config["target_quantile"])
    )
    adapt_arguments = dict(
        features=reference_x,
        target=reference_target,
        sample_weight=np.ones(len(reference_x), dtype="float32"),
        epochs=int(model_config["target_epochs"]),
        learning_rate=float(model_config["target_learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        batch_size=min(int(model_config["batch_size"]), 256),
        seed=seed,
        device=device_object,
    )
    fit_model(
        ordinary,
        high_state=np.zeros(len(reference_x), dtype=bool),
        additional_small_weight=0.0,
        **adapt_arguments,
    )
    fit_model(
        peak,
        high_state=reference_high,
        additional_small_weight=float(peak_config["additional_small_cell_loss"]),
        **adapt_arguments,
    )

    gate = HistGradientBoostingClassifier(
        max_leaf_nodes=int(gate_config["max_leaf_nodes"]),
        max_iter=int(gate_config["max_iter"]),
        learning_rate=float(gate_config["learning_rate"]),
        l2_regularization=float(gate_config["l2_regularization"]),
        early_stopping=False,
        random_state=seed,
    )
    gate.fit(
        data["source_x"],
        source_high,
        sample_weight=site_and_class_equal_weights(
            data["source_site"], source_high
        ),
    )
    score = gate.predict_proba(data["test_x"])[:, 1]
    general = np.expm1(np.clip(predict(ordinary, test_x, device_object) * scale + mean, 0, 20))
    specialist = np.expm1(np.clip(predict(peak, test_x, device_object) * scale + mean, 0, 20))
    hard = blend_small_cells(general, specialist, score)
    centers = np.sqrt(data["edges_nm"][:-1] * data["edges_nm"][1:])
    final = boundary_continuous_return(general, hard, centers)
    return {
        "ordinary_adaptation": general,
        "peak_specialist": specialist,
        "source_score": score,
        "PeakST_tapered": final,
    }
