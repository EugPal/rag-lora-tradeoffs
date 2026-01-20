from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    rank: int
    target_modules: list[str]
    target_layers: str
    data_fraction: float


PRESETS = {
    "L4-S": LoRAConfig(rank=4, target_modules=["attention"], target_layers="top", data_fraction=0.25),
    "L4-F": LoRAConfig(rank=4, target_modules=["attention"], target_layers="top", data_fraction=1.0),
    "L8-S": LoRAConfig(rank=8, target_modules=["attention"], target_layers="all", data_fraction=0.25),
    "L8-F": LoRAConfig(rank=8, target_modules=["attention"], target_layers="all", data_fraction=1.0),
    "L16-S": LoRAConfig(rank=16, target_modules=["attention", "ffn"], target_layers="top", data_fraction=0.25),
    "L16-F": LoRAConfig(rank=16, target_modules=["attention", "ffn"], target_layers="all", data_fraction=1.0),
}


def get_preset(preset_id: str) -> LoRAConfig:
    if preset_id not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_id}")
    return PRESETS[preset_id]
