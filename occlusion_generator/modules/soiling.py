from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


class SoilingModule(BaseOcclusionModule):
    """
    Generates a dirt mask without modifying the input image.

    `intensity` controls only the number of dirt patches (degree of contamination from 0 to 1).

    patch textures are sampled from a provided buffer of dirt textures. 
    The textures can be provided as PIL images or torch tensors.

    Expected texture:
        RGB/RGBA PIL Image
        [C, H, W] Tensor
        [B, C, H, W] Tensor

    For RGB textures, alpha is assumed to be 1 everywhere.

    Returns:
        image:
            Original image, unchanged.

        soil_mask:
            [B, 1, H, W] float tensor in [0, 1].
    """

    @staticmethod
    def _texture_to_alpha(
        texture: Any,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(texture, Image.Image):
            image = texture.convert("RGBA")
            array = np.asarray(image)

            tensor = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .contiguous()
                .to(device=device, dtype=torch.float32)
                / 255.0
            )

        elif isinstance(texture, torch.Tensor):
            tensor = texture.to(device=device, dtype=torch.float32)

            if tensor.ndim == 4:
                if tensor.shape[0] == 0:
                    raise ValueError("Empty texture tensor")
                tensor = tensor[0]

            if tensor.ndim != 3:
                raise ValueError(
                    "Texture tensor must have shape [C,H,W] "
                    f"or [B,C,H,W], got {tuple(tensor.shape)}"
                )

            if tensor.numel() > 0 and tensor.max() > 1.0:
                tensor = tensor / 255.0

            tensor = tensor.clamp(0.0, 1.0)

        else:
            raise TypeError(
                "Dirt texture must be a PIL Image or torch.Tensor, "
                f"got {type(texture)!r}"
            )

        channels = tensor.shape[0]

        if channels == 1:
            alpha = tensor[0]
        elif channels == 3:
            alpha = torch.ones(
                tensor.shape[-2:],
                device=device,
                dtype=torch.float32,
            )
        elif channels == 4:
            alpha = tensor[3]
        else:
            raise ValueError(
                f"Texture must have 1, 3 or 4 channels, got {channels}"
            )

        return alpha.clamp(0.0, 1.0)

    def apply(
    self,
    image: torch.Tensor,
    depth: torch.Tensor,
    car_mask: torch.Tensor | None,
    cfg: SoilingConfig,
    dirt_buffer=None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_size, _, height, width = image.shape
    device = image.device
    dtype = image.dtype

    soil_mask = torch.zeros(
        (batch_size, 1, height, width),
        device=device,
        dtype=dtype,
    )

    if dirt_buffer is None or len(dirt_buffer) == 0:
        return image, soil_mask

    intensity = float(cfg.intensity)

    if intensity <= 0:
        return image, soil_mask

    num_patches = max(
        1,
        int(round(cfg.max_patches * intensity)),
    )

    for batch_idx in range(batch_size):
        for _ in range(num_patches):
            texture_idx = torch.randint(
                0,
                len(dirt_buffer),
                (1,),
                device=device,
            ).item()

            alpha = self._texture_to_alpha(
                dirt_buffer[texture_idx],
                device,
            )

            if alpha.numel() == 0:
                continue

            tex_height, tex_width = alpha.shape

            patch_width = min(
                width,
                max(1, int(width * 0.2)),
            )

            patch_height = min(
                height,
                max(
                    1,
                    int(patch_width * tex_height / tex_width),
                ),
            )

            alpha = F.interpolate(
                alpha[None, None],
                size=(patch_height, patch_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            x = torch.randint(
                0,
                max(1, width - patch_width + 1),
                (1,),
                device=device,
            ).item()

            y = torch.randint(
                0,
                max(1, height - patch_height + 1),
                (1,),
                device=device,
            ).item()

            soil_mask[
                batch_idx,
                0,
                y : y + patch_height,
                x : x + patch_width,
            ] = torch.maximum(
                soil_mask[
                    batch_idx,
                    0,
                    y : y + patch_height,
                    x : x + patch_width,
                ],
                alpha.to(dtype),
            )
    print(
    "soil_mask:",
    soil_mask.min().item(),
    soil_mask.max().item(),
    soil_mask.mean().item(),
    "nonzero:",
    (soil_mask > 0).sum().item(),
)
    return image, soil_mask