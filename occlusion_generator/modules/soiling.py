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
        if image.ndim != 4:
            raise ValueError(
                f"`image` must have shape [B,C,H,W], got {tuple(image.shape)}"
            )

        batch_size, _, height, width = image.shape
        device = image.device
        dtype = image.dtype

        soil_mask = torch.zeros(
            (batch_size, 1, height, width),
            device=device,
            dtype=dtype,
        )

        if not dirt_buffer:
            return image, soil_mask

        intensity = float(cfg.intensity)
        intensity = max(0.0, min(1.0, intensity))

        if intensity <= 0.0:
            return image, soil_mask

        max_patches = int(getattr(cfg, "max_patches", 32))
        num_patches = int(round(intensity * max_patches))

        if num_patches <= 0:
            return image, soil_mask

        min_scale = float(getattr(cfg, "min_scale", 0.05))
        max_scale = float(getattr(cfg, "max_scale", 0.25))

        generator = kwargs.get("generator")

        if car_mask is not None:
            if car_mask.ndim == 3:
                car_mask = car_mask.unsqueeze(1)

            if car_mask.shape[-2:] != (height, width):
                car_mask = F.interpolate(
                    car_mask.float(),
                    size=(height, width),
                    mode="nearest",
                )

            car_mask = car_mask.to(
                device=device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)

        for batch_idx in range(batch_size):
            for _ in range(num_patches):
                texture_idx = torch.randint(
                    len(dirt_buffer),
                    (1,),
                    device=device,
                    generator=generator,
                ).item()

                alpha = self._texture_to_alpha(
                    dirt_buffer[texture_idx],
                    device,
                )

                tex_height, tex_width = alpha.shape

                scale = torch.empty(
                    1,
                    device=device,
                ).uniform_(
                    min_scale,
                    max_scale,
                    generator=generator,
                ).item()

                patch_width = max(1, int(width * scale))
                patch_height = max(
                    1,
                    int(patch_width * tex_height / tex_width),
                )

                patch_width = min(patch_width, width)
                patch_height = min(patch_height, height)

                alpha = F.interpolate(
                    alpha[None, None],
                    size=(patch_height, patch_width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

                max_x = width - patch_width
                max_y = height - patch_height

                x = torch.randint(
                    max_x + 1,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()

                y = torch.randint(
                    max_y + 1,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()

                patch = alpha

                if car_mask is not None:
                    patch = patch * car_mask[
                        batch_idx,
                        0,
                        y : y + patch_height,
                        x : x + patch_width,
                    ]

                current = soil_mask[
                    batch_idx,
                    0,
                    y : y + patch_height,
                    x : x + patch_width,
                ]

                soil_mask[
                    batch_idx,
                    0,
                    y : y + patch_height,
                    x : x + patch_width,
                ] = 1.0 - (1.0 - current) * (1.0 - patch)

        return image, soil_mask.clamp(0.0, 1.0)