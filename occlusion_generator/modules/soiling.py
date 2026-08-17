from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


class SoilingModule(BaseOcclusionModule):
    _MAX_PATCHES = 32

    @staticmethod
    def _texture_to_alpha(
        texture: Any,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(texture, Image.Image):
            image = texture.convert("RGBA")
            array = np.array(image, copy=True)

            tensor = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .contiguous()
                .to(device=device, dtype=torch.float32)
                / 255.0
            )

        elif isinstance(texture, torch.Tensor):
            tensor = texture.to(
                device=device,
                dtype=torch.float32,
            )

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
            return tensor[0]

        if channels == 3:
            return torch.ones(
                tensor.shape[-2:],
                device=device,
                dtype=torch.float32,
            )

        if channels == 4:
            return tensor[3]

        raise ValueError(
            f"Texture must have 1, 3 or 4 channels, got {channels}"
        )

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
        intensity = max(0.0, min(1.0, intensity))

        if intensity <= 0.0:
            return image, soil_mask

        num_patches = max(
            1,
            int(round(self._MAX_PATCHES * intensity)),
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

                tex_height, tex_width = alpha.shape

                patch_width = min(
                    width,
                    max(1, int(width * 0.2)),
                )

                patch_height = min(
                    height,
                    max(
                        1,
                        int(
                            patch_width
                            * tex_height
                            / tex_width
                        ),
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

                current = soil_mask[
                    batch_idx,
                    0,
                    y:y + patch_height,
                    x:x + patch_width,
                ]

                soil_mask[
                    batch_idx,
                    0,
                    y:y + patch_height,
                    x:x + patch_width,
                ] = torch.maximum(
                    current,
                    alpha.to(dtype),
                )

        return image, soil_mask.clamp(0.0, 1.0)