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
    def _texture_to_rgb(
        texture: Any,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(texture, Image.Image):
            image = texture.convert("RGB")
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
            tensor = tensor.repeat(3, 1, 1)

        elif channels == 4:
            tensor = tensor[:3]

        elif channels != 3:
            raise ValueError(
                f"Texture must have 1, 3 or 4 channels, got {channels}"
            )

        return tensor

    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor | None,
        cfg: SoilingConfig,
        dirt_buffer=None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size, channels, height, width = image.shape

        device = image.device
        dtype = image.dtype

        result = image.clone()

        soil_mask = torch.zeros(
            (batch_size, 1, height, width),
            device=device,
            dtype=dtype,
        )

        if dirt_buffer is None or len(dirt_buffer) == 0:
            return result, soil_mask

        intensity = float(cfg.intensity)
        intensity = max(0.0, min(1.0, intensity))

        if intensity <= 0.0:
            return result, soil_mask

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

                texture = self._texture_to_rgb(
                    dirt_buffer[texture_idx],
                    device,
                )

                tex_height, tex_width = texture.shape[-2:]

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

                texture = F.interpolate(
                    texture[None],
                    size=(patch_height, patch_width),
                    mode="bilinear",
                    align_corners=False,
                )[0]

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

                # Полностью копируем dirt texture на image.
                result[
                    batch_idx,
                    :3,
                    y:y + patch_height,
                    x:x + patch_width,
                ] = texture.to(dtype)

                # GT = 1 везде, где texture была наложена.
                soil_mask[
                    batch_idx,
                    0,
                    y:y + patch_height,
                    x:x + patch_width,
                ] = 1.0

        return result.clamp(0.0, 1.0), soil_mask