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
    def _texture_to_rgba(
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
            rgb = tensor.repeat(3, 1, 1)
            alpha = torch.ones_like(tensor)

        elif channels == 3:
            rgb = tensor
            alpha = torch.ones(
                (1, tensor.shape[1], tensor.shape[2]),
                device=device,
                dtype=torch.float32,
            )

        elif channels == 4:
            rgb = tensor[:3]
            alpha = tensor[3:4]

        else:
            raise ValueError(
                f"Texture must have 1, 3 or 4 channels, got {channels}"
            )

        return torch.cat((rgb, alpha), dim=0)

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

        batch_size, channels, height, width = image.shape

        if channels not in (1, 3, 4):
            raise ValueError(
                f"`image` must have 1, 3 or 4 channels, got {channels}"
            )

        device = image.device
        dtype = image.dtype

        dirty_image = image.clone()

        soil_mask = torch.zeros(
            (batch_size, 1, height, width),
            device=device,
            dtype=dtype,
        )

        if dirt_buffer is None or len(dirt_buffer) == 0:
            return dirty_image, soil_mask

        intensity = float(cfg.intensity)
        intensity = max(0.0, min(1.0, intensity))

        if intensity <= 0.0:
            return dirty_image, soil_mask

        num_patches = max(
            1,
            int(round(self._MAX_PATCHES * intensity)),
        )

        if car_mask is not None:
            if car_mask.ndim == 3:
                car_mask = car_mask.unsqueeze(1)

            if car_mask.ndim != 4:
                raise ValueError(
                    "`car_mask` must have shape [B,1,H,W] "
                    f"or [B,H,W], got {tuple(car_mask.shape)}"
                )

            if car_mask.shape[0] != batch_size:
                raise ValueError(
                    "Batch size mismatch between image and car_mask: "
                    f"{batch_size} vs {car_mask.shape[0]}"
                )

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

        generator = kwargs.get("generator")

        for batch_idx in range(batch_size):
            for _ in range(num_patches):
                texture_idx = torch.randint(
                    0,
                    len(dirt_buffer),
                    (1,),
                    device=device,
                    generator=generator,
                ).item()

                texture = self._texture_to_rgba(
                    dirt_buffer[texture_idx],
                    device,
                )

                texture_rgb = texture[:3]
                texture_alpha = texture[3]

                tex_height, tex_width = texture_alpha.shape

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

                texture_rgb = F.interpolate(
                    texture_rgb[None],
                    size=(patch_height, patch_width),
                    mode="bilinear",
                    align_corners=False,
                )[0]

                texture_alpha = F.interpolate(
                    texture_alpha[None, None],
                    size=(patch_height, patch_width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

                max_x = width - patch_width
                max_y = height - patch_height

                x = torch.randint(
                    0,
                    max_x + 1,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()

                y = torch.randint(
                    0,
                    max_y + 1,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()

                patch_alpha = texture_alpha

                if car_mask is not None:
                    patch_alpha = patch_alpha * car_mask[
                        batch_idx,
                        0,
                        y:y + patch_height,
                        x:x + patch_width,
                    ]

                patch_alpha = patch_alpha.clamp(0.0, 1.0)

                if not torch.any(patch_alpha > 0):
                    continue

                current_mask = soil_mask[
                    batch_idx,
                    0,
                    y:y + patch_height,
                    x:x + patch_width,
                ]

                new_mask = torch.maximum(
                    current_mask,
                    patch_alpha.to(dtype),
                )

                soil_mask[
                    batch_idx,
                    0,
                    y:y + patch_height,
                    x:x + patch_width,
                ] = new_mask

                if channels == 1:
                    patch_rgb = (
                        texture_rgb.mean(dim=0, keepdim=True)
                    )
                else:
                    patch_rgb = texture_rgb

                    if channels == 4:
                        patch_rgb = torch.cat(
                            (
                                patch_rgb,
                                torch.ones(
                                    (1, patch_height, patch_width),
                                    device=device,
                                    dtype=patch_rgb.dtype,
                                ),
                            ),
                            dim=0,
                        )

                image_region = dirty_image[
                    batch_idx,
                    :patch_rgb.shape[0],
                    y:y + patch_height,
                    x:x + patch_width,
                ]

                alpha = patch_alpha.to(dtype)

                if patch_rgb.shape[0] == 1:
                    blended = (
                        image_region * (1.0 - alpha)
                        + patch_rgb.to(dtype) * alpha
                    )
                else:
                    blended = (
                        image_region * (1.0 - alpha.unsqueeze(0))
                        + patch_rgb.to(dtype)
                        * alpha.unsqueeze(0)
                    )

                dirty_image[
                    batch_idx,
                    :patch_rgb.shape[0],
                    y:y + patch_height,
                    x:x + patch_width,
                ] = blended

        return dirty_image.clamp(0.0, 1.0), soil_mask.clamp(0.0, 1.0)