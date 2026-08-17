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

    The module intentionally performs no texture processing:
        - no resize
        - no rotation
        - no blur
        - no color processing
        - no opacity modification
        - no severity
        - no additional randomness inside the texture

    `intensity` controls only the number of dirt patches.

    Texture alpha is used directly as the dirt mask.

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

    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor | None,
        cfg: SoilingConfig,
        dirt_buffer=None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if image.ndim != 4 or image.shape[1] < 3:
            raise ValueError(
                "SoilingModule expects image with shape [B, C, H, W], "
                f"got {tuple(image.shape)}"
            )

        if not cfg.enabled or float(cfg.intensity) <= 0.0:
            return image, torch.zeros_like(image[:, :1])

        if dirt_buffer is None:
            dirt_buffer = kwargs.get("dirt_textures")

        if not dirt_buffer:
            raise RuntimeError(
                "SoilingModule requires a non-empty dirt_buffer."
            )

        batch_size, _, image_height, image_width = image.shape
        device = image.device
        dtype = image.dtype

        intensity = max(
            0.0,
            min(1.0, float(cfg.intensity)),
        )

        # Number of independently placed dirt textures.
        num_patches = max(
            1,
            round(1 + 9 * intensity),
        )

        soil_mask = torch.zeros(
            batch_size,
            1,
            image_height,
            image_width,
            device=device,
            dtype=dtype,
        )

        for _ in range(num_patches):

            # ----------------------------------------------------------
            # Sample texture
            # ----------------------------------------------------------

            texture = dirt_buffer[
                torch.randint(
                    len(dirt_buffer),
                    (),
                    device=device,
                ).item()
            ]

            if isinstance(texture, Image.Image):

                texture = np.asarray(
                    texture.convert("RGBA")
                )

                texture = (
                    torch.from_numpy(texture)
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )

            elif torch.is_tensor(texture):

                if texture.ndim == 4:
                    texture = texture[0]

                if texture.ndim != 3:
                    raise ValueError(
                        "Dirt texture must have shape [C,H,W] "
                        "or [B,C,H,W], "
                        f"got {tuple(texture.shape)}"
                    )

                texture = texture.float()

                if texture.max() > 1.0:
                    texture = texture / 255.0

            else:
                raise TypeError(
                    "Unsupported dirt texture type: "
                    f"{type(texture)}"
                )

            texture = texture.to(
                device=device,
                dtype=dtype,
            ).clamp(0.0, 1.0)

            if texture.shape[0] == 3:

                alpha = torch.ones(
                    1,
                    texture.shape[1],
                    texture.shape[2],
                    device=device,
                    dtype=dtype,
                )

                texture = torch.cat(
                    [texture, alpha],
                    dim=0,
                )

            elif texture.shape[0] != 4:

                raise ValueError(
                    "Dirt texture must contain 3 or 4 channels, "
                    f"got {texture.shape[0]}"
                )

            # ----------------------------------------------------------
            # Take alpha exactly as supplied by the texture.
            # ----------------------------------------------------------

            patch = texture[3:4]

            patch_height, patch_width = patch.shape[-2:]

            # Ignore textures that cannot fit at all.
            if (
                patch_height > image_height
                or patch_width > image_width
            ):
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(
                        min(patch_height, image_height),
                        min(patch_width, image_width),
                    ),
                    mode="nearest",
                )[0]

                patch_height, patch_width = patch.shape[-2:]

            # ----------------------------------------------------------
            # Random placement.
            # ----------------------------------------------------------

            max_x = image_width - patch_width
            max_y = image_height - patch_height

            x = (
                torch.randint(
                    max_x + 1,
                    (),
                    device=device,
                ).item()
                if max_x > 0
                else 0
            )

            y = (
                torch.randint(
                    max_y + 1,
                    (),
                    device=device,
                ).item()
                if max_y > 0
                else 0
            )

            # ----------------------------------------------------------
            # Composite dirt masks.
            #
            # Overlapping dirt accumulates:
            #
            #   A + B * (1 - A)
            #
            # so several overlapping patches can reach opacity 1.
            # ----------------------------------------------------------

            existing = soil_mask[
                :,
                :,
                y:y + patch_height,
                x:x + patch_width,
            ]

            combined = (
                existing
                + patch.unsqueeze(0)
                * (1.0 - existing)
            )

            soil_mask[
                :,
                :,
                y:y + patch_height,
                x:x + patch_width,
            ] = combined.clamp(0.0, 1.0)

        # --------------------------------------------------------------
        # Restrict dirt to the valid car region.
        # --------------------------------------------------------------

        if car_mask is not None:

            if car_mask.ndim == 3:
                car_mask = car_mask.unsqueeze(1)

            if car_mask.ndim != 4:
                raise ValueError(
                    "car_mask must have shape [B,H,W] or [B,1,H,W], "
                    f"got {tuple(car_mask.shape)}"
                )

            car_mask = car_mask[:, :1]

            if car_mask.shape[-2:] != (
                image_height,
                image_width,
            ):
                car_mask = F.interpolate(
                    car_mask.float(),
                    size=(
                        image_height,
                        image_width,
                    ),
                    mode="nearest",
                )

            car_mask = car_mask.to(
                device=device,
                dtype=dtype,
            ).clamp(0.0, 1.0)

            soil_mask *= car_mask

        # The module is a pure mask generator.
        return image, soil_maskм