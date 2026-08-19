from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


class SoilingModule(BaseOcclusionModule):
    """
    Physical lens soiling with patch placement.

    IMPORTANT:
        Dirt patches are NOT resized to the full image.

    Expected:
        image:    [B, C, H, W]
        depth:    [B, 1, H, W] / [B, H, W]
        car_mask: [B, 1, H, W] / [B, H, W]

        dirt_buffer:
            [1, C, H_patch, W_patch]
            [B, C, H_patch, W_patch]

    Returns:
        soiled_image: [B, C, H, W]
        final_alpha:  [B, 1, H, W]
    """

    @torch.no_grad()
    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor,
        cfg: SoilingConfig,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. EARLY EXIT
        if not cfg.enabled or cfg.intensity == 0.0:
            return (
                image,
                torch.zeros_like(
                    image[:, 0:1, :, :]
                ),
            )
        # 2. INPUT
        if image.ndim != 4:
            raise ValueError(
                "[SOILING] Expected image [B,C,H,W], "
                f"got {image.shape}"
            )

        b, c, h, w = image.shape

        device = image.device

        # 3. NORMALIZE CAR MASK

        if car_mask.ndim == 2:
            car_mask = (
                car_mask
                .unsqueeze(0)
                .unsqueeze(0)
            )

        elif car_mask.ndim == 3:
            car_mask = car_mask.unsqueeze(1)

        elif car_mask.ndim != 4:
            raise ValueError(
                "[SOILING] Unexpected car_mask shape: "
                f"{car_mask.shape}"
            )

        car_mask = car_mask.to(
            device=device,
            dtype=image.dtype,
        )

        if car_mask.shape[0] == 1 and b > 1:
            car_mask = car_mask.expand(
                b,
                -1,
                -1,
                -1,
            )

        if car_mask.shape[1] > 1:
            car_mask = car_mask.mean(
                dim=1,
                keepdim=True,
            )

        if car_mask.shape[-2:] != (h, w):
            car_mask = F.interpolate(
                car_mask,
                size=(h, w),
                mode="nearest",
            )

        car_mask = torch.clamp(
            car_mask,
            0.0,
            1.0,
        )

        # =========================================================
        # 4. GET DIRT PATCH
        # =========================================================

        soil_texture = kwargs.get(
            "dirt_buffer",
            None,
        )

        # =========================================================
        # 5. FALLBACK
        # =========================================================

        if soil_texture is None:
            patch_size = min(
                256,
                h,
                w,
            )

            soil_texture = (
                torch.rand(
                    (
                        b,
                        1,
                        patch_size,
                        patch_size,
                    ),
                    device=device,
                    dtype=image.dtype,
                )
                * 0.6
                + 0.2
            )

        # =========================================================
        # 6. TO DEVICE / DTYPE
        # =========================================================

        if not torch.is_tensor(soil_texture):
            raise TypeError(
                "[SOILING] dirt_buffer must be "
                f"a torch.Tensor, got {type(soil_texture)}"
            )

        soil_texture = soil_texture.to(
            device=device,
            dtype=image.dtype,
        )

        # =========================================================
        # 7. NORMALIZE DIMENSIONS
        # =========================================================

        if soil_texture.ndim == 2:
            soil_texture = (
                soil_texture
                .unsqueeze(0)
                .unsqueeze(0)
            )

        elif soil_texture.ndim == 3:
            soil_texture = (
                soil_texture
                .unsqueeze(0)
            )

        elif soil_texture.ndim != 4:
            raise ValueError(
                "[SOILING] Unsupported soil_texture shape: "
                f"{soil_texture.shape}"
            )

        # =========================================================
        # 8. BATCH
        # =========================================================

        patch_batch = soil_texture.shape[0]

        if patch_batch == 1 and b > 1:
            soil_texture = soil_texture.expand(
                b,
                -1,
                -1,
                -1,
            )

        elif patch_batch != b:
            raise ValueError(
                "[SOILING] Batch mismatch:\n"
                f"image batch = {b}\n"
                f"soil batch  = {patch_batch}"
            )

        # =========================================================
        # 9. NO RESIZE !!!
        # =========================================================

        patch_h = soil_texture.shape[2]
        patch_w = soil_texture.shape[3]

        if patch_h > h or patch_w > w:
            raise ValueError(
                "[SOILING] Dirt patch is larger "
                "than image.\n"
                f"patch = {patch_w}x{patch_h}\n"
                f"image = {w}x{h}\n"
                "Resize the patch before passing it "
                "to the pipeline."
            )

        # =========================================================
        # 10. CONVERT PATCH TO RGB + ALPHA
        # =========================================================

        texture_channels = soil_texture.shape[1]

        if texture_channels == 1:
            soil_rgb_patch = soil_texture.expand(
                -1,
                c,
                -1,
                -1,
            )

            soil_alpha_patch = soil_texture

        elif texture_channels == 3:
            soil_rgb_patch = soil_texture

            soil_alpha_patch = (
                soil_texture.mean(
                    dim=1,
                    keepdim=True,
                )
            )

        else:
            raise ValueError(
                "[SOILING] soil_texture must have "
                f"1 or 3 channels, got "
                f"{texture_channels}"
            )

        # =========================================================
        # 11. CREATE FULL-SIZE CANVAS
        # =========================================================

        soil_rgb = torch.zeros(
            (
                b,
                c,
                h,
                w,
            ),
            device=device,
            dtype=image.dtype,
        )

        soil_alpha = torch.zeros(
            (
                b,
                1,
                h,
                w,
            ),
            device=device,
            dtype=image.dtype,
        )

        # =========================================================
        # 12. PLACE PATCH
        # =========================================================
        #
        # For now:
        # top-left corner
        #
        # NO RESIZE
        # NO ROTATION
        # NO RANDOM POSITION
        #

        x = 0
        y = 0

        soil_rgb[
            :,
            :,
            y:y + patch_h,
            x:x + patch_w,
        ] = soil_rgb_patch

        soil_alpha[
            :,
            :,
            y:y + patch_h,
            x:x + patch_w,
        ] = soil_alpha_patch

        # =========================================================
        # 13. CAR MASK
        # =========================================================

        # IMPORTANT:
        #
        # Your current pipeline creates:
        #
        # car_mask = zeros(...)
        #
        # therefore multiplying by it would completely
        # remove the dirt.
        #
        # For this patch-placement test we DO NOT apply
        # car_mask if it is completely empty.

        car_mask_is_empty = (
            car_mask.max().item() <= 1e-6
        )

        if car_mask_is_empty:
            final_alpha = soil_alpha

        else:
            final_alpha = (
                soil_alpha * car_mask
            )

        # =========================================================
        # 14. INTENSITY
        # =========================================================

        final_alpha = (
            final_alpha
            * float(cfg.intensity)
        )

        final_alpha = torch.clamp(
            final_alpha,
            0.0,
            1.0,
        )

        # =========================================================
        # 15. BLENDING
        # =========================================================

        final_alpha_3c = final_alpha.expand(
            -1,
            c,
            -1,
            -1,
        )

        soiled_image = (
            image * (1.0 - final_alpha_3c)
            + soil_rgb * final_alpha_3c
        )

        # =========================================================
        # 16. OUTPUT
        # =========================================================

        soiled_image = torch.clamp(
            soiled_image,
            0.0,
            1.0,
        )

        return (
            soiled_image,
            final_alpha,
        )