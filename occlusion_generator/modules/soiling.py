from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


class SoilingModule(BaseOcclusionModule):

    @torch.no_grad()
    def apply(
        self,
        image: torch.Tensor,
        depth,
        car_mask,
        cfg: SoilingConfig,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        DEBUG = True

        # =========================================================
        # 1. INPUT
        # =========================================================

        if not cfg.enabled or cfg.intensity <= 0:
            return (
                image,
                torch.zeros_like(image[:, 0:1]),
            )

        if image.ndim != 4:
            raise ValueError(
                f"Expected image [B,C,H,W], got {image.shape}"
            )

        b, c, h, w = image.shape
        device = image.device
        dtype = image.dtype

        print("\n" + "=" * 70)
        print("[SOILING] SIMPLE PATCH MODE")
        print("=" * 70)

        print(
            "[SOILING] image:",
            tuple(image.shape),
        )

        # =========================================================
        # 2. GET PATCH
        # =========================================================

        soil_texture = kwargs.get(
            "dirt_buffer",
            None,
        )

        if soil_texture is None:
            raise ValueError(
                "[SOILING] dirt_buffer was not provided"
            )

        if not torch.is_tensor(soil_texture):
            raise TypeError(
                "[SOILING] dirt_buffer must be torch.Tensor"
            )

        soil_texture = soil_texture.to(
            device=device,
            dtype=dtype,
        )

        # =========================================================
        # 3. NORMALIZE PATCH DIMENSIONS
        # =========================================================

        if soil_texture.ndim == 2:

            # [H,W]
            soil_texture = (
                soil_texture
                .unsqueeze(0)
                .unsqueeze(0)
            )

        elif soil_texture.ndim == 3:

            # [C,H,W]
            soil_texture = (
                soil_texture
                .unsqueeze(0)
            )

        elif soil_texture.ndim != 4:

            raise ValueError(
                "[SOILING] Invalid dirt_buffer shape: "
                f"{soil_texture.shape}"
            )

        print(
            "[SOILING] patch:",
            tuple(soil_texture.shape),
        )

        # =========================================================
        # 4. BATCH
        # =========================================================

        if soil_texture.shape[0] == 1 and b > 1:

            soil_texture = soil_texture.expand(
                b,
                -1,
                -1,
                -1,
            )

        elif soil_texture.shape[0] != b:

            raise ValueError(
                "[SOILING] Batch mismatch: "
                f"image={b}, patch={soil_texture.shape[0]}"
            )

        # =========================================================
        # 5. CHANNELS
        # =========================================================

        channels = soil_texture.shape[1]

        if channels == 1:

            patch_rgb = soil_texture.expand(
                -1,
                c,
                -1,
                -1,
            )

        elif channels == 3:

            patch_rgb = soil_texture

        else:

            raise ValueError(
                "[SOILING] dirt_buffer must have "
                f"1 or 3 channels, got {channels}"
            )

        # =========================================================
        # 6. IMPORTANT:
        #    NO RESIZE
        # =========================================================

        patch_h = patch_rgb.shape[2]
        patch_w = patch_rgb.shape[3]

        print(
            "[SOILING] patch size:",
            patch_w,
            "x",
            patch_h,
        )

        print(
            "[SOILING] image size:",
            w,
            "x",
            h,
        )

        if patch_h > h or patch_w > w:

            raise ValueError(
                "[SOILING] Patch is larger than image: "
                f"patch={patch_w}x{patch_h}, "
                f"image={w}x{h}"
            )

        # =========================================================
        # 7. PATCH POSITION
        #
        # For now: TOP LEFT.
        #
        # No random position.
        # No resize.
        # No distortion.
        # =========================================================

        x = 0
        y = 0

        print(
            "[SOILING] position:",
            f"x={x}, y={y}",
        )

        # =========================================================
        # 8. CREATE CANVAS
        # =========================================================

        overlay = torch.zeros(
            (
                b,
                c,
                h,
                w,
            ),
            device=device,
            dtype=dtype,
        )

        alpha = torch.zeros(
            (
                b,
                1,
                h,
                w,
            ),
            device=device,
            dtype=dtype,
        )

        # =========================================================
        # 9. PLACE PATCH
        # =========================================================

        overlay[
            :,
            :,
            y:y + patch_h,
            x:x + patch_w,
        ] = patch_rgb

        # =========================================================
        # 10. CONSTANT ALPHA
        #
        # Do NOT derive alpha from RGB.
        #
        # This is deliberately simple.
        # =========================================================

        patch_alpha = float(
            cfg.intensity
        )

        alpha[
            :,
            :,
            y:y + patch_h,
            x:x + patch_w,
        ] = patch_alpha

        alpha = torch.clamp(
            alpha,
            0.0,
            1.0,
        )

        # =========================================================
        # 11. BLEND
        # =========================================================

        alpha_rgb = alpha.expand(
            -1,
            c,
            -1,
            -1,
        )

        result = (
            image * (1.0 - alpha_rgb)
            + overlay * alpha_rgb
        )

        result = torch.clamp(
            result,
            0.0,
            1.0,
        )

        # =========================================================
        # 12. DEBUG
        # =========================================================

        diff = (
            result - image
        ).abs()

        print(
            "[SOILING] alpha:",
            alpha.min().item(),
            alpha.max().item(),
        )

        print(
            "[SOILING] changed pixels:",
            (
                diff > 1e-4
            ).float().mean().item() * 100,
            "%",
        )

        print(
            "[SOILING] difference:",
            diff.mean().item(),
        )

        print("=" * 70)
        print("[SOILING] END")
        print("=" * 70)

        return result, alpha