from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


class SoilingModule(BaseOcclusionModule):
    """
    Physical lens soiling with extensive debugging.

    Expected:
        image:    [B, C, H, W]
        depth:    [B, 1, H, W] / [B, H, W]
        car_mask: [B, 1, H, W] / [B, H, W]

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

        DEBUG = True

        def dbg(name: str, x: Any):
            if not DEBUG:
                return

            if x is None:
                print(f"[SOILING DEBUG] {name}: None")
                return

            if not torch.is_tensor(x):
                print(f"[SOILING DEBUG] {name}: {x}")
                return

            print(
                f"[SOILING DEBUG] {name}: "
                f"shape={tuple(x.shape)}, "
                f"dtype={x.dtype}, "
                f"device={x.device}, "
                f"min={x.min().item():.6f}, "
                f"max={x.max().item():.6f}, "
                f"mean={x.mean().item():.6f}"
            )

        print("\n" + "=" * 70)
        print("[SOILING DEBUG] APPLY START")
        print("=" * 70)

        # ---------------------------------------------------------
        # 0. Config
        # ---------------------------------------------------------

        print(f"[SOILING DEBUG] cfg.enabled   = {cfg.enabled}")
        print(f"[SOILING DEBUG] cfg.intensity = {cfg.intensity}")

        # ---------------------------------------------------------
        # 1. Early exit
        # ---------------------------------------------------------

        if not cfg.enabled or cfg.intensity == 0.0:
            print(
                "[SOILING DEBUG] !!! EARLY EXIT !!! "
                "Soiling is disabled or intensity == 0"
            )

            return image, torch.zeros_like(image[:, 0:1, :, :])

        # ---------------------------------------------------------
        # 2. Input
        # ---------------------------------------------------------

        dbg("image INPUT", image)
        dbg("depth INPUT", depth)
        dbg("car_mask INPUT", car_mask)

        if image.ndim != 4:
            raise ValueError(
                f"[SOILING] Expected image [B,C,H,W], got {image.shape}"
            )

        b, c, h, w = image.shape
        device = image.device

        # ---------------------------------------------------------
        # 3. Normalize car_mask
        # ---------------------------------------------------------

        if car_mask.ndim == 2:
            # [H,W]
            car_mask = car_mask.unsqueeze(0).unsqueeze(0)

        elif car_mask.ndim == 3:
            # Could be [B,H,W] or [1,H,W]
            car_mask = car_mask.unsqueeze(1)

        elif car_mask.ndim != 4:
            raise ValueError(
                f"[SOILING] Unexpected car_mask shape: {car_mask.shape}"
            )

        car_mask = car_mask.to(device=device, dtype=image.dtype)

        if car_mask.shape[0] == 1 and b > 1:
            car_mask = car_mask.expand(b, -1, -1, -1)

        if car_mask.shape[1] > 1:
            car_mask = car_mask.mean(dim=1, keepdim=True)

        if car_mask.shape[-2:] != (h, w):
            print(
                "[SOILING DEBUG] Resizing car_mask:",
                tuple(car_mask.shape[-2:]),
                "->",
                (h, w),
            )

            car_mask = F.interpolate(
                car_mask,
                size=(h, w),
                mode="nearest",
            )

        car_mask = torch.clamp(car_mask, 0.0, 1.0)

        dbg("car_mask NORMALIZED", car_mask)

        # ---------------------------------------------------------
        # 4. Get soil texture
        # ---------------------------------------------------------

        soil_texture = kwargs.get("dirt_buffer", None)

        print(
            "[SOILING DEBUG] kwargs keys =",
            list(kwargs.keys()),
        )

        dbg("soil_texture RAW", soil_texture)

        # ---------------------------------------------------------
        # 5. Fallback texture
        # ---------------------------------------------------------

        if soil_texture is None:
            print(
                "[SOILING DEBUG] !!! soil_texture is None !!!"
            )
            print(
                "[SOILING DEBUG] Generating fallback texture"
            )

            soil_texture = (
                torch.rand(
                    (b, 1, h, w),
                    device=device,
                    dtype=image.dtype,
                )
                * 0.6
                + 0.2
            )

        # ---------------------------------------------------------
        # 6. Convert texture to tensor/device/dtype
        # ---------------------------------------------------------

        soil_texture = soil_texture.to(
            device=device,
            dtype=image.dtype,
        )

        # ---------------------------------------------------------
        # 7. Normalize texture dimensions
        # ---------------------------------------------------------

        if soil_texture.ndim == 2:
            # [H,W]
            soil_texture = soil_texture.unsqueeze(0).unsqueeze(0)

        elif soil_texture.ndim == 3:
            # Assume [C,H,W] OR [B,H,W]
            #
            # For this pipeline we interpret it as [C,H,W]
            # and add batch dimension.
            soil_texture = soil_texture.unsqueeze(0)

        elif soil_texture.ndim != 4:
            raise ValueError(
                "[SOILING] Unsupported soil_texture shape: "
                f"{soil_texture.shape}"
            )

        dbg("soil_texture AFTER DIM NORMALIZATION", soil_texture)

        # ---------------------------------------------------------
        # 8. Batch normalization
        # ---------------------------------------------------------

        if soil_texture.shape[0] == 1 and b > 1:
            soil_texture = soil_texture.expand(
                b,
                -1,
                -1,
                -1,
            )

        elif soil_texture.shape[0] != b:
            raise ValueError(
                "[SOILING] Batch mismatch:\n"
                f"image batch = {b}\n"
                f"soil batch  = {soil_texture.shape[0]}"
            )

        # ---------------------------------------------------------
        # 9. Resize texture
        # ---------------------------------------------------------

        if soil_texture.shape[-2:] != (h, w):

            print(
                "[SOILING DEBUG] Resizing soil_texture:",
                tuple(soil_texture.shape[-2:]),
                "->",
                (h, w),
            )

            soil_texture = F.interpolate(
                soil_texture,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        dbg(
            "soil_texture AFTER RESIZE",
            soil_texture,
        )

        # ---------------------------------------------------------
        # 10. Convert texture to RGB + alpha
        # ---------------------------------------------------------

        texture_channels = soil_texture.shape[1]

        print(
            "[SOILING DEBUG] texture channels =",
            texture_channels,
        )

        if texture_channels == 1:

            soil_alpha = soil_texture

            soil_rgb = soil_texture.expand(
                -1,
                c,
                -1,
                -1,
            )

        elif texture_channels == 3:

            soil_rgb = soil_texture

            # RGB -> grayscale alpha
            soil_alpha = soil_texture.mean(
                dim=1,
                keepdim=True,
            )

        else:

            raise ValueError(
                "[SOILING] soil_texture must have "
                f"1 or 3 channels, got {texture_channels}"
            )

        dbg("soil_rgb", soil_rgb)
        dbg("soil_alpha", soil_alpha)

        # ---------------------------------------------------------
        # 11. VERY IMPORTANT DEBUG
        # ---------------------------------------------------------

        print(
            "[SOILING DEBUG] "
            f"car_mask mean      = {car_mask.mean().item():.6f}"
        )

        print(
            "[SOILING DEBUG] "
            f"soil_alpha mean    = {soil_alpha.mean().item():.6f}"
        )

        print(
            "[SOILING DEBUG] "
            f"cfg.intensity      = {cfg.intensity}"
        )

        # ---------------------------------------------------------
        # 12. Apply car mask
        # ---------------------------------------------------------

        final_alpha = soil_alpha * car_mask

        dbg(
            "alpha AFTER car_mask",
            final_alpha,
        )

        # ---------------------------------------------------------
        # 13. Apply intensity
        # ---------------------------------------------------------

        final_alpha = final_alpha * float(
            cfg.intensity
        )

        dbg(
            "alpha AFTER intensity",
            final_alpha,
        )

        # ---------------------------------------------------------
        # 14. Clamp alpha
        # ---------------------------------------------------------

        final_alpha = torch.clamp(
            final_alpha,
            0.0,
            1.0,
        )

        dbg(
            "final_alpha",
            final_alpha,
        )

        # ---------------------------------------------------------
        # 15. Check whether alpha is actually non-zero
        # ---------------------------------------------------------

        alpha_nonzero = (
            final_alpha > 1e-6
        ).float().mean().item()

        alpha_gt_01 = (
            final_alpha > 0.01
        ).float().mean().item()

        alpha_gt_05 = (
            final_alpha > 0.05
        ).float().mean().item()

        print(
            "[SOILING DEBUG] alpha > 0:",
            f"{alpha_nonzero * 100:.2f}%",
        )

        print(
            "[SOILING DEBUG] alpha > 0.01:",
            f"{alpha_gt_01 * 100:.2f}%",
        )

        print(
            "[SOILING DEBUG] alpha > 0.05:",
            f"{alpha_gt_05 * 100:.2f}%",
        )

        # ---------------------------------------------------------
        # 16. Alpha blending
        # ---------------------------------------------------------

        final_alpha_3c = final_alpha.expand(
            -1,
            c,
            -1,
            -1,
        )

        dbg(
            "final_alpha_3c",
            final_alpha_3c,
        )

        # Difference before blending
        overlay_difference = (
            soil_rgb - image
        )

        dbg(
            "soil_rgb - image",
            overlay_difference,
        )

        # Actual blend
        soiled_image = (
            image * (1.0 - final_alpha_3c)
            + soil_rgb * final_alpha_3c
        )

        dbg(
            "soiled_image BEFORE clamp",
            soiled_image,
        )

        # ---------------------------------------------------------
        # 17. Clamp output
        # ---------------------------------------------------------

        soiled_image = torch.clamp(
            soiled_image,
            0.0,
            1.0,
        )

        dbg(
            "soiled_image FINAL",
            soiled_image,
        )

        # ---------------------------------------------------------
        # 18. Final difference
        # ---------------------------------------------------------

        diff = (
            soiled_image - image
        )

        abs_diff = diff.abs()

        print(
            "\n[SOILING DEBUG] FINAL IMAGE DIFFERENCE"
        )

        print(
            f"diff min  = {diff.min().item():.8f}"
        )

        print(
            f"diff max  = {diff.max().item():.8f}"
        )

        print(
            f"diff mean = {diff.mean().item():.8f}"
        )

        print(
            f"abs diff mean = {abs_diff.mean().item():.8f}"
        )

        print(
            "pixels changed > 1e-4:",
            f"{(abs_diff > 1e-4).float().mean().item() * 100:.2f}%"
        )

        print(
            "pixels changed > 1e-3:",
            f"{(abs_diff > 1e-3).float().mean().item() * 100:.2f}%"
        )

        print(
            "pixels changed > 1e-2:",
            f"{(abs_diff > 1e-2).float().mean().item() * 100:.2f}%"
        )

        print("=" * 70)
        print("[SOILING DEBUG] APPLY END")
        print("=" * 70 + "\n")

        return soiled_image, final_alpha