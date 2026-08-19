from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import kornia

from ..config import FlareConfig
from ..interfaces import BaseOcclusionModule


class FlareModule(BaseOcclusionModule):
    """
    Lens flare / light streak occlusion.

    Input:
        image:    [B, C, H, W]
        depth:    [B, 1, H, W]
        car_mask: [B, 1, H, W]
        soil_mask:[B, 1, H, W] or None

    Output:
        flare_image: [B, C, H, W]
        flare_mask:  [B, 1, H, W]
    """

    @torch.no_grad()
    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor | None,
        car_mask: torch.Tensor | None,
        cfg: FlareConfig,
        soil_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        DEBUG = True

        def dbg(name: str, x: Any):
            if not DEBUG:
                return

            if x is None:
                print(f"[FLARE DEBUG] {name}: None")
                return

            if not torch.is_tensor(x):
                print(f"[FLARE DEBUG] {name}: {x}")
                return

            print(
                f"[FLARE DEBUG] {name}: "
                f"shape={tuple(x.shape)}, "
                f"dtype={x.dtype}, "
                f"device={x.device}, "
                f"min={x.min().item():.6f}, "
                f"max={x.max().item():.6f}, "
                f"mean={x.mean().item():.6f}"
            )

        print("\n" + "=" * 70)
        print("[FLARE DEBUG] APPLY START")
        print("=" * 70)

        # =========================================================
        # 1. Input
        # =========================================================

        if image.ndim != 4:
            raise ValueError(
                f"[FLARE] image must be [B,C,H,W], got {image.shape}"
            )

        b, c, h, w = image.shape
        device = image.device
        dtype = image.dtype

        dbg("image", image)
        dbg("soil_mask RAW", soil_mask)

        print(
            f"[FLARE DEBUG] cfg.enabled   = {cfg.enabled}"
        )

        print(
            f"[FLARE DEBUG] cfg.intensity = {cfg.intensity}"
        )

        # =========================================================
        # 2. Early exit
        # =========================================================

        if not cfg.enabled or cfg.intensity <= 0.0:

            print("[FLARE DEBUG] DISABLED")

            return (
                image,
                torch.zeros(
                    (b, 1, h, w),
                    device=device,
                    dtype=dtype,
                ),
            )

        # =========================================================
        # 3. Normalize soil mask
        # =========================================================

        if soil_mask is None:

            print(
                "[FLARE DEBUG] No soil mask. "
                "Using zero mask."
            )

            soil_mask = torch.zeros(
                (b, 1, h, w),
                device=device,
                dtype=dtype,
            )

        else:

            soil_mask = soil_mask.to(
                device=device,
                dtype=dtype,
            )

            if soil_mask.ndim == 2:

                soil_mask = (
                    soil_mask
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

            elif soil_mask.ndim == 3:

                soil_mask = soil_mask.unsqueeze(1)

            elif soil_mask.ndim != 4:

                raise ValueError(
                    "[FLARE] Invalid soil_mask shape: "
                    f"{soil_mask.shape}"
                )

            if soil_mask.shape[0] == 1 and b > 1:

                soil_mask = soil_mask.expand(
                    b,
                    -1,
                    -1,
                    -1,
                )

            if soil_mask.shape[0] != b:

                raise ValueError(
                    "[FLARE] soil_mask batch mismatch: "
                    f"{soil_mask.shape[0]} vs {b}"
                )

            if soil_mask.shape[1] > 1:

                soil_mask = soil_mask.mean(
                    dim=1,
                    keepdim=True,
                )

            if soil_mask.shape[-2:] != (h, w):

                print(
                    "[FLARE DEBUG] Resizing soil mask:",
                    tuple(soil_mask.shape[-2:]),
                    "->",
                    (h, w),
                )

                soil_mask = F.interpolate(
                    soil_mask,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

        soil_mask = torch.clamp(
            soil_mask,
            0.0,
            1.0,
        )

        dbg(
            "soil_mask NORMALIZED",
            soil_mask,
        )

        # =========================================================
        # 4. Create flare streak
        # =========================================================

        # Base canvas.
        #
        # Start with a small square streak. It will be transformed
        # and then resized back to image resolution.
        #

        streak_size = min(
            max(64, min(h, w) // 2),
            512,
        )

        streak_size = int(streak_size)

        streak_tensor = torch.zeros(
            (1, 1, streak_size, streak_size),
            device=device,
            dtype=dtype,
        )

        # =========================================================
        # 5. Generate horizontal bright streak
        # =========================================================

        center_y = streak_size // 2

        line_width = max(
            1,
            streak_size // 40,
        )

        y0 = max(
            0,
            center_y - line_width,
        )

        y1 = min(
            streak_size,
            center_y + line_width + 1,
        )

        streak_tensor[
            :,
            :,
            y0:y1,
            :,
        ] = 1.0

        # =========================================================
        # 6. Gaussian blur
        # =========================================================

        kernel_size = max(
            3,
            (streak_size // 16) | 1,
        )

        streak_tensor = kornia.filters.gaussian_blur2d(
            streak_tensor,
            (
                kernel_size,
                kernel_size,
            ),
            (
                max(1.0, kernel_size / 6.0),
                max(1.0, kernel_size / 6.0),
            ),
        )

        dbg(
            "streak AFTER blur",
            streak_tensor,
        )

        # =========================================================
        # 7. Rotate / scale
        # =========================================================

        # IMPORTANT:
        #
        # Kornia affine requires [B, 2, 3].
        #

        streak_length = 1.5
        angle = -15.0

        theta = torch.tensor(
            angle
            * torch.pi
            / 180.0,
            device=device,
            dtype=dtype,
        )

        cos_a = torch.cos(theta)
        sin_a = torch.sin(theta)

        scale_x = torch.tensor(
            streak_length,
            device=device,
            dtype=dtype,
        )

        affine_matrix = torch.stack(
            [
                torch.stack(
                    [
                        cos_a * scale_x,
                        -sin_a,
                        torch.tensor(
                            0.0,
                            device=device,
                            dtype=dtype,
                        ),
                    ]
                ),
                torch.stack(
                    [
                        sin_a,
                        cos_a,
                        torch.tensor(
                            0.0,
                            device=device,
                            dtype=dtype,
                        ),
                    ]
                ),
            ]
        ).unsqueeze(0)

        dbg(
            "affine_matrix",
            affine_matrix,
        )

        streak_tensor = kornia.geometry.transform.affine(
            streak_tensor,
            affine_matrix,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        dbg(
            "streak AFTER affine",
            streak_tensor,
        )

        # =========================================================
        # 8. Resize flare to ORIGINAL image resolution
        # =========================================================

        if streak_tensor.shape[-2:] != (h, w):

            print(
                "[FLARE DEBUG] Resizing streak:",
                tuple(streak_tensor.shape[-2:]),
                "->",
                (h, w),
            )

            streak_tensor = F.interpolate(
                streak_tensor,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        dbg(
            "streak FINAL SIZE",
            streak_tensor,
        )

        # =========================================================
        # 9. Batch
        # =========================================================

        if b > 1:

            streak_tensor = streak_tensor.expand(
                b,
                -1,
                -1,
                -1,
            )

        # =========================================================
        # 10. RGB
        # =========================================================

        flare_rgb = streak_tensor.expand(
            -1,
            c,
            -1,
            -1,
        )

        dbg(
            "flare_rgb",
            flare_rgb,
        )

        # =========================================================
        # 11. Soil interaction
        # =========================================================

        # IMPORTANT:
        #
        # At this point BOTH tensors are guaranteed to be
        # [B,1,H,W].
        #

        if soil_mask is not None:

            scattered_soil = soil_mask

            if scattered_soil.shape[-2:] != (h, w):

                scattered_soil = F.interpolate(
                    scattered_soil,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

            flare_multiplier = (
                1.0
                - scattered_soil * 0.8
            )

        else:

            flare_multiplier = torch.ones(
                (b, 1, h, w),
                device=device,
                dtype=dtype,
            )

        flare_multiplier = torch.clamp(
            flare_multiplier,
            0.0,
            1.0,
        )

        dbg(
            "flare_multiplier",
            flare_multiplier,
        )

        # =========================================================
        # 12. Final flare
        # =========================================================

        flare_canvas = (
            flare_rgb
            * flare_multiplier
            * float(cfg.intensity)
        )

        dbg(
            "flare_canvas",
            flare_canvas,
        )

        # =========================================================
        # 13. Composite
        # =========================================================

        flare_alpha = (
            streak_tensor
            * flare_multiplier
            * float(cfg.intensity)
        )

        flare_alpha = torch.clamp(
            flare_alpha,
            0.0,
            1.0,
        )

        flare_alpha_3c = flare_alpha.expand(
            -1,
            c,
            -1,
            -1,
        )

        # Additive light flare.
        #
        # This is visually more appropriate than alpha-blending
        # a black background.
        #

        result = image + flare_canvas

        result = torch.clamp(
            result,
            0.0,
            1.0,
        )

        # =========================================================
        # 14. Debug
        # =========================================================

        diff = (
            result - image
        ).abs()

        print(
            "[FLARE DEBUG] changed mean:",
            diff.mean().item(),
        )

        print(
            "[FLARE DEBUG] changed max:",
            diff.max().item(),
        )

        print(
            "[FLARE DEBUG] pixels > 1e-3:",
            (
                diff > 1e-3
            ).float().mean().item()
            * 100,
            "%",
        )

        print("=" * 70)
        print("[FLARE DEBUG] APPLY END")
        print("=" * 70)

        return result, flare_alpha