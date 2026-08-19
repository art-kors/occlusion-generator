from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


class SoilingModule(BaseOcclusionModule):
    """
    Randomly placed lens-soiling / dirt patch.

    Input:
        image:     [B, C, H, W]
        depth:     [B, 1, H, W] / [B, H, W]
        car_mask:  [B, 1, H, W] / [B, H, W]

    dirt_buffer:
        [H, W]
        [C, H, W]
        [B, C, H, W]
        [H, W, C]
        [B, H, W, C]

    Supported texture channels:
        1 -> grayscale
        3 -> RGB
        4 -> RGBA

    Returns:
        image: [B, C, H, W]
        mask:  [B, 1, H, W]
    """

    @torch.no_grad()
    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor | None,
        car_mask: torch.Tensor | None,
        cfg: SoilingConfig,
        **kwargs: Any,
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

        # =========================================================
        # START
        # =========================================================

        print("\n" + "=" * 70)
        print("[SOILING DEBUG] APPLY START")
        print("=" * 70)

        # =========================================================
        # 1. INPUT
        # =========================================================

        if image.ndim != 4:
            raise ValueError(
                f"[SOILING] image must be [B,C,H,W], "
                f"got {image.shape}"
            )

        b, c, h, w = image.shape

        device = image.device
        dtype = image.dtype

        print(
            f"[SOILING DEBUG] cfg.enabled   = {cfg.enabled}"
        )

        print(
            f"[SOILING DEBUG] cfg.intensity = {cfg.intensity}"
        )

        dbg("image INPUT", image)

        # =========================================================
        # 2. EARLY EXIT
        # =========================================================

        if not cfg.enabled or cfg.intensity <= 0.0:

            print(
                "[SOILING DEBUG] Soiling disabled."
            )

            return (
                image,
                torch.zeros(
                    (b, 1, h, w),
                    device=device,
                    dtype=dtype,
                ),
            )

        # =========================================================
        # 3. CAR MASK
        # =========================================================

        if car_mask is None:

            print(
                "[SOILING DEBUG] car_mask=None -> "
                "using FULL IMAGE mask"
            )

            car_mask = torch.ones(
                (b, 1, h, w),
                device=device,
                dtype=dtype,
            )

        else:

            car_mask = car_mask.to(
                device=device,
                dtype=dtype,
            )

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
                    "[SOILING] Invalid car_mask shape: "
                    f"{car_mask.shape}"
                )

            if car_mask.shape[0] == 1 and b > 1:

                car_mask = car_mask.expand(
                    b,
                    -1,
                    -1,
                    -1,
                )

            if car_mask.shape[0] != b:

                raise ValueError(
                    "[SOILING] car_mask batch mismatch: "
                    f"{car_mask.shape[0]} vs {b}"
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

        dbg(
            "car_mask",
            car_mask,
        )

        # =========================================================
        # 4. GET DIRT PATCH
        # =========================================================

        dirt = kwargs.get(
            "dirt_buffer",
            None,
        )

        dbg(
            "dirt_buffer RAW",
            dirt,
        )

        if dirt is None:

            print(
                "[SOILING DEBUG] No dirt_buffer."
            )

            return (
                image,
                torch.zeros(
                    (b, 1, h, w),
                    device=device,
                    dtype=dtype,
                ),
            )

        # ---------------------------------------------------------
        # List -> tensor
        # ---------------------------------------------------------

        if isinstance(dirt, list):

            if len(dirt) == 0:

                raise ValueError(
                    "[SOILING] dirt_buffer is empty."
                )

            tensors = []

            for item in dirt:

                if not torch.is_tensor(item):

                    raise TypeError(
                        "[SOILING] dirt_buffer list "
                        "must contain tensors."
                    )

                tensors.append(item)

            # If all patches same shape -> stack.
            dirt = torch.stack(
                tensors,
                dim=0,
            )

        if not torch.is_tensor(dirt):

            raise TypeError(
                "[SOILING] dirt_buffer must be "
                f"Tensor or list, got {type(dirt)}"
            )

        dirt = dirt.to(
            device=device,
            dtype=dtype,
        )

        dbg(
            "dirt AFTER tensor conversion",
            dirt,
        )

        # =========================================================
        # 5. NORMALIZE DIRT DIMENSIONS
        # =========================================================

        if dirt.ndim == 2:

            # [H,W]

            dirt = dirt.unsqueeze(
                0
            ).unsqueeze(
                0
            )

        elif dirt.ndim == 3:

            # Usually [C,H,W]
            #
            # But if last dimension looks like channel,
            # assume [H,W,C].

            if dirt.shape[-1] in (1, 3, 4):

                # [H,W,C]

                dirt = dirt.permute(
                    2,
                    0,
                    1,
                ).unsqueeze(0)

            else:

                # [C,H,W]

                dirt = dirt.unsqueeze(0)

        elif dirt.ndim == 4:

            # Could be:
            #
            # [B,C,H,W]
            # or
            # [B,H,W,C]

            if dirt.shape[-1] in (1, 3, 4):

                # [B,H,W,C]
                dirt = dirt.permute(
                    0,
                    3,
                    1,
                    2,
                )

            elif dirt.shape[1] in (1, 3, 4):

                # Already [B,C,H,W]
                pass

            else:

                raise ValueError(
                    "[SOILING] Cannot determine "
                    "channel dimension from dirt shape: "
                    f"{dirt.shape}"
                )

        else:

            raise ValueError(
                "[SOILING] Unsupported dirt shape: "
                f"{dirt.shape}"
            )

        dbg(
            "dirt NORMALIZED [B,C,H,W]",
            dirt,
        )

        # =========================================================
        # 6. BATCH HANDLING
        # =========================================================

        dirt_batch = dirt.shape[0]

        if dirt_batch == 1 and b > 1:

            dirt = dirt.expand(
                b,
                -1,
                -1,
                -1,
            )

        elif dirt_batch != b:

            # Multiple patches can be supplied.
            #
            # For a single image, randomly select one patch.
            #

            if b == 1:

                index = torch.randint(
                    0,
                    dirt_batch,
                    (1,),
                    device=device,
                ).item()

                print(
                    "[SOILING DEBUG] Selecting "
                    f"random patch {index}/{dirt_batch}"
                )

                dirt = dirt[
                    index:index + 1
                ]

            else:

                raise ValueError(
                    "[SOILING] dirt batch mismatch: "
                    f"{dirt_batch} vs image batch {b}"
                )

        # =========================================================
        # 7. CHANNELS
        # =========================================================

        channels = dirt.shape[1]

        print(
            "[SOILING DEBUG] dirt channels:",
            channels,
        )

        if channels == 1:

            dirt_rgb = dirt.expand(
                -1,
                3,
                -1,
                -1,
            )

            dirt_alpha = dirt

        elif channels == 3:

            dirt_rgb = dirt

            # RGB has no explicit alpha.
            #
            # Normalize brightness into a useful alpha.
            #

            brightness = dirt.mean(
                dim=1,
                keepdim=True,
            )

            # Normalize per patch.
            #
            # This prevents very dark patches from becoming
            # practically invisible.
            #

            dmin = brightness.amin(
                dim=(-2, -1),
                keepdim=True,
            )

            dmax = brightness.amax(
                dim=(-2, -1),
                keepdim=True,
            )

            brightness = (
                brightness - dmin
            ) / (
                dmax - dmin + 1e-6
            )

            dirt_alpha = brightness

        elif channels == 4:

            print(
                "[SOILING DEBUG] RGBA dirt detected."
            )

            dirt_rgb = dirt[:, :3]

            dirt_alpha = dirt[:, 3:4]

        else:

            raise ValueError(
                "[SOILING] Dirt texture must have "
                "1, 3 or 4 channels. Got "
                f"{channels}"
            )

        dbg(
            "dirt_rgb",
            dirt_rgb,
        )

        dbg(
            "dirt_alpha",
            dirt_alpha,
        )

        # =========================================================
        # 8. PATCH VALUE NORMALIZATION
        # =========================================================

        # Your previous patches had values around 1e-5.
        #
        # If that happens, automatically normalize them.
        #

        rgb_max = dirt_rgb.amax().item()

        if rgb_max > 1.0:

            print(
                "[SOILING DEBUG] RGB > 1.0 -> "
                "normalizing."
            )

            dirt_rgb = torch.clamp(
                dirt_rgb / 255.0,
                0.0,
                1.0,
            )

        elif 0.0 < rgb_max < 1e-3:

            print(
                "[SOILING DEBUG] VERY SMALL RGB "
                f"range detected ({rgb_max:.8e})."
            )

            print(
                "[SOILING DEBUG] Normalizing dirt "
                "RGB per patch."
            )

            rgb_min = dirt_rgb.amin(
                dim=(-3, -2, -1),
                keepdim=True,
            )

            rgb_max_tensor = dirt_rgb.amax(
                dim=(-3, -2, -1),
                keepdim=True,
            )

            dirt_rgb = (
                dirt_rgb - rgb_min
            ) / (
                rgb_max_tensor
                - rgb_min
                + 1e-8
            )

        dirt_rgb = torch.clamp(
            dirt_rgb,
            0.0,
            1.0,
        )

        dirt_alpha = torch.clamp(
            dirt_alpha,
            0.0,
            1.0,
        )

        # =========================================================
        # 9. RANDOM TRANSFORM
        # =========================================================

        # Defaults are intentionally moderate.
        #
        # Patch is NOT resized to image dimensions.
        #

        rotation_min = -45.0
        rotation_max = 45.0

        scale_min = 0.7
        scale_max = 1.3

        # Optional config attributes.
        #

        if hasattr(
            cfg,
            "rotation_range",
        ):

            rotation_min = float(
                cfg.rotation_range[0]
            )

            rotation_max = float(
                cfg.rotation_range[1]
            )

        if hasattr(
            cfg,
            "scale_range",
        ):

            scale_min = float(
                cfg.scale_range[0]
            )

            scale_max = float(
                cfg.scale_range[1]
            )

        # =========================================================
        # 10. RANDOM ROTATION
        # =========================================================

        angle = torch.empty(
            b,
            device=device,
            dtype=dtype,
        ).uniform_(
            rotation_min,
            rotation_max,
        )

        # =========================================================
        # 11. RANDOM SCALE
        # =========================================================

        scale = torch.empty(
            b,
            device=device,
            dtype=dtype,
        ).uniform_(
            scale_min,
            scale_max,
        )

        print(
            "[SOILING DEBUG] rotation:",
            angle.detach().cpu().numpy(),
        )

        print(
            "[SOILING DEBUG] scale:",
            scale.detach().cpu().numpy(),
        )

        # =========================================================
        # 12. AFFINE TRANSFORM
        # =========================================================

        radians = (
            angle
            * torch.pi
            / 180.0
        )

        cos_a = torch.cos(
            radians
        ) * scale

        sin_a = torch.sin(
            radians
        ) * scale

        theta = torch.zeros(
            (
                b,
                2,
                3,
            ),
            device=device,
            dtype=dtype,
        )

        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a

        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        # =========================================================
        # 13. RANDOM POSITION
        # =========================================================
        #
        # Translation in affine_grid is normalized:
        #
        # -1 = left / top
        #  0 = center
        # +1 = right / bottom
        #

        tx = torch.empty(
            b,
            device=device,
            dtype=dtype,
        ).uniform_(
            -0.55,
            0.55,
        )

        ty = torch.empty(
            b,
            device=device,
            dtype=dtype,
        ).uniform_(
            -0.55,
            0.55,
        )

        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        dbg(
            "theta",
            theta,
        )

        # =========================================================
        # 14. CREATE IMAGE-SIZED CANVAS
        # =========================================================

        output_size = (
            b,
            1,
            h,
            w,
        )

        # =========================================================
        # 15. TRANSFORM RGB
        # =========================================================

        dirt_rgb_canvas = F.affine_grid(
            theta,
            (
                b,
                3,
                h,
                w,
            ),
            align_corners=False,
        )

        dirt_rgb_canvas = F.grid_sample(
            dirt_rgb,
            dirt_rgb_canvas,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        dbg(
            "dirt_rgb_canvas",
            dirt_rgb_canvas,
        )

        # =========================================================
        # 16. TRANSFORM ALPHA
        # =========================================================

        dirt_alpha_canvas = F.affine_grid(
            theta,
            output_size,
            align_corners=False,
        )

        dirt_alpha_canvas = F.grid_sample(
            dirt_alpha,
            dirt_alpha_canvas,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        dbg(
            "dirt_alpha_canvas",
            dirt_alpha_canvas,
        )

        # =========================================================
        # 17. MASK / POSITION
        # =========================================================

        # The transformed patch already contains zero outside
        # the patch, so no additional bounding-box placement
        # is required.
        #

        final_alpha = (
            dirt_alpha_canvas
            * car_mask
            * float(cfg.intensity)
        )

        final_alpha = torch.clamp(
            final_alpha,
            0.0,
            1.0,
        )

        dbg(
            "final_alpha",
            final_alpha,
        )

        # =========================================================
        # 18. RGB CHANNEL COUNT
        # =========================================================

        if c == 1:

            dirt_rgb_canvas = dirt_rgb_canvas.mean(
                dim=1,
                keepdim=True,
            )

        elif c == 3:

            pass

        else:

            raise ValueError(
                "[SOILING] image must have "
                f"1 or 3 channels, got {c}"
            )

        # =========================================================
        # 19. ALPHA BLENDING
        # =========================================================

        alpha_3c = final_alpha.expand(
            -1,
            c,
            -1,
            -1,
        )

        result = (
            image * (1.0 - alpha_3c)
            + dirt_rgb_canvas * alpha_3c
        )

        result = torch.clamp(
            result,
            0.0,
            1.0,
        )

        # =========================================================
        # 20. DEBUG DIFFERENCE
        # =========================================================

        diff = (
            result - image
        ).abs()

        print(
            "\n[SOILING DEBUG] FINAL"
        )

        print(
            "result:",
            tuple(result.shape),
        )

        print(
            "alpha mean:",
            final_alpha.mean().item(),
        )

        print(
            "alpha max:",
            final_alpha.max().item(),
        )

        print(
            "changed mean:",
            diff.mean().item(),
        )

        print(
            "changed max:",
            diff.max().item(),
        )

        print(
            "pixels > 1e-3:",
            (
                diff > 1e-3
            ).float().mean().item()
            * 100,
            "%",
        )

        print(
            "pixels > 1e-2:",
            (
                diff > 1e-2
            ).float().mean().item()
            * 100,
            "%",
        )

        print("=" * 70)
        print("[SOILING DEBUG] APPLY END")
        print("=" * 70)

        return result, final_alpha