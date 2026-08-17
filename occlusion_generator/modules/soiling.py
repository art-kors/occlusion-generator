import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from ..interfaces import BaseOcclusionModule
from ..config import SoilingConfig


class SoilingModule(BaseOcclusionModule):
    """
    Applies dirt/soiling textures to an image.

    Input:
        image:    [B, 3, H, W], expected in [0, 1]
        car_mask: [B, 1, H, W] or [B, H, W], expected in [0, 1]

    dirt_buffer:
        List of PIL RGB/RGBA images or torch tensors.

    Output:
        soil_texture: [B, 3, H, W]
        soil_mask:    [B, 1, H, W]
    """

    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor,
        cfg: SoilingConfig,
        dirt_buffer=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # =========================================================
        # Disabled
        # =========================================================

        if not cfg.enabled or cfg.intensity <= 0.0:
            return (
                image,
                torch.zeros_like(image[:, 0:1, :, :]),
            )

        # =========================================================
        # Validate image
        # =========================================================

        if image.ndim != 4:
            raise ValueError(
                f"SoilingModule expects image [B,C,H,W], "
                f"got {tuple(image.shape)}"
            )

        b, c, h, w = image.shape

        if c < 3:
            raise ValueError(
                f"SoilingModule expects at least 3 channels, "
                f"got {c}"
            )

        device = image.device
        dtype = image.dtype

        # =========================================================
        # Get dirt buffer
        # =========================================================

        if dirt_buffer is None:
            dirt_buffer = kwargs.get("dirt_textures")

        if dirt_buffer is None or len(dirt_buffer) == 0:
            raise RuntimeError(
                "SoilingModule: dirt_buffer is empty. "
                "Pass dirt_buffer=... or dirt_textures=..."
            )

        # =========================================================
        # Prepare image
        # =========================================================

        soil_texture = image[:, :3].clone()

        # Make sure blending operates in [0, 1].
        soil_texture = soil_texture.clamp(0.0, 1.0)

        # =========================================================
        # Prepare car mask
        # =========================================================

        if car_mask is None:

            print(
                "[Soiling] WARNING: car_mask is None. "
                "Using full-frame mask for debugging."
            )

            effective_car_mask = torch.ones(
                b,
                1,
                h,
                w,
                device=device,
                dtype=dtype,
            )

        else:

            if car_mask.ndim == 3:
                car_mask = car_mask.unsqueeze(1)

            if car_mask.ndim != 4:
                raise ValueError(
                    f"car_mask must be [B,1,H,W] or [B,H,W], "
                    f"got {tuple(car_mask.shape)}"
                )

            if car_mask.shape[0] != b:
                raise ValueError(
                    f"car_mask batch={car_mask.shape[0]} "
                    f"does not match image batch={b}"
                )

            if car_mask.shape[-2:] != (h, w):

                car_mask = F.interpolate(
                    car_mask.float(),
                    size=(h, w),
                    mode="nearest",
                )

            if car_mask.shape[1] > 1:
                car_mask = car_mask[:, :1]

            car_mask = car_mask.to(
                device=device,
                dtype=dtype,
            )

            car_mask = car_mask.clamp(0.0, 1.0)

            mask_min = car_mask.min().item()
            mask_max = car_mask.max().item()

            print(
                f"[Soiling] car_mask_range="
                f"({mask_min:.3f}, {mask_max:.3f})"
            )

            # -----------------------------------------------------
            # TEMPORARY DEBUG FALLBACK
            #
            # Your current pipeline produces:
            #
            # car_mask_range=(0.000, 0.000)
            #
            # That completely removes every dirt patch.
            #
            # For now, use full-frame mask so we can verify
            # that the actual texture compositing works.
            # -----------------------------------------------------

            if mask_max <= 0.0:

                print(
                    "[Soiling] WARNING: car_mask is completely empty. "
                    "Ignoring car_mask temporarily."
                )

                effective_car_mask = torch.ones_like(
                    car_mask
                )

            else:

                effective_car_mask = car_mask

        # =========================================================
        # Intensity
        # =========================================================

        intensity = float(cfg.intensity)

        intensity = max(
            0.0,
            min(1.0, intensity),
        )

        # 1.0 -> 8 patches
        num_defects = max(
            1,
            int(round(intensity * 8)),
        )

        print(
            f"[Soiling] image={tuple(image.shape)} "
            f"image_range=("
            f"{soil_texture.min().item():.3f},"
            f"{soil_texture.max().item():.3f}) "
            f"textures={len(dirt_buffer)} "
            f"intensity={intensity:.3f} "
            f"defects={num_defects}"
        )

        # =========================================================
        # Soil mask
        # =========================================================

        soil_mask = torch.zeros(
            b,
            1,
            h,
            w,
            device=device,
            dtype=dtype,
        )

        # =========================================================
        # Generate patches
        # =========================================================

        for defect_idx in range(num_defects):

            # -----------------------------------------------------
            # Random patch size
            # -----------------------------------------------------

            min_size = max(
                2,
                int(min(h, w) * 0.05),
            )

            max_size = max(
                min_size + 1,
                int(min(h, w) * 0.40),
            )

            rand_h = torch.randint(
                min_size,
                max_size + 1,
                (1,),
                device=device,
            ).item()

            rand_w = torch.randint(
                min_size,
                max_size + 1,
                (1,),
                device=device,
            ).item()

            # -----------------------------------------------------
            # Random center
            # -----------------------------------------------------

            cx = torch.randint(
                0,
                w,
                (1,),
                device=device,
            ).item()

            cy = torch.randint(
                0,
                h,
                (1,),
                device=device,
            ).item()

            # -----------------------------------------------------
            # Select random texture
            # -----------------------------------------------------

            texture_idx = torch.randint(
                0,
                len(dirt_buffer),
                (1,),
                device=device,
            ).item()

            item = dirt_buffer[texture_idx]

            # -----------------------------------------------------
            # Convert texture
            # -----------------------------------------------------

            tex = self._texture_to_tensor(
                item=item,
                device=device,
                dtype=dtype,
            )

            # -----------------------------------------------------
            # Resize texture
            # -----------------------------------------------------

            tex = F.interpolate(
                tex,
                size=(rand_h, rand_w),
                mode="bilinear",
                align_corners=False,
            )

            patch_rgb = tex[:, :3, :, :]
            patch_alpha = tex[:, 3:4, :, :]

            # -----------------------------------------------------
            # Apply intensity to alpha
            # -----------------------------------------------------

            patch_alpha = (
                patch_alpha * intensity
            ).clamp(0.0, 1.0)

            # -----------------------------------------------------
            # Calculate position
            # -----------------------------------------------------

            x1 = cx - rand_w // 2
            y1 = cy - rand_h // 2

            x2 = x1 + rand_w
            y2 = y1 + rand_h

            # Destination coordinates in image
            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            dst_x2 = min(w, x2)
            dst_y2 = min(h, y2)

            # Source coordinates in texture
            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)

            src_x2 = src_x1 + (
                dst_x2 - dst_x1
            )

            src_y2 = src_y1 + (
                dst_y2 - dst_y1
            )

            # -----------------------------------------------------
            # Patch completely outside image
            # -----------------------------------------------------

            if (
                dst_x2 <= dst_x1
                or dst_y2 <= dst_y1
            ):
                continue

            # =====================================================
            # Create full-frame patch
            # =====================================================

            full_rgb = torch.zeros(
                1,
                3,
                h,
                w,
                device=device,
                dtype=dtype,
            )

            full_alpha = torch.zeros(
                1,
                1,
                h,
                w,
                device=device,
                dtype=dtype,
            )

            # -----------------------------------------------------
            # Put RGB texture into full-frame canvas
            # -----------------------------------------------------

            full_rgb[
                :,
                :,
                dst_y1:dst_y2,
                dst_x1:dst_x2,
            ] = patch_rgb[
                :,
                :,
                src_y1:src_y2,
                src_x1:src_x2,
            ]

            # -----------------------------------------------------
            # Put alpha into full-frame canvas
            # -----------------------------------------------------

            full_alpha[
                :,
                :,
                dst_y1:dst_y2,
                dst_x1:dst_x2,
            ] = patch_alpha[
                :,
                :,
                src_y1:src_y2,
                src_x1:src_x2,
            ]

            # -----------------------------------------------------
            # Debug before car mask
            # -----------------------------------------------------

            print(
                f"[Soiling] patch "
                f"{defect_idx + 1}/{num_defects}: "
                f"pos=({cx},{cy}) "
                f"size=({rand_w},{rand_h}) "
                f"alpha_before_mask="
                f"{full_alpha.max().item():.3f}"
            )

            # =====================================================
            # Apply car mask
            # =====================================================

            full_alpha = (
                full_alpha * effective_car_mask
            )

            print(
                f"[Soiling] patch "
                f"{defect_idx + 1}: "
                f"alpha_after_mask="
                f"{full_alpha.max().item():.3f}"
            )

            # =====================================================
            # Expand RGB to batch
            # =====================================================

            full_rgb = full_rgb.expand(
                b,
                -1,
                -1,
                -1,
            )

            # =====================================================
            # Alpha blend
            # =====================================================

            soil_texture = (
                soil_texture * (1.0 - full_alpha)
                + full_rgb * full_alpha
            )

            # =====================================================
            # Update mask
            # =====================================================

            soil_mask = torch.maximum(
                soil_mask,
                full_alpha,
            )

        # =========================================================
        # Final clamp
        # =========================================================

        soil_texture = soil_texture.clamp(
            0.0,
            1.0,
        )

        soil_mask = soil_mask.clamp(
            0.0,
            1.0,
        )

        print(
            "[Soiling] FINAL: "
            f"mask=("
            f"{soil_mask.min().item():.3f},"
            f"{soil_mask.max().item():.3f}) "
            f"image=("
            f"{soil_texture.min().item():.3f},"
            f"{soil_texture.max().item():.3f})"
        )

        return soil_texture, soil_mask

    # =============================================================
    # Texture conversion
    # =============================================================

    @staticmethod
    def _texture_to_tensor(
        item,
        device,
        dtype,
    ) -> torch.Tensor:

        # ---------------------------------------------------------
        # PIL Image
        # ---------------------------------------------------------

        if isinstance(item, Image.Image):

            # Force RGBA
            item = item.convert("RGBA")

            # .copy() avoids PyTorch warning about read-only
            # NumPy memory returned by np.asarray(PIL).
            arr = np.asarray(item).copy()

            if arr.ndim != 3 or arr.shape[2] != 4:
                raise ValueError(
                    f"Expected RGBA texture, got {arr.shape}"
                )

            # HWC -> CHW
            tex = torch.from_numpy(
                arr
            ).permute(
                2,
                0,
                1,
            ).contiguous()

            tex = tex.float() / 255.0

            # CHW -> BCHW
            tex = tex.unsqueeze(0)

        # ---------------------------------------------------------
        # Tensor
        # ---------------------------------------------------------

        elif torch.is_tensor(item):

            tex = item

            # CHW -> BCHW
            if tex.ndim == 3:
                tex = tex.unsqueeze(0)

            if tex.ndim != 4:
                raise ValueError(
                    "Texture tensor must be "
                    "[C,H,W] or [B,C,H,W], "
                    f"got {tuple(tex.shape)}"
                )

            # One texture at a time
            if tex.shape[0] > 1:
                tex = tex[:1]

            channels = tex.shape[1]

            # -----------------------------------------------------
            # RGB -> RGBA
            # -----------------------------------------------------

            if channels == 3:

                alpha = torch.ones(
                    tex.shape[0],
                    1,
                    tex.shape[2],
                    tex.shape[3],
                    device=tex.device,
                    dtype=tex.dtype,
                )

                tex = torch.cat(
                    [tex, alpha],
                    dim=1,
                )

            elif channels != 4:

                raise ValueError(
                    "Texture tensor must have "
                    "3 or 4 channels, "
                    f"got {channels}"
                )

            tex = tex.float()

            # uint8 / 0..255 -> 0..1
            if (
                tex.numel() > 0
                and tex.max() > 1.0
            ):
                tex = tex / 255.0

        # ---------------------------------------------------------
        # Unsupported type
        # ---------------------------------------------------------

        else:

            raise TypeError(
                f"Unsupported dirt texture type: "
                f"{type(item)}. "
                "Expected PIL.Image.Image "
                "or torch.Tensor."
            )

        # ---------------------------------------------------------
        # Device / dtype / range
        # ---------------------------------------------------------

        tex = tex.to(
            device=device,
            dtype=dtype,
        )

        tex = tex.clamp(
            0.0,
            1.0,
        )

        return tex