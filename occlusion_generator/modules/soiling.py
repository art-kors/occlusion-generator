import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from ..interfaces import BaseOcclusionModule
from ..config import SoilingConfig


class SoilingModule(BaseOcclusionModule):
    """
    Applies dirt/soiling textures to an image.

    Expected:
        image:     [B, 3, H, W], float tensor in [0, 1]
        car_mask:  [B, 1, H, W] or [B, H, W], values in [0, 1]
        dirt_textures / dirt_buffer:
            list of PIL RGB/RGBA images or torch tensors.

    Returns:
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

        # ---------------------------------------------------------
        # Disabled
        # ---------------------------------------------------------
        if not cfg.enabled or cfg.intensity <= 0.0:
            return (
                image,
                torch.zeros_like(image[:, 0:1, :, :]),
            )

        # ---------------------------------------------------------
        # Validate image
        # ---------------------------------------------------------
        if image.ndim != 4:
            raise ValueError(
                f"SoilingModule expects image [B, C, H, W], "
                f"got {tuple(image.shape)}"
            )

        b, c, h, w = image.shape

        if c < 3:
            raise ValueError(
                f"SoilingModule expects at least 3 image channels, "
                f"got {c}"
            )

        device = image.device
        dtype = image.dtype

        # ---------------------------------------------------------
        # Support both:
        #
        # pipeline.process(..., dirt_buffer=patches)
        #
        # and:
        #
        # pipeline.process(..., **{"dirt_textures": patches})
        # ---------------------------------------------------------
        if dirt_buffer is None:
            dirt_buffer = kwargs.get("dirt_textures")

        if dirt_buffer is None or len(dirt_buffer) == 0:
            raise RuntimeError(
                "SoilingModule requires a non-empty dirt texture buffer. "
                "Pass it as dirt_buffer=... or dirt_textures=..."
            )

        # ---------------------------------------------------------
        # Prepare car mask
        # ---------------------------------------------------------
        if car_mask is None:
            car_mask = torch.ones(
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
                    f"car_mask must be [B, 1, H, W] or [B, H, W], "
                    f"got {tuple(car_mask.shape)}"
                )

            if car_mask.shape[0] != b:
                raise ValueError(
                    f"car_mask batch size {car_mask.shape[0]} "
                    f"does not match image batch size {b}"
                )

            if car_mask.shape[1] != 1:
                # If mask accidentally has multiple channels,
                # use the first one.
                car_mask = car_mask[:, :1]

            if car_mask.shape[-2:] != (h, w):
                car_mask = F.interpolate(
                    car_mask.float(),
                    size=(h, w),
                    mode="nearest",
                )

            car_mask = car_mask.to(
                device=device,
                dtype=dtype,
            )

            car_mask = car_mask.clamp(0.0, 1.0)

        # ---------------------------------------------------------
        # Intensity
        # ---------------------------------------------------------
        intensity = float(cfg.intensity)
        intensity = max(0.0, min(1.0, intensity))

        # Number of dirt patches.
        num_defects = max(
            1,
            int(round(intensity * 8)),
        )

        # ---------------------------------------------------------
        # Output buffers
        # ---------------------------------------------------------
        soil_texture = image.clone()

        soil_mask = torch.zeros(
            b,
            1,
            h,
            w,
            device=device,
            dtype=dtype,
        )

        # ---------------------------------------------------------
        # Generate dirt patches
        # ---------------------------------------------------------
        for _ in range(num_defects):

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
            # Select texture
            # -----------------------------------------------------
            texture_idx = torch.randint(
                0,
                len(dirt_buffer),
                (1,),
                device=device,
            ).item()

            item = dirt_buffer[texture_idx]

            # -----------------------------------------------------
            # Convert texture to [1, 4, H, W]
            # -----------------------------------------------------
            tex = self._texture_to_tensor(
                item=item,
                device=device,
                dtype=dtype,
            )

            # -----------------------------------------------------
            # Resize
            # -----------------------------------------------------
            tex_resized = F.interpolate(
                tex,
                size=(rand_h, rand_w),
                mode="bilinear",
                align_corners=False,
            )

            patch_rgb = tex_resized[:, :3, :, :]
            patch_alpha = tex_resized[:, 3:4, :, :]

            # Global intensity controls opacity.
            patch_alpha = patch_alpha * intensity

            patch_alpha = patch_alpha.clamp(0.0, 1.0)

            # -----------------------------------------------------
            # Calculate destination rectangle.
            #
            # The patch can extend outside the image.
            # We clip it safely.
            # -----------------------------------------------------
            x1 = cx - rand_w // 2
            y1 = cy - rand_h // 2

            x2 = x1 + rand_w
            y2 = y1 + rand_h

            # Destination coordinates in the image.
            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            dst_x2 = min(w, x2)
            dst_y2 = min(h, y2)

            # Source coordinates in the texture.
            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)

            # Nothing visible.
            if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                continue

            # -----------------------------------------------------
            # Full-frame patch.
            #
            # This is the important part:
            #
            # patch_rgb:
            #     [1, 3, rand_h, rand_w]
            #
            # becomes:
            #
            # full_patch_rgb:
            #     [1, 3, H, W]
            #
            # so it can safely interact with the input image.
            # -----------------------------------------------------
            full_patch_rgb = torch.zeros(
                1,
                3,
                h,
                w,
                device=device,
                dtype=dtype,
            )

            full_patch_alpha = torch.zeros(
                1,
                1,
                h,
                w,
                device=device,
                dtype=dtype,
            )

            full_patch_rgb[
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

            full_patch_alpha[
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
            # Apply only on the car.
            #
            # full_patch_alpha:
            #     [1, 1, H, W]
            #
            # car_mask:
            #     [B, 1, H, W]
            #
            # result:
            #     [B, 1, H, W]
            # -----------------------------------------------------
            full_patch_alpha = (
                full_patch_alpha * car_mask
            )

            # -----------------------------------------------------
            # Blend dirt into image.
            #
            # soil_texture:
            #     [B, 3, H, W]
            #
            # full_patch_rgb:
            #     [1, 3, H, W]
            #
            # full_patch_alpha:
            #     [B, 1, H, W]
            #
            # Broadcasting is intentional and safe.
            # -----------------------------------------------------
            soil_texture = (
                soil_texture * (1.0 - full_patch_alpha)
                + full_patch_rgb * full_patch_alpha
            )

            # -----------------------------------------------------
            # Keep strongest opacity where patches overlap.
            # -----------------------------------------------------
            soil_mask = torch.maximum(
                soil_mask,
                full_patch_alpha,
            )

        # ---------------------------------------------------------
        # Final safety clamp
        # ---------------------------------------------------------
        soil_texture = soil_texture.clamp(0.0, 1.0)
        soil_mask = soil_mask.clamp(0.0, 1.0)

        return soil_texture, soil_mask

    @staticmethod
    def _texture_to_tensor(
        item,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert a dirt texture into:
            [1, 4, H, W]

        Supported inputs:
            - PIL RGB
            - PIL RGBA
            - torch [3, H, W]
            - torch [4, H, W]
            - torch [1, 3, H, W]
            - torch [1, 4, H, W]
        """

        # =========================================================
        # PIL
        # =========================================================
        if isinstance(item, Image.Image):

            # Always normalize PIL textures to RGBA.
            item = item.convert("RGBA")

            arr = np.asarray(item)

            if arr.ndim != 3 or arr.shape[2] != 4:
                raise ValueError(
                    f"Expected RGBA PIL texture, got {arr.shape}"
                )

            tex = torch.from_numpy(arr)

            # HWC -> CHW
            tex = tex.permute(
                2,
                0,
                1,
            ).contiguous()

            tex = tex.float() / 255.0

            # [C,H,W] -> [1,C,H,W]
            tex = tex.unsqueeze(0)

        # =========================================================
        # Tensor
        # =========================================================
        elif torch.is_tensor(item):

            tex = item

            if tex.ndim == 3:
                tex = tex.unsqueeze(0)

            if tex.ndim != 4:
                raise ValueError(
                    "Dirt texture tensor must have shape "
                    "[C,H,W] or [B,C,H,W], "
                    f"got {tuple(tex.shape)}"
                )

            # We use one texture at a time.
            if tex.shape[0] > 1:
                tex = tex[:1]

            channels = tex.shape[1]

            # RGB -> RGBA
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
                    "Dirt texture tensor must have 3 or 4 channels, "
                    f"got {channels}"
                )

            tex = tex.float()

            # Handle uint8 / 0..255 tensors.
            if tex.numel() > 0 and tex.max() > 1.0:
                tex = tex / 255.0

        # =========================================================
        # Unsupported
        # =========================================================
        else:
            raise TypeError(
                "Unsupported dirt texture type: "
                f"{type(item)}. "
                "Expected PIL.Image.Image or torch.Tensor."
            )

        # =========================================================
        # Final normalization
        # =========================================================
        tex = tex.to(
            device=device,
            dtype=dtype,
        )

        tex = tex.clamp(0.0, 1.0)

        return tex