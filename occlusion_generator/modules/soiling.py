import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from ..interfaces import BaseOcclusionModule
from ..config import SoilingConfig


class SoilingModule(BaseOcclusionModule):
    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor,
        cfg: SoilingConfig,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not cfg.enabled or cfg.intensity <= 0.0:
            return image, torch.zeros_like(image[:, 0:1])

        if image.ndim != 4:
            raise ValueError(
                f"Expected image with shape [B, C, H, W], got {image.shape}"
            )

        if image.shape[1] < 3:
            raise ValueError(
                f"Expected image to have at least 3 channels, got {image.shape[1]}"
            )

        b, _, h, w = image.shape
        device = image.device
        dtype = image.dtype

        # ---------------------------------------------------------
        # Validate car mask
        # ---------------------------------------------------------
        if car_mask is None:
            car_mask = torch.ones(
                b, 1, h, w,
                device=device,
                dtype=dtype,
            )
        else:
            if car_mask.ndim == 3:
                car_mask = car_mask.unsqueeze(1)

            if car_mask.shape[0] != b:
                raise ValueError(
                    f"car_mask batch mismatch: image={b}, mask={car_mask.shape[0]}"
                )

            if car_mask.shape[-2:] != (h, w):
                car_mask = F.interpolate(
                    car_mask.float(),
                    size=(h, w),
                    mode="nearest",
                )

            car_mask = car_mask.to(device=device, dtype=dtype)
            car_mask = car_mask.clamp(0.0, 1.0)

        # ---------------------------------------------------------
        # Dirt textures
        # ---------------------------------------------------------
        dirt_buffer = kwargs.get("dirt_textures")

        if dirt_buffer is None or len(dirt_buffer) == 0:
            raise RuntimeError(
                "SoilingModule requires non-empty 'dirt_textures' buffer."
            )

        # Number of defects.
        # intensity=1.0 -> up to 8 defects.
        num_defects = max(
            1,
            int(round(float(cfg.intensity) * 8)),
        )

        # Keep intensity in a sane range.
        intensity = float(max(0.0, min(1.0, cfg.intensity)))

        soil_texture = image.clone()

        # This represents the strongest dirt opacity at every pixel.
        soil_mask = torch.zeros(
            b, 1, h, w,
            device=device,
            dtype=dtype,
        )

        # ---------------------------------------------------------
        # Generate dirt patches
        # ---------------------------------------------------------
        for _ in range(num_defects):

            min_size = max(2, int(min(h, w) * 0.05))
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
            # Get random texture
            # -----------------------------------------------------
            idx = torch.randint(
                0,
                len(dirt_buffer),
                (1,),
                device=device,
            ).item()

            item = dirt_buffer[idx]

            # PIL -> Tensor
            if isinstance(item, Image.Image):
                item = item.convert("RGBA")

                arr = np.asarray(item)

                tex = torch.from_numpy(arr)

                if tex.ndim != 3 or tex.shape[2] != 4:
                    raise ValueError(
                        f"Expected RGBA texture, got shape {tex.shape}"
                    )

                tex = tex.permute(2, 0, 1).contiguous()
                tex = tex.float() / 255.0

            # Tensor
            elif torch.is_tensor(item):
                tex = item

                if tex.ndim == 3:
                    tex = tex.unsqueeze(0)

                if tex.ndim != 4:
                    raise ValueError(
                        f"Texture tensor must be [C,H,W] or [B,C,H,W], "
                        f"got {tex.shape}"
                    )

                # Dirt texture should describe one patch.
                if tex.shape[0] != 1:
                    tex = tex[:1]

                if tex.shape[1] == 3:
                    # RGB -> RGBA
                    alpha = torch.ones(
                        1,
                        1,
                        tex.shape[2],
                        tex.shape[3],
                        device=tex.device,
                        dtype=tex.dtype,
                    )
                    tex = torch.cat([tex, alpha], dim=1)

                elif tex.shape[1] != 4:
                    raise ValueError(
                        f"Expected RGB/RGBA texture, got {tex.shape[1]} channels"
                    )

                tex = tex.float()

                # Normalize uint8-like tensors.
                if tex.max() > 1.0:
                    tex = tex / 255.0

            else:
                raise TypeError(
                    f"Unsupported dirt texture type: {type(item)}"
                )

            tex = tex.to(
                device=device,
                dtype=dtype,
            )

            # -----------------------------------------------------
            # Resize texture
            # -----------------------------------------------------
            tex_resized = F.interpolate(
                tex,
                size=(rand_h, rand_w),
                mode="bilinear",
                align_corners=False,
            )

            patch_rgb = tex_resized[:, :3]
            patch_alpha = tex_resized[:, 3:4]

            # -----------------------------------------------------
            # Apply global intensity to opacity
            # -----------------------------------------------------
            patch_alpha = patch_alpha * intensity

            # -----------------------------------------------------
            # Position + clipping
            # -----------------------------------------------------
            x1 = cx - rand_w // 2
            y1 = cy - rand_h // 2
            x2 = x1 + rand_w
            y2 = y1 + rand_h

            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)
            src_x2 = rand_w - max(0, x2 - w)
            src_y2 = rand_h - max(0, y2 - h)

            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            dst_x2 = min(w, x2)
            dst_y2 = min(h, y2)

            # Nothing visible.
            if src_x2 <= src_x1 or src_y2 <= src_y1:
                continue

            # -----------------------------------------------------
            # Build full-size patch
            # -----------------------------------------------------
            full_patch_rgb = torch.zeros(
                1, 3, h, w,
                device=device,
                dtype=dtype,
            )

            full_patch_alpha = torch.zeros(
                1, 1, h, w,
                device=device,
                dtype=dtype,
            )

            full_patch_rgb[
                :, :,
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
            # Apply only to the car
            # -----------------------------------------------------
            full_patch_alpha = full_patch_alpha * car_mask

            # -----------------------------------------------------
            # Apply dirt to every image in the batch
            # -----------------------------------------------------
            soil_texture = (
                soil_texture * (1.0 - full_patch_alpha)
                + full_patch_rgb * full_patch_alpha
            )

            # Strongest dirt opacity at every pixel.
            soil_mask = torch.maximum(
                soil_mask,
                full_patch_alpha,
            )

        return soil_texture.clamp(0.0, 1.0), soil_mask