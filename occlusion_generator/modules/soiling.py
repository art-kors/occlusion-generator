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
        dirt_buffer=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not cfg.enabled or cfg.intensity <= 0.0:
            return (
                image,
                torch.zeros_like(image[:, 0:1, :, :]),
            )

        if image.ndim != 4:
            raise ValueError(
                f"Expected image [B,C,H,W], got {tuple(image.shape)}"
            )

        b, c, h, w = image.shape

        if c < 3:
            raise ValueError(
                f"Expected at least 3 channels, got {c}"
            )

        device = image.device
        dtype = image.dtype

        # ---------------------------------------------------------
        # Get dirt buffer
        # ---------------------------------------------------------

        if dirt_buffer is None:
            dirt_buffer = kwargs.get("dirt_textures")

        if dirt_buffer is None or len(dirt_buffer) == 0:
            raise RuntimeError(
                "SoilingModule: dirt_buffer is empty."
            )

        # ---------------------------------------------------------
        # Prepare image
        # ---------------------------------------------------------

        soil_texture = image[:, :3].clone()

        # Important:
        # Keep image in valid range.
        soil_texture = soil_texture.clamp(0.0, 1.0)

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

        # ---------------------------------------------------------
        # DEBUG
        # ---------------------------------------------------------

        print(
            "[Soiling]"
            f" image={tuple(image.shape)}"
            f" image_range=({soil_texture.min().item():.3f},"
            f"{soil_texture.max().item():.3f})"
            f" car_mask_range=({car_mask.min().item():.3f},"
            f"{car_mask.max().item():.3f})"
            f" textures={len(dirt_buffer)}"
        )

        # ---------------------------------------------------------
        # Intensity
        # ---------------------------------------------------------

        intensity = float(cfg.intensity)
        intensity = max(0.0, min(1.0, intensity))

        num_defects = max(
            1,
            int(round(intensity * 8)),
        )

        # ---------------------------------------------------------
        # Output mask
        # ---------------------------------------------------------

        soil_mask = torch.zeros(
            b,
            1,
            h,
            w,
            device=device,
            dtype=dtype,
        )

        # ---------------------------------------------------------
        # Generate patches
        # ---------------------------------------------------------

        for defect_idx in range(num_defects):

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

            idx = torch.randint(
                0,
                len(dirt_buffer),
                (1,),
                device=device,
            ).item()

            tex = self._texture_to_tensor(
                dirt_buffer[idx],
                device=device,
                dtype=dtype,
            )

            # -----------------------------------------------------
            # Resize
            # -----------------------------------------------------

            tex = F.interpolate(
                tex,
                size=(rand_h, rand_w),
                mode="bilinear",
                align_corners=False,
            )

            patch_rgb = tex[:, :3]
            patch_alpha = tex[:, 3:4]

            # -----------------------------------------------------
            # Alpha
            # -----------------------------------------------------

            patch_alpha = (
                patch_alpha * intensity
            ).clamp(0.0, 1.0)

            # -----------------------------------------------------
            # Coordinates
            # -----------------------------------------------------

            x1 = cx - rand_w // 2
            y1 = cy - rand_h // 2

            x2 = x1 + rand_w
            y2 = y1 + rand_h

            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            dst_x2 = min(w, x2)
            dst_y2 = min(h, y2)

            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)

            src_x2 = src_x1 + (
                dst_x2 - dst_x1
            )

            src_y2 = src_y1 + (
                dst_y2 - dst_y1
            )

            if (
                dst_x2 <= dst_x1
                or dst_y2 <= dst_y1
            ):
                continue

            # -----------------------------------------------------
            # Full-size RGB patch
            # -----------------------------------------------------

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
            # IMPORTANT DEBUG
            # -----------------------------------------------------

            print(
                f"[Soiling] patch {defect_idx + 1}/{num_defects}:"
                f" pos=({cx},{cy})"
                f" size=({rand_w},{rand_h})"
                f" alpha_before_mask="
                f"{full_alpha.max().item():.3f}"
            )

            # -----------------------------------------------------
            # Apply car mask
            # -----------------------------------------------------

            full_alpha = (
                full_alpha * car_mask
            )

            print(
                f"[Soiling] patch {defect_idx + 1}:"
                f" alpha_after_mask="
                f"{full_alpha.max().item():.3f}"
            )

            # -----------------------------------------------------
            # Expand patch RGB over batch
            # -----------------------------------------------------

            full_rgb = full_rgb.expand(
                b,
                -1,
                -1,
                -1,
            )

            # -----------------------------------------------------
            # Blend
            # -----------------------------------------------------

            soil_texture = (
                soil_texture * (1.0 - full_alpha)
                + full_rgb * full_alpha
            )

            # -----------------------------------------------------
            # Mask
            # -----------------------------------------------------

            soil_mask = torch.maximum(
                soil_mask,
                full_alpha,
            )

        # ---------------------------------------------------------
        # Final
        # ---------------------------------------------------------

        soil_texture = soil_texture.clamp(
            0.0,
            1.0,
        )

        soil_mask = soil_mask.clamp(
            0.0,
            1.0,
        )

        print(
            "[Soiling]"
            f" final_mask="
            f"({soil_mask.min().item():.3f},"
            f"{soil_mask.max().item():.3f})"
            f" final_image="
            f"({soil_texture.min().item():.3f},"
            f"{soil_texture.max().item():.3f})"
        )

        return soil_texture, soil_mask

    @staticmethod
    def _texture_to_tensor(
        item,
        device,
        dtype,
    ) -> torch.Tensor:

        # =========================================================
        # PIL
        # =========================================================

        if isinstance(item, Image.Image):

            item = item.convert("RGBA")

            # .copy() is important:
            # np.asarray(PIL) can return read-only memory.
            arr = np.asarray(item).copy()

            tex = torch.from_numpy(arr)

            tex = tex.permute(
                2,
                0,
                1,
            ).contiguous()

            tex = tex.float() / 255.0

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
                    f"Texture must be [C,H,W] or [B,C,H,W], "
                    f"got {tuple(tex.shape)}"
                )

            if tex.shape[0] > 1:
                tex = tex[:1]

            channels = tex.shape[1]

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
                    f"Texture must have 3 or 4 channels, "
                    f"got {channels}"
                )

            tex = tex.float()

            if (
                tex.numel() > 0
                and tex.max() > 1.0
            ):
                tex = tex / 255.0

        else:

            raise TypeError(
                f"Unsupported texture type: {type(item)}"
            )

        tex = tex.to(
            device=device,
            dtype=dtype,
        )

        return tex.clamp(0.0, 1.0)