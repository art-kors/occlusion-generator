from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import ReflectionConfig
from ..interfaces import BaseOcclusionModule


class ReflectionModule(BaseOcclusionModule):

    @torch.no_grad()
    def apply(
        self,
        image: torch.Tensor,
        depth,
        car_mask,
        cfg: ReflectionConfig,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # =========================================================
        # 0. CONFIG
        # =========================================================

        if not cfg.enabled or cfg.intensity <= 0:
            return (
                image,
                torch.zeros_like(image[:, 0:1]),
            )

        DEBUG = True

        b, c, h, w = image.shape
        device = image.device
        dtype = image.dtype

        if DEBUG:
            print("\n" + "=" * 70)
            print("[REFLECTION] REFLECTION MODE")
            print("=" * 70)

        # =========================================================
        # 1. GET REFLECTION TEXTURE
        # =========================================================

        reflection = kwargs.get(
            "reflection_texture",
            None,
        )

        if reflection is None:
            raise ValueError(
                "[REFLECTION] reflection_texture is required"
            )

        if not torch.is_tensor(reflection):
            raise TypeError(
                "[REFLECTION] reflection_texture must be Tensor, "
                f"got {type(reflection)}"
            )

        reflection = reflection.to(
            device=device,
            dtype=dtype,
        )

        # =========================================================
        # 2. NORMALIZE DIMENSIONS
        # =========================================================

        if reflection.ndim == 2:

            # [H,W]
            reflection = reflection[None, None]

        elif reflection.ndim == 3:

            # [C,H,W]
            reflection = reflection[None]

        elif reflection.ndim != 4:

            raise ValueError(
                "[REFLECTION] Invalid reflection texture shape: "
                f"{reflection.shape}"
            )

        # =========================================================
        # 3. BATCH
        # =========================================================

        if reflection.shape[0] == 1 and b > 1:

            reflection = reflection.expand(
                b,
                -1,
                -1,
                -1,
            )

        elif reflection.shape[0] != b:

            raise ValueError(
                "[REFLECTION] Batch mismatch: "
                f"image={b}, "
                f"reflection={reflection.shape[0]}"
            )

        # =========================================================
        # 4. CHANNELS
        # =========================================================

        reflection_channels = reflection.shape[1]

        if reflection_channels == 1:

            # grayscale -> RGB
            reflection = reflection.expand(
                -1,
                3,
                -1,
                -1,
            )

        elif reflection_channels == 3:

            pass

        else:

            raise ValueError(
                "[REFLECTION] Expected 1 or 3 channels, "
                f"got {reflection_channels}"
            )

        # =========================================================
        # 5. NO RESIZE
        # =========================================================

        patch_h = reflection.shape[2]
        patch_w = reflection.shape[3]

        if patch_h > h or patch_w > w:

            raise ValueError(
                "[REFLECTION] Reflection texture is larger "
                "than image: "
                f"texture={patch_w}x{patch_h}, "
                f"image={w}x{h}"
            )

        if DEBUG:

            print(
                "[REFLECTION] image:",
                (w, h),
            )

            print(
                "[REFLECTION] texture:",
                (patch_w, patch_h),
            )

        # =========================================================
        # 6. CREATE FULL CANVAS
        # =========================================================

        texture_canvas = torch.zeros(
            (b, 3, h, w),
            device=device,
            dtype=dtype,
        )

        # Same philosophy as SoilingModule:
        # put the texture at (0, 0)

        x = 0
        y = 0

        texture_canvas[
            :,
            :,
            y:y + patch_h,
            x:x + patch_w,
        ] = reflection

        # =========================================================
        # 7. NORMALIZE TEXTURE
        # =========================================================

        tex_min = texture_canvas.amin(
            dim=(2, 3),
            keepdim=True,
        )

        tex_max = texture_canvas.amax(
            dim=(2, 3),
            keepdim=True,
        )

        texture_norm = (
            texture_canvas - tex_min
        ) / (
            tex_max - tex_min + 1e-8
        )

        texture_norm = torch.clamp(
            texture_norm,
            0.0,
            1.0,
        )

        # =========================================================
        # 8. CREATE REFLECTION MASK
        # =========================================================
        #
        # Reflection mask is derived from brightness
        # of the supplied reflection texture.
        #
        # Bright regions -> stronger reflection.
        #
        # =========================================================

        gray = texture_norm.mean(
            dim=1,
            keepdim=True,
        )

        mask = torch.where(
            gray > 0.5,
            torch.ones_like(gray),
            gray,
        )

        # =========================================================
        # 9. BLUR MASK
        # =========================================================

        sigma = 6.0
        kernel_size = 37

        coords = torch.arange(
            kernel_size,
            device=device,
            dtype=dtype,
        )

        coords = coords - (
            kernel_size - 1
        ) / 2

        kernel = torch.exp(
            -coords ** 2
            / (2 * sigma ** 2)
        )

        kernel = kernel / kernel.sum()

        kernel_2d = (
            kernel[:, None]
            * kernel[None, :]
        )

        kernel_2d = kernel_2d[None, None]

        mask = F.conv2d(
            mask,
            kernel_2d,
            padding=kernel_size // 2,
        )

        mask = torch.clamp(
            mask,
            0.0,
            1.0,
        )

        # =========================================================
        # 10. OPTIONAL REFLECTION BLUR
        # =========================================================

        sigma_reflection = 3.0
        kernel_size_reflection = 19

        coords = torch.arange(
            kernel_size_reflection,
            device=device,
            dtype=dtype,
        )

        coords = coords - (
            kernel_size_reflection - 1
        ) / 2

        kernel = torch.exp(
            -coords ** 2
            / (2 * sigma_reflection ** 2)
        )

        kernel = kernel / kernel.sum()

        kernel_2d = (
            kernel[:, None]
            * kernel[None, :]
        )

        kernel_2d = kernel_2d[None, None]

        kernel_rgb = kernel_2d.expand(
            3,
            1,
            kernel_size_reflection,
            kernel_size_reflection,
        )

        reflection_blur = F.conv2d(
            texture_canvas,
            kernel_rgb,
            padding=kernel_size_reflection // 2,
            groups=3,
        )

        # =========================================================
        # 11. SCREEN BLENDING
        # =========================================================

        screen = (
            1.0
            - (1.0 - image)
            * (1.0 - reflection_blur)
        )

        # =========================================================
        # 12. BLEND REFLECTION
        # =========================================================

        mask_rgb = mask.expand(
            -1,
            c,
            -1,
            -1,
        )

        reflection_result = (
            image
            * (1.0 - mask_rgb)
            +
            screen
            * mask_rgb
        )

        # =========================================================
        # 13. INTENSITY
        # =========================================================

        intensity = float(
            cfg.intensity
        )

        result = (
            image
            * (1.0 - intensity)
            +
            reflection_result
            * intensity
        )

        result = torch.clamp(
            result,
            0.0,
            1.0,
        )

        # =========================================================
        # 14. DEBUG
        # =========================================================

        if DEBUG:

            diff = (
                result - image
            ).abs()

            print(
                "[REFLECTION] texture range:",
                texture_norm.min().item(),
                texture_norm.max().item(),
            )

            print(
                "[REFLECTION] mask range:",
                mask.min().item(),
                mask.max().item(),
            )

            print(
                "[REFLECTION] mask mean:",
                mask.mean().item(),
            )

            print(
                "[REFLECTION] result difference:",
                diff.mean().item(),
            )

            print("=" * 70)
            print("[REFLECTION] END")
            print("=" * 70)

        return result, mask