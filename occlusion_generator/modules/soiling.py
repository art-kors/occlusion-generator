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
            print("[SOILING] MUD MODE")
            print("=" * 70)

        # =========================================================
        # 1. GET DIRT PATCH
        # =========================================================

        dirt = kwargs.get("dirt_buffer", None)

        if dirt is None:
            raise ValueError(
                "[SOILING] dirt_buffer is required"
            )

        if not torch.is_tensor(dirt):
            raise TypeError(
                f"[SOILING] dirt_buffer must be Tensor, "
                f"got {type(dirt)}"
            )

        dirt = dirt.to(
            device=device,
            dtype=dtype,
        )

        # =========================================================
        # 2. NORMALIZE DIMENSIONS
        # =========================================================

        if dirt.ndim == 2:

            # [H,W]
            dirt = dirt[None, None]

        elif dirt.ndim == 3:

            # [C,H,W]
            dirt = dirt[None]

        elif dirt.ndim != 4:

            raise ValueError(
                f"[SOILING] Invalid dirt shape: {dirt.shape}"
            )

        # =========================================================
        # 3. BATCH
        # =========================================================

        if dirt.shape[0] == 1 and b > 1:

            dirt = dirt.expand(
                b,
                -1,
                -1,
                -1,
            )

        elif dirt.shape[0] != b:

            raise ValueError(
                "[SOILING] Batch mismatch: "
                f"image={b}, dirt={dirt.shape[0]}"
            )

        # =========================================================
        # 4. CHANNELS
        # =========================================================

        dirt_channels = dirt.shape[1]

        if dirt_channels == 1:

            texture = dirt

        elif dirt_channels == 3:

            # Convert RGB patch to grayscale texture
            texture = dirt.mean(
                dim=1,
                keepdim=True,
            )

        else:

            raise ValueError(
                "[SOILING] Expected 1 or 3 channel dirt texture, "
                f"got {dirt_channels}"
            )

        # =========================================================
        # 5. NO RESIZE
        # =========================================================

        patch_h = texture.shape[2]
        patch_w = texture.shape[3]

        if patch_h > h or patch_w > w:

            raise ValueError(
                "[SOILING] Dirt patch is larger than image: "
                f"patch={patch_w}x{patch_h}, "
                f"image={w}x{h}"
            )

        if DEBUG:
            print(
                "[SOILING] image:",
                (w, h),
            )

            print(
                "[SOILING] patch:",
                (patch_w, patch_h),
            )

        # =========================================================
        # 6. PLACE TEXTURE ON FULL IMAGE CANVAS
        # =========================================================

        texture_canvas = torch.zeros(
            (b, 1, h, w),
            device=device,
            dtype=dtype,
        )

        x = 0
        y = 0

        texture_canvas[
            :,
            :,
            y:y + patch_h,
            x:x + patch_w,
        ] = texture

        # =========================================================
        # 7. NORMALIZE TEXTURE
        #
        # Equivalent to:
        #
        # dir_texture =
        #     (dir_texture - min) /
        #     (max - min)
        # =========================================================

        tex_min = texture_canvas.amin(
            dim=(-2, -1),
            keepdim=True,
        )

        tex_max = texture_canvas.amax(
            dim=(-2, -1),
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
        # 8. CREATE MUD MASK
        #
        # Old code:
        #
        # dir_mask = dir_texture.copy()
        # dir_mask[dir_mask > 0.5] = 1
        # GaussianBlur(sigma=6)
        #
        # Here we reproduce the idea with PyTorch.
        # =========================================================

        mask = torch.where(
            texture_norm > 0.5,
            torch.ones_like(texture_norm),
            texture_norm,
        )

        # =========================================================
        # 9. GAUSSIAN BLUR MASK
        # =========================================================

        # OpenCV sigmaX=6 equivalent-ish.
        #
        # Kernel size ~ 6*sigma + 1
        # -> 37
        #
        # Use odd kernel.
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
            -(
                coords ** 2
            ) / (
                2 * sigma ** 2
            )
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
        # 10. BROWN MUD COLOR
        #
        # Original:
        #
        # base_color = np.array([20,42,63])
        #
        # IMPORTANT:
        # OpenCV uses BGR.
        #
        # [20,42,63] BGR
        # =>
        # [63,42,20] RGB
        #
        # =========================================================

        base_color = torch.tensor(
            [63.0, 42.0, 20.0],
            device=device,
            dtype=dtype,
        )

        base_color = base_color / 255.0

        base_color = base_color.view(
            1, 3, 1, 1,
        )

        # =========================================================
        # 11. IMAGE MEAN
        #
        # Old code:
        #
        # mean_color = image.mean()
        # alpha = 0.1
        # mix_color =
        #     base_color*(1-alpha)
        #     + mean_color*alpha
        #
        # =========================================================

        mean_color = image.mean()

        color_mix_alpha = 0.1

        mud_color = (
            base_color
            * (1.0 - color_mix_alpha)
            +
            mean_color
            * color_mix_alpha
        )

        # =========================================================
        # 12. BLUR ORIGINAL IMAGE
        #
        # Original:
        #
        # image_blur = GaussianBlur(image, sigmaX=15)
        #
        # First create Gaussian kernel.
        # =========================================================

        sigma_img = 15.0

        kernel_size_img = 91

        coords = torch.arange(
            kernel_size_img,
            device=device,
            dtype=dtype,
        )

        coords = coords - (
            kernel_size_img - 1
        ) / 2

        kernel = torch.exp(
            -(
                coords ** 2
            ) / (
                2 * sigma_img ** 2
            )
        )

        kernel = kernel / kernel.sum()

        kernel_2d = (
            kernel[:, None]
            * kernel[None, :]
        )

        kernel_2d = kernel_2d[None, None]

        kernel_rgb = kernel_2d.expand(
            c,
            1,
            kernel_size_img,
            kernel_size_img,
        )

        image_blur = F.conv2d(
            image,
            kernel_rgb,
            padding=kernel_size_img // 2,
            groups=c,
        )

        # =========================================================
        # 13. FIRST BLENDING STAGE
        #
        # Original:
        #
        # image_blur =
        #     image*(1-mask)
        #     + image_blur*mask
        #
        # This makes the dirty area blurry.
        # =========================================================

        mask_rgb = mask.expand(
            -1,
            c,
            -1,
            -1,
        )

        blurred_image = (
            image
            * (1.0 - mask_rgb)
            +
            image_blur
            * mask_rgb
        )

        # =========================================================
        # 14. SECOND BLENDING STAGE
        #
        # Original:
        #
        # image_out =
        #     image_blur*(1-dir_mask)
        #     + mix_color*dir_mask
        #
        # =========================================================

        mud_color_full = mud_color.expand(
            b,
            -1,
            h,
            w,
        )

        result = (
            blurred_image
            * (1.0 - mask_rgb)
            +
            mud_color_full
            * mask_rgb
        )

        # =========================================================
        # 15. INTENSITY
        # =========================================================

        # Instead of making the mask itself weaker,
        # blend the complete mud result with original image.

        intensity = float(
            cfg.intensity
        )

        result = (
            image * (1.0 - intensity)
            +
            result * intensity
        )

        result = torch.clamp(
            result,
            0.0,
            1.0,
        )

        # =========================================================
        # 16. DEBUG
        # =========================================================

        if DEBUG:

            diff = (
                result - image
            ).abs()

            print(
                "[SOILING] texture range:",
                texture_norm.min().item(),
                texture_norm.max().item(),
            )

            print(
                "[SOILING] mask range:",
                mask.min().item(),
                mask.max().item(),
            )

            print(
                "[SOILING] mask mean:",
                mask.mean().item(),
            )

            print(
                "[SOILING] result difference:",
                diff.mean().item(),
            )

            print("=" * 70)
            print("[SOILING] END")
            print("=" * 70)

        return result, mask