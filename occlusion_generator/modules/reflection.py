import torch
import torch.nn.functional as F
import kornia
import kornia.filters as KF

from ..interfaces import BaseOcclusionModule
from ..config import ReflectionConfig


class ReflectionModule(BaseOcclusionModule):

    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor,
        cfg: ReflectionConfig,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not cfg.enabled or cfg.intensity == 0.0:
            return (
                image,
                torch.zeros_like(image[:, 0:1]),
            )

        b, _, h, w = image.shape
        device = image.device
        dtype = image.dtype

        # ======================================================
        # 1. REFLECTION TEXTURE
        # ======================================================

        ref_texture = kwargs.get(
            "reflection_texture"
        )

        if ref_texture is None:

            ref_texture = torch.rand(
                b,
                3,
                h,
                w,
                device=device,
                dtype=dtype,
            )

        else:

            ref_texture = ref_texture.to(
                device=device,
                dtype=dtype,
            )

            if ref_texture.ndim == 3:
                ref_texture = ref_texture.unsqueeze(0)

            ref_texture = kornia.geometry.transform.resize(
                ref_texture,
                (h, w),
                antialias=True,
            )

        # ======================================================
        # 2. BLUR
        # ======================================================

        ref_texture = KF.gaussian_blur2d(
            ref_texture,
            kernel_size=(5, 5),
            sigma=(1.0, 1.0),
        )

        # ======================================================
        # 3. COLOR TRANSFER
        # ======================================================

        bg_mean = image.mean(
            dim=(2, 3),
            keepdim=True,
        )

        bg_std = image.std(
            dim=(2, 3),
            keepdim=True,
        ) + 1e-6

        ref_mean = ref_texture.mean(
            dim=(2, 3),
            keepdim=True,
        )

        ref_std = ref_texture.std(
            dim=(2, 3),
            keepdim=True,
        ) + 1e-6

        ref_texture = (
            (ref_texture - ref_mean)
            * (bg_std / ref_std)
            + bg_mean
        )

        ref_texture = torch.clamp(
            ref_texture,
            0.0,
            1.0,
        )

        # ======================================================
        # 4. BARREL DISTORTION
        # ======================================================

        y, x = torch.meshgrid(
            torch.linspace(
                -1,
                1,
                h,
                device=device,
                dtype=dtype,
            ),
            torch.linspace(
                -1,
                1,
                w,
                device=device,
                dtype=dtype,
            ),
            indexing="ij",
        )

        r = torch.sqrt(
            x ** 2 + y ** 2
        )

        distortion_strength = (
            0.15 * cfg.intensity
        )

        r_distorted = (
            r
            * (
                1.0
                + distortion_strength * r ** 2
            )
        )

        scale = torch.where(
            r > 0,
            r_distorted / (r + 1e-6),
            torch.ones_like(r),
        )

        grid = torch.stack(
            [
                x * scale,
                y * scale,
            ],
            dim=-1,
        )

        grid = (
            grid
            .unsqueeze(0)
            .expand(b, -1, -1, -1)
        )

        ref_texture = F.grid_sample(
            ref_texture,
            grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="zeros",
        )

        # ======================================================
        # 5. ORGANIC REFLECTION MASK
        # ======================================================

        ref_mask = torch.zeros(
            b,
            1,
            h,
            w,
            device=device,
            dtype=dtype,
        )

        y_m, x_m = torch.meshgrid(
            torch.arange(
                h,
                device=device,
                dtype=dtype,
            ),
            torch.arange(
                w,
                device=device,
                dtype=dtype,
            ),
            indexing="ij",
        )

        for _ in range(4):

            cx = (
                torch.rand(
                    1,
                    device=device,
                ).item()
                * w
            )

            cy = (
                torch.rand(
                    1,
                    device=device,
                ).item()
                * h
            )

            radius = (
                torch.rand(
                    1,
                    device=device,
                ).item()
                * (w * 0.7)
                + (w * 0.3)
            )

            dist = torch.sqrt(
                (x_m - cx) ** 2
                + (y_m - cy) ** 2
            )

            blob = torch.clamp(
                1.0 - dist / radius,
                0.0,
                1.0,
            )

            ref_mask = torch.clamp(
                ref_mask + blob[None, None],
                0.0,
                1.0,
            )

        # Blur mask
        ref_mask = KF.gaussian_blur2d(
            ref_mask,
            kernel_size=(31, 31),
            sigma=(12.0, 12.0),
        )

        # ======================================================
        # 6. ONLY CAR
        # ======================================================

        ref_mask = ref_mask * car_mask

        # ======================================================
        # 7. SCREEN BLENDING
        # ======================================================

        screen_blended = (
            1.0
            - (1.0 - image)
            * (1.0 - ref_texture)
        )

        alpha = (
            ref_mask
            * cfg.intensity
        )

        final_image = (
            image * (1.0 - alpha)
            + screen_blended * alpha
        )

        return (
            final_image,
            ref_mask,
        )