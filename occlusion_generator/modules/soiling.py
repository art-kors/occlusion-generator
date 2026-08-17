import torch
import torch.nn.functional as F
import kornia

from PIL import Image
import numpy as np

from ..interfaces import BaseOcclusionModule
from ..config import SoilingConfig


class SoilingModule(BaseOcclusionModule):
    """
    Realistic camera soiling augmentation.

    The dirt texture is used primarily as an alpha/distribution map.
    The actual appearance of the dirty region is generated from the
    original image using blur, darkening and desaturation.
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

        if self._is_disabled(cfg):
            return image, torch.zeros_like(image[:, 0:1])

        dirt_buffer = self._get_dirt_buffer(
            dirt_buffer,
            kwargs,
        )

        self._validate_image(image)

        b, _, h, w = image.shape
        device = image.device
        dtype = image.dtype

        result = image[:, :3].clamp(0.0, 1.0).clone()

        effective_car_mask = self._prepare_car_mask(
            car_mask=car_mask,
            batch_size=b,
            height=h,
            width=w,
            device=device,
            dtype=dtype,
        )

        intensity = self._get_intensity(cfg)
        num_defects = self._get_num_defects(intensity)

        soil_mask = torch.zeros(
            b,
            1,
            h,
            w,
            device=device,
            dtype=dtype,
        )

        for _ in range(num_defects):

            patch = self._create_patch(
                dirt_buffer=dirt_buffer,
                image_height=h,
                image_width=w,
                intensity=intensity,
                device=device,
                dtype=dtype,
            )

            full_alpha = self._place_patch(
                patch_alpha=patch["alpha"],
                center=patch["center"],
                height=h,
                width=w,
                device=device,
                dtype=dtype,
            )

            full_alpha = self._apply_car_mask(
                alpha=full_alpha,
                car_mask=effective_car_mask,
            )

            dirty_image = self._create_dirty_image(
                image=result,
            )

            result = self._composite(
                image=result,
                dirty_image=dirty_image,
                alpha=full_alpha,
            )

            soil_mask = torch.maximum(
                soil_mask,
                full_alpha,
            )

        return (
            result.clamp(0.0, 1.0),
            soil_mask.clamp(0.0, 1.0),
        )

    # =============================================================
    # Configuration / validation
    # =============================================================

    @staticmethod
    def _is_disabled(cfg: SoilingConfig) -> bool:
        return (
            not cfg.enabled
            or cfg.intensity <= 0.0
        )

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:

        if image.ndim != 4:
            raise ValueError(
                "SoilingModule expects image "
                f"[B,C,H,W], got {tuple(image.shape)}"
            )

        if image.shape[1] < 3:
            raise ValueError(
                "SoilingModule expects at least "
                f"3 channels, got {image.shape[1]}"
            )

    @staticmethod
    def _get_dirt_buffer(
        dirt_buffer,
        kwargs,
    ):
        if dirt_buffer is None:
            dirt_buffer = kwargs.get("dirt_textures")

        if not dirt_buffer:
            raise RuntimeError(
                "SoilingModule requires a non-empty "
                "dirt_buffer."
            )

        return dirt_buffer

    @staticmethod
    def _get_intensity(cfg: SoilingConfig) -> float:
        return max(
            0.0,
            min(1.0, float(cfg.intensity)),
        )

    @staticmethod
    def _get_num_defects(intensity: float) -> int:
        return max(
            1,
            int(round(intensity * 8)),
        )

    # =============================================================
    # Car mask
    # =============================================================

    def _prepare_car_mask(
        self,
        car_mask: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        device,
        dtype,
    ) -> torch.Tensor:

        if car_mask is None:
            return torch.ones(
                batch_size,
                1,
                height,
                width,
                device=device,
                dtype=dtype,
            )

        if car_mask.ndim == 3:
            car_mask = car_mask.unsqueeze(1)

        if car_mask.ndim != 4:
            raise ValueError(
                "car_mask must be [B,1,H,W] "
                f"or [B,H,W], got {tuple(car_mask.shape)}"
            )

        if car_mask.shape[0] != batch_size:
            raise ValueError(
                "car_mask batch mismatch: "
                f"{car_mask.shape[0]} != {batch_size}"
            )

        car_mask = car_mask[:, :1]

        if car_mask.shape[-2:] != (height, width):
            car_mask = F.interpolate(
                car_mask.float(),
                size=(height, width),
                mode="nearest",
            )

        car_mask = car_mask.to(
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)

        # Temporary fallback while car_mask generation
        # is being fixed.
        if car_mask.max() <= 0:
            print(
                "[Soiling] WARNING: empty car_mask. "
                "Using full-frame mask."
            )

            return torch.ones_like(car_mask)

        return car_mask

    # =============================================================
    # Patch generation
    # =============================================================

    def _create_patch(
        self,
        dirt_buffer,
        image_height: int,
        image_width: int,
        intensity: float,
        device,
        dtype,
    ) -> dict:

        texture = self._sample_texture(
            dirt_buffer=dirt_buffer,
            device=device,
            dtype=dtype,
        )

        patch_h, patch_w = self._sample_patch_size(
            image_height=image_height,
            image_width=image_width,
            device=device,
        )

        texture = self._resize_texture(
            texture,
            height=patch_h,
            width=patch_w,
        )

        texture = self._rotate_texture(
            texture,
            device=device,
        )

        alpha = texture[:, 3:4]

        alpha = self._soften_alpha(
            alpha,
            device=device,
        )

        alpha = self._apply_random_opacity(
            alpha,
            intensity=intensity,
            device=device,
        )

        center = self._sample_patch_center(
            width=image_width,
            height=image_height,
            device=device,
        )

        return {
            "alpha": alpha,
            "center": center,
        }

    def _sample_texture(
        self,
        dirt_buffer,
        device,
        dtype,
    ) -> torch.Tensor:

        idx = torch.randint(
            0,
            len(dirt_buffer),
            (1,),
            device=device,
        ).item()

        return self._texture_to_tensor(
            dirt_buffer[idx],
            device=device,
            dtype=dtype,
        )

    # =============================================================
    # Patch geometry
    # =============================================================

    @staticmethod
    def _sample_patch_size(
        image_height: int,
        image_width: int,
        device,
    ) -> tuple[int, int]:

        base_size = min(
            image_height,
            image_width,
        )

        scale = torch.empty(
            1,
            device=device,
        ).uniform_(0.05, 0.35).item()

        patch_h = max(
            8,
            int(base_size * scale),
        )

        aspect = torch.empty(
            1,
            device=device,
        ).uniform_(0.55, 1.8).item()

        patch_w = max(
            8,
            int(patch_h * aspect),
        )

        patch_h = min(
            patch_h,
            int(image_height * 0.7),
        )

        patch_w = min(
            patch_w,
            int(image_width * 0.7),
        )

        return patch_h, patch_w

    @staticmethod
    def _sample_patch_center(
        width: int,
        height: int,
        device,
    ) -> tuple[int, int]:

        cx = torch.randint(
            0,
            width,
            (1,),
            device=device,
        ).item()

        cy = torch.randint(
            0,
            height,
            (1,),
            device=device,
        ).item()

        return cx, cy

    @staticmethod
    def _resize_texture(
        texture: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:

        return F.interpolate(
            texture,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _rotate_texture(
        texture: torch.Tensor,
        device,
    ) -> torch.Tensor:

        angle = torch.empty(
            1,
            device=device,
        ).uniform_(-35.0, 35.0).item()

        if abs(angle) < 0.1:
            return texture

        _, _, h, w = texture.shape

        center = torch.tensor(
            [[
                w / 2.0,
                h / 2.0,
            ]],
            device=device,
            dtype=texture.dtype,
        )

        angle_tensor = torch.tensor(
            [angle],
            device=device,
            dtype=texture.dtype,
        )

        scale = torch.ones(
            1,
            device=device,
            dtype=texture.dtype,
        )

        matrix = kornia.geometry.transform.get_rotation_matrix2d(
            center=center,
            angle=angle_tensor,
            scale=scale,
        )

        return kornia.geometry.transform.warp_affine(
            texture,
            matrix,
            dsize=(h, w),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    # =============================================================
    # Alpha processing
    # =============================================================

    def _soften_alpha(
        self,
        alpha: torch.Tensor,
        device,
    ) -> torch.Tensor:

        kernel = self._sample_odd_kernel(
            image_size=min(
                alpha.shape[-2],
                alpha.shape[-1],
            ),
            low=9,
            high=31,
            device=device,
        )

        sigma = torch.empty(
            1,
            device=device,
        ).uniform_(2.0, 8.0).item()

        alpha = kornia.filters.gaussian_blur2d(
            alpha,
            kernel_size=(kernel, kernel),
            sigma=(sigma, sigma),
        )

        # Restore useful alpha range after blur.
        alpha_max = alpha.amax(
            dim=(-2, -1),
            keepdim=True,
        )

        alpha = alpha / (
            alpha_max + 1e-6
        )

        return alpha.clamp(0.0, 1.0)

    @staticmethod
    def _apply_random_opacity(
        alpha: torch.Tensor,
        intensity: float,
        device,
    ) -> torch.Tensor:

        min_opacity = (
            0.08 + 0.10 * intensity
        )

        max_opacity = (
            0.35 + 0.45 * intensity
        )

        opacity = torch.empty(
            1,
            device=device,
        ).uniform_(
            min_opacity,
            max_opacity,
        ).item()

        return (
            alpha * opacity
        ).clamp(0.0, 1.0)

    @staticmethod
    def _apply_car_mask(
        alpha: torch.Tensor,
        car_mask: torch.Tensor,
    ) -> torch.Tensor:

        return alpha * car_mask

    # =============================================================
    # Patch placement
    # =============================================================

    @staticmethod
    def _place_patch(
        patch_alpha: torch.Tensor,
        center: tuple[int, int],
        height: int,
        width: int,
        device,
        dtype,
    ) -> torch.Tensor:

        cx, cy = center

        patch_h = patch_alpha.shape[-2]
        patch_w = patch_alpha.shape[-1]

        x1 = cx - patch_w // 2
        y1 = cy - patch_h // 2

        x2 = x1 + patch_w
        y2 = y1 + patch_h

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(width, x2)
        dst_y2 = min(height, y2)

        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)

        src_x2 = src_x1 + (
            dst_x2 - dst_x1
        )

        src_y2 = src_y1 + (
            dst_y2 - dst_y1
        )

        full_alpha = torch.zeros(
            1,
            1,
            height,
            width,
            device=device,
            dtype=dtype,
        )

        if (
            dst_x2 <= dst_x1
            or dst_y2 <= dst_y1
        ):
            return full_alpha

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

        return full_alpha

    # =============================================================
    # Dirty appearance
    # =============================================================

    def _create_dirty_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:

        blurred = self._blur_image(
            image
        )

        darkened = self._darken(
            blurred
        )

        desaturated = self._desaturate(
            blurred
        )

        dirty = (
            0.55 * darkened
            + 0.45 * desaturated
        )

        dirty = self._apply_tint(
            dirty
        )

        return dirty.clamp(
            0.0,
            1.0,
        )

    @staticmethod
    def _blur_image(
        image: torch.Tensor,
    ) -> torch.Tensor:

        device = image.device

        kernel = SoilingModule._sample_odd_kernel(
            image_size=min(
                image.shape[-2],
                image.shape[-1],
            ),
            low=9,
            high=25,
            device=device,
        )

        sigma = torch.empty(
            1,
            device=device,
        ).uniform_(2.0, 6.0).item()

        return kornia.filters.gaussian_blur2d(
            image,
            kernel_size=(kernel, kernel),
            sigma=(sigma, sigma),
        )

    @staticmethod
    def _darken(
        image: torch.Tensor,
    ) -> torch.Tensor:

        strength = torch.empty(
            1,
            device=image.device,
        ).uniform_(0.05, 0.30).item()

        return image * (
            1.0 - strength
        )

    @staticmethod
    def _desaturate(
        image: torch.Tensor,
    ) -> torch.Tensor:

        gray = (
            image[:, 0:1] * 0.299
            + image[:, 1:2] * 0.587
            + image[:, 2:3] * 0.114
        )

        strength = torch.empty(
            1,
            device=image.device,
        ).uniform_(0.05, 0.35).item()

        return (
            image * (1.0 - strength)
            + gray * strength
        )

    @staticmethod
    def _apply_tint(
        image: torch.Tensor,
    ) -> torch.Tensor:

        tint = torch.tensor(
            [0.92, 0.90, 0.87],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)

        return image * tint

    # =============================================================
    # Compositing
    # =============================================================

    @staticmethod
    def _composite(
        image: torch.Tensor,
        dirty_image: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:

        return (
            image * (1.0 - alpha)
            + dirty_image * alpha
        )

    # =============================================================
    # Utility
    # =============================================================

    @staticmethod
    def _sample_odd_kernel(
        image_size: int,
        low: int,
        high: int,
        device,
    ) -> int:

        max_allowed = min(
            high,
            image_size - 1,
        )

        max_allowed = max(
            3,
            max_allowed,
        )

        if max_allowed % 2 == 0:
            max_allowed -= 1

        min_allowed = min(
            low,
            max_allowed,
        )

        if min_allowed % 2 == 1:
            pass
        else:
            min_allowed += 1

        if min_allowed > max_allowed:
            return max_allowed

        count = (
            (max_allowed - min_allowed) // 2
        ) + 1

        idx = torch.randint(
            0,
            count,
            (1,),
            device=device,
        ).item()

        return (
            min_allowed
            + idx * 2
        )

    # =============================================================
    # Texture conversion
    # =============================================================

    @staticmethod
    def _texture_to_tensor(
        item,
        device,
        dtype,
    ) -> torch.Tensor:

        if isinstance(item, Image.Image):

            item = item.convert("RGBA")

            # copy() prevents the non-writable NumPy warning.
            arr = np.asarray(item).copy()

            texture = torch.from_numpy(
                arr
            ).permute(
                2,
                0,
                1,
            ).contiguous()

            texture = (
                texture.float() / 255.0
            )

            texture = texture.unsqueeze(0)

        elif torch.is_tensor(item):

            texture = item

            if texture.ndim == 3:
                texture = texture.unsqueeze(0)

            if texture.ndim != 4:
                raise ValueError(
                    "Texture must have shape "
                    "[C,H,W] or [B,C,H,W], "
                    f"got {tuple(texture.shape)}"
                )

            texture = texture[:1]

            channels = texture.shape[1]

            if channels == 3:

                alpha = torch.ones(
                    texture.shape[0],
                    1,
                    texture.shape[2],
                    texture.shape[3],
                    device=texture.device,
                    dtype=texture.dtype,
                )

                texture = torch.cat(
                    [texture, alpha],
                    dim=1,
                )

            elif channels != 4:

                raise ValueError(
                    "Texture must contain "
                    f"3 or 4 channels, got {channels}"
                )

            texture = texture.float()

            if texture.max() > 1.0:
                texture = texture / 255.0

        else:

            raise TypeError(
                "Unsupported texture type: "
                f"{type(item)}"
            )

        return texture.to(
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)