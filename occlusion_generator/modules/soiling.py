from __future__ import annotations

from typing import Any

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


class SoilingModule(BaseOcclusionModule):
    """
    Generates a dirt mask only.

    The input image is NEVER modified.

    intensity controls:
        - number of dirt patches
        - patch size

    Dirt texture alpha controls:
        - shape
        - local density
        - transparency

    Returned:
        image:
            unchanged input image

        soil_mask:
            [B, 1, H, W]
            dirt mask in [0, 1]
    """

    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor | None,
        cfg: SoilingConfig,
        dirt_buffer=None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self._validate_image(image)

        if not cfg.enabled or float(cfg.intensity) <= 0.0:
            return image, torch.zeros_like(image[:, :1])

        dirt_buffer = self._resolve_dirt_buffer(
            dirt_buffer=dirt_buffer,
            kwargs=kwargs,
        )

        batch_size, _, height, width = image.shape
        device = image.device
        dtype = image.dtype

        intensity = self._clamp01(float(cfg.intensity))

        car_mask = self._prepare_car_mask(
            car_mask=car_mask,
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )

        soil_mask = torch.zeros(
            batch_size,
            1,
            height,
            width,
            device=device,
            dtype=dtype,
        )

        num_patches = self._get_num_patches(intensity)

        for _ in range(num_patches):
            patch = self._create_patch(
                dirt_buffer=dirt_buffer,
                image_height=height,
                image_width=width,
                intensity=intensity,
                device=device,
                dtype=dtype,
            )

            self._place_patch(
                soil_mask=soil_mask,
                patch=patch,
                image_height=height,
                image_width=width,
            )

        # Dirt is allowed only inside the car mask.
        soil_mask.mul_(car_mask)

        # IMPORTANT:
        # The original image is returned untouched.
        return image, soil_mask.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _resolve_dirt_buffer(dirt_buffer, kwargs: dict):
        if dirt_buffer is None:
            dirt_buffer = kwargs.get("dirt_textures")

        if dirt_buffer is None or len(dirt_buffer) == 0:
            raise RuntimeError(
                "SoilingModule requires a non-empty dirt_buffer."
            )

        return dirt_buffer

    @staticmethod
    def _get_num_patches(intensity: float) -> int:
        """
        intensity=0 -> 1 patch
        intensity=1 -> 10 patches
        """

        return max(
            1,
            int(round(1.0 + 9.0 * intensity)),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if image.ndim != 4:
            raise ValueError(
                "SoilingModule expects image [B,C,H,W], "
                f"got {tuple(image.shape)}"
            )

        if image.shape[1] < 3:
            raise ValueError(
                "SoilingModule expects at least 3 channels, "
                f"got {image.shape[1]}"
            )

    # ------------------------------------------------------------------
    # Car mask
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_car_mask(
        car_mask: torch.Tensor | None,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
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
                "car_mask must be [B,H,W] or [B,1,H,W], "
                f"got {tuple(car_mask.shape)}"
            )

        if car_mask.shape[0] != batch_size:
            raise ValueError(
                "car_mask batch size does not match image: "
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

        # Empty mask means no restriction.
        if car_mask.max() <= 0:
            return torch.ones_like(car_mask)

        return car_mask

    # ------------------------------------------------------------------
    # Patch creation
    # ------------------------------------------------------------------

    def _create_patch(
        self,
        dirt_buffer,
        image_height: int,
        image_width: int,
        intensity: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:

        texture = self._sample_texture(
            dirt_buffer=dirt_buffer,
            device=device,
            dtype=dtype,
        )

        patch_height, patch_width = self._sample_patch_size(
            image_height=image_height,
            image_width=image_width,
            intensity=intensity,
            device=device,
        )

        texture = F.interpolate(
            texture,
            size=(patch_height, patch_width),
            mode="bilinear",
            align_corners=False,
        )

        texture = self._rotate_texture(
            texture,
            device=device,
        )

        alpha = texture[:, 3:4]

        # Only soften the mask.
        #
        # NO normalization.
        # NO severity.
        # NO opacity multiplier.
        alpha = self._process_alpha(
            alpha,
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

    # ------------------------------------------------------------------
    # Texture sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_texture(
        dirt_buffer,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        index = torch.randint(
            0,
            len(dirt_buffer),
            (1,),
            device=device,
        ).item()

        return SoilingModule._texture_to_tensor(
            dirt_buffer[index],
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Patch size
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_patch_size(
        image_height: int,
        image_width: int,
        intensity: float,
        device: torch.device,
    ) -> tuple[int, int]:

        base_size = min(
            image_height,
            image_width,
        )

        # intensity controls how large patches can become.
        min_scale = 0.025
        max_scale = 0.12 + 0.48 * intensity

        max_scale = min(
            max_scale,
            0.75,
        )

        scale = torch.empty(
            1,
            device=device,
        ).uniform_(
            min_scale,
            max_scale,
        ).item()

        patch_height = max(
            8,
            int(base_size * scale),
        )

        aspect = torch.empty(
            1,
            device=device,
        ).uniform_(
            0.45,
            2.2,
        ).item()

        patch_width = max(
            8,
            int(patch_height * aspect),
        )

        patch_height = min(
            patch_height,
            int(image_height * 0.8),
        )

        patch_width = min(
            patch_width,
            int(image_width * 0.8),
        )

        return patch_height, patch_width

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_patch_center(
        width: int,
        height: int,
        device: torch.device,
    ) -> tuple[int, int]:

        x = torch.randint(
            0,
            width,
            (1,),
            device=device,
        ).item()

        y = torch.randint(
            0,
            height,
            (1,),
            device=device,
        ).item()

        return x, y

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_texture(
        texture: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:

        angle = torch.empty(
            1,
            device=device,
        ).uniform_(
            -35.0,
            35.0,
        ).item()

        if abs(angle) < 0.1:
            return texture

        _, _, height, width = texture.shape

        center = torch.tensor(
            [[width / 2.0, height / 2.0]],
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
            2,
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
            dsize=(height, width),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    # ------------------------------------------------------------------
    # Alpha
    # ------------------------------------------------------------------

    @staticmethod
    def _process_alpha(
        alpha: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:

        alpha = alpha.clamp(0.0, 1.0)

        image_size = min(
            alpha.shape[-2],
            alpha.shape[-1],
        )

        kernel = SoilingModule._sample_odd_kernel(
            image_size=image_size,
            low=3,
            high=11,
            device=device,
        )

        alpha = kornia.filters.gaussian_blur2d(
            alpha,
            kernel_size=(kernel, kernel),
            sigma=(1.0, 1.0),
        )

        return alpha.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    @staticmethod
    def _place_patch(
        soil_mask: torch.Tensor,
        patch: dict[str, Any],
        image_height: int,
        image_width: int,
    ) -> None:

        alpha = patch["alpha"]

        center_x, center_y = patch["center"]

        patch_height = alpha.shape[-2]
        patch_width = alpha.shape[-1]

        x1 = center_x - patch_width // 2
        y1 = center_y - patch_height // 2

        x2 = x1 + patch_width
        y2 = y1 + patch_height

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(image_width, x2)
        dst_y2 = min(image_height, y2)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return

        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)

        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        patch_alpha = alpha[
            :,
            :,
            src_y1:src_y2,
            src_x1:src_x2,
        ]

        existing = soil_mask[
            :,
            :,
            dst_y1:dst_y2,
            dst_x1:dst_x2,
        ]

        # Alpha compositing of masks.
        #
        # First patch:
        #   mask = alpha
        #
        # Overlap:
        #   mask = old + new * (1-old)
        #
        # This allows overlapping dirt to become denser.
        combined = (
            existing
            + patch_alpha * (1.0 - existing)
        )

        soil_mask[
            :,
            :,
            dst_y1:dst_y2,
            dst_x1:dst_x2,
        ] = combined.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_odd_kernel(
        image_size: int,
        low: int,
        high: int,
        device: torch.device,
    ) -> int:

        if image_size < 3:
            return 3

        max_allowed = min(
            high,
            image_size - 1,
        )

        if max_allowed % 2 == 0:
            max_allowed -= 1

        min_allowed = min(
            low,
            max_allowed,
        )

        if min_allowed % 2 == 0:
            min_allowed += 1

        if min_allowed > max_allowed:
            return max_allowed

        count = (
            (max_allowed - min_allowed) // 2
        ) + 1

        index = torch.randint(
            0,
            count,
            (1,),
            device=device,
        ).item()

        return min_allowed + index * 2

    # ------------------------------------------------------------------
    # Texture conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _texture_to_tensor(
        item,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        if isinstance(item, Image.Image):

            # Always convert RGB → RGBA.
            #
            # RGB texture therefore becomes:
            #   alpha = 1 everywhere
            item = item.convert("RGBA")

            array = np.asarray(item).copy()

            texture = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .contiguous()
                .float()
                / 255.0
            )

            texture = texture.unsqueeze(0)

        elif torch.is_tensor(item):

            texture = item

            if texture.ndim == 3:
                texture = texture.unsqueeze(0)

            if texture.ndim != 4:
                raise ValueError(
                    "Texture must be [C,H,W] or [B,C,H,W], "
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
                    "Texture must contain 3 or 4 channels, "
                    f"got {channels}"
                )

            texture = texture.float()

            if texture.numel() > 0 and texture.max() > 1.0:
                texture = texture / 255.0

        else:
            raise TypeError(
                f"Unsupported dirt texture type: {type(item)}"
            )

        return texture.to(
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)