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
    Camera/lens soiling augmentation.

    intensity
        Controls the amount of contamination:
        number and size of dirt patches.

    severity
        Controls the visual/optical strength of contamination:

            severity=0.0
                almost invisible contamination

            severity=0.25
                weak transparent dirt

            severity=0.5
                clearly visible dirt

            severity=0.75
                dense dirt with substantial opaque regions

            severity=1.0
                heavy opaque mud with large alpha=1 regions

    Texture RGB
        Provides local spatial structure:
        lumps, cracks, gradients and surface variation.

    Texture Alpha
        Provides the basic geometric shape of the dirt.

    Important:
        Texture alpha is treated as SHAPE, not as final physical opacity.

        Final opacity is synthesized separately from:
            - dirt shape
            - severity
            - opaque core
            - soft boundary

    Expected image:
        [B, 3, H, W], float, nominally in [0, 1]

    Expected dirt texture:
        PIL RGB/RGBA image or torch Tensor with shape
        [C, H, W] / [B, C, H, W]

    Returned:
        result:
            [B, 3, H, W]

        soil_mask:
            [B, 1, H, W]

            Actual optical opacity applied to the image.
    """

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor,
        cfg: SoilingConfig,
        dirt_buffer=None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self._is_disabled(cfg):
            return (
                image,
                torch.zeros_like(image[:, 0:1]),
            )

        dirt_buffer = self._resolve_dirt_buffer(
            dirt_buffer=dirt_buffer,
            kwargs=kwargs,
        )

        self._validate_image(image)

        batch_size, _, height, width = image.shape
        device = image.device
        dtype = image.dtype

        intensity = self._clamp01(float(cfg.intensity))
        severity = self._get_severity(cfg)

        result = image[:, :3].clamp(0.0, 1.0).clone()

        effective_car_mask = self._prepare_car_mask(
            car_mask=car_mask,
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )

        num_patches = self._get_num_patches(
            intensity=intensity,
            severity=severity,
        )

        soil_mask = torch.zeros(
            batch_size,
            1,
            height,
            width,
            device=device,
            dtype=dtype,
        )

        for _ in range(num_patches):

            patch = self._create_patch(
                dirt_buffer=dirt_buffer,
                image_height=height,
                image_width=width,
                intensity=intensity,
                severity=severity,
                device=device,
                dtype=dtype,
            )

            center_x, center_y = patch["center"]

            patch_h, patch_w = patch["texture_rgb"].shape[-2:]

            x1 = center_x - patch_w // 2
            y1 = center_y - patch_h // 2

            x2 = x1 + patch_w
            y2 = y1 + patch_h

            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)

            dst_x2 = min(width, x2)
            dst_y2 = min(height, y2)

            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)

            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)

            if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                continue

            # ----------------------------------------------------------
            # Place alpha / optical occlusion mask
            # ----------------------------------------------------------

            full_alpha = torch.zeros(
                batch_size,
                1,
                height,
                width,
                device=device,
                dtype=dtype,
            )

            full_alpha[
                :,
                :,
                dst_y1:dst_y2,
                dst_x1:dst_x2,
            ] = patch["alpha"][
                :,
                :,
                src_y1:src_y2,
                src_x1:src_x2,
            ]

            full_alpha = self._apply_car_mask(
                full_alpha,
                effective_car_mask,
            )

            # ----------------------------------------------------------
            # Place RGB structure
            # ----------------------------------------------------------

            full_texture_rgb = torch.zeros_like(result)

            full_texture_rgb[
                :,
                :,
                dst_y1:dst_y2,
                dst_x1:dst_x2,
            ] = patch["texture_rgb"][
                :,
                :,
                src_y1:src_y2,
                src_x1:src_x2,
            ]

            # ----------------------------------------------------------
            # Create actual dirt appearance
            # ----------------------------------------------------------

            dirty_image = self._create_dirty_image(
                image=result,
                texture_rgb=full_texture_rgb,
                severity=severity,
            )

            # ----------------------------------------------------------
            # Composite
            # ----------------------------------------------------------

            result = self._composite(
                image=result,
                dirty_image=dirty_image,
                alpha=full_alpha,
            )

            # ----------------------------------------------------------
            # Ground-truth optical opacity
            # ----------------------------------------------------------

            soil_mask = torch.maximum(
                soil_mask,
                full_alpha,
            )

        return (
            result.clamp(0.0, 1.0),
            soil_mask.clamp(0.0, 1.0),
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _is_disabled(cfg: SoilingConfig) -> bool:
        return (
            not cfg.enabled
            or float(cfg.intensity) <= 0.0
        )

    @staticmethod
    def _get_severity(cfg: SoilingConfig) -> float:
        severity = getattr(
            cfg,
            "severity",
            0.5,
        )

        return max(
            0.0,
            min(
                1.0,
                float(severity),
            ),
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    @staticmethod
    def _resolve_dirt_buffer(
        dirt_buffer,
        kwargs: dict,
    ):
        if dirt_buffer is None:
            dirt_buffer = kwargs.get(
                "dirt_textures"
            )

        if (
            dirt_buffer is None
            or len(dirt_buffer) == 0
        ):
            raise RuntimeError(
                "SoilingModule requires a non-empty dirt_buffer."
            )

        return dirt_buffer

    @staticmethod
    def _get_num_patches(
        intensity: float,
        severity: float,
    ) -> int:

        # Intensity remains the primary amount control.
        base_count = int(
            round(
                2.0
                + intensity * 8.0
            )
        )

        # Small severity contribution.
        severity_bonus = int(
            round(
                severity * 2.0
            )
        )

        return max(
            1,
            base_count + severity_bonus,
        )

    # ------------------------------------------------------------------
    # Image validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(
        image: torch.Tensor,
    ) -> None:

        if image.ndim != 4:
            raise ValueError(
                "SoilingModule expects image "
                "with shape [B, C, H, W], "
                f"got {tuple(image.shape)}"
            )

        if image.shape[1] < 3:
            raise ValueError(
                "SoilingModule expects at least "
                f"3 image channels, got {image.shape[1]}"
            )

    # ------------------------------------------------------------------
    # Car mask
    # ------------------------------------------------------------------

    def _prepare_car_mask(
        self,
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
                "car_mask must have shape "
                "[B, 1, H, W] or [B, H, W], "
                f"got {tuple(car_mask.shape)}"
            )

        if car_mask.shape[0] != batch_size:
            raise ValueError(
                "car_mask batch size does not "
                "match image: "
                f"{car_mask.shape[0]} != {batch_size}"
            )

        car_mask = car_mask[:, :1]

        if car_mask.shape[-2:] != (
            height,
            width,
        ):
            car_mask = F.interpolate(
                car_mask.float(),
                size=(
                    height,
                    width,
                ),
                mode="nearest",
            )

        car_mask = car_mask.to(
            device=device,
            dtype=dtype,
        ).clamp(
            0.0,
            1.0,
        )

        # Silent fallback for empty masks.
        if car_mask.max() <= 0:
            return torch.ones_like(
                car_mask
            )

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
        severity: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:

        texture = self._sample_texture(
            dirt_buffer=dirt_buffer,
            device=device,
            dtype=dtype,
        )

        patch_height, patch_width = (
            self._sample_patch_size(
                image_height=image_height,
                image_width=image_width,
                intensity=intensity,
                severity=severity,
                device=device,
            )
        )

        texture = self._resize_texture(
            texture=texture,
            height=patch_height,
            width=patch_width,
        )

        texture = self._rotate_texture(
            texture=texture,
            device=device,
        )

        texture_rgb = texture[:, :3]
        texture_alpha = texture[:, 3:4]

        # Build physical optical opacity from the texture shape.
        alpha = self._build_occlusion_alpha(
            texture_alpha=texture_alpha,
            severity=severity,
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
            "texture_rgb": texture_rgb,
        }

    # ------------------------------------------------------------------
    # Texture sampling
    # ------------------------------------------------------------------

    def _sample_texture(
        self,
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

        return self._texture_to_tensor(
            item=dirt_buffer[index],
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Patch geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_patch_size(
        image_height: int,
        image_width: int,
        intensity: float,
        severity: float,
        device: torch.device,
    ) -> tuple[int, int]:

        base_size = min(
            image_height,
            image_width,
        )

        min_scale = (
            0.035
            + 0.015 * severity
        )

        max_scale = min(
            0.70,
            0.25
            + 0.20 * intensity
            + 0.10 * severity,
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
            int(
                base_size * scale
            ),
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
            int(
                patch_height * aspect
            ),
        )

        return (
            min(
                patch_height,
                int(
                    image_height * 0.75
                ),
            ),
            min(
                patch_width,
                int(
                    image_width * 0.75
                ),
            ),
        )

    @staticmethod
    def _sample_patch_center(
        width: int,
        height: int,
        device: torch.device,
    ) -> tuple[int, int]:

        return (
            torch.randint(
                0,
                width,
                (1,),
                device=device,
            ).item(),
            torch.randint(
                0,
                height,
                (1,),
                device=device,
            ).item(),
        )

    @staticmethod
    def _resize_texture(
        texture: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:

        return F.interpolate(
            texture,
            size=(
                height,
                width,
            ),
            mode="bilinear",
            align_corners=False,
        )

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

        batch_size, _, height, width = (
            texture.shape
        )

        center = torch.tensor(
            [[
                width / 2.0,
                height / 2.0,
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
            batch_size,
            2,
            device=device,
            dtype=texture.dtype,
        )

        rotation_matrix = (
            kornia.geometry.transform
            .get_rotation_matrix2d(
                center=center,
                angle=angle_tensor,
                scale=scale,
            )
        )

        return (
            kornia.geometry.transform
            .warp_affine(
                texture,
                rotation_matrix,
                dsize=(
                    height,
                    width,
                ),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )

    # ------------------------------------------------------------------
    # Occlusion alpha
    # ------------------------------------------------------------------

    def _build_occlusion_alpha(
        self,
        texture_alpha: torch.Tensor,
        severity: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts texture alpha into actual optical occlusion.

        Important distinction:

            texture_alpha
                = shape information

            final alpha
                = actual optical opacity

        High severity creates an actual opaque core.

        At severity=1.0 the core is explicitly forced to alpha=1.
        """

        # --------------------------------------------------------------
        # 1. Normalize source alpha into a shape mask
        # --------------------------------------------------------------

        alpha = texture_alpha.clamp(
            0.0,
            1.0,
        )

        alpha_max = alpha.amax(
            dim=(-2, -1),
            keepdim=True,
        )

        shape = (
            alpha
            / (alpha_max + 1e-6)
        ).clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # 2. Light blur for natural boundaries
        # --------------------------------------------------------------
        #
        # Do NOT heavily blur the whole alpha.
        #
        # The previous implementation used large Gaussian kernels,
        # which turned dirt blobs into translucent clouds.

        image_size = min(
            shape.shape[-2],
            shape.shape[-1],
        )

        kernel = self._sample_odd_kernel(
            image_size=image_size,
            low=3,
            high=11,
            device=device,
        )

        sigma_min = (
            0.6
            - 0.2 * severity
        )

        sigma_max = (
            2.5
            - 1.0 * severity
        )

        sigma_max = max(
            sigma_min,
            sigma_max,
        )

        sigma = torch.empty(
            1,
            device=device,
        ).uniform_(
            sigma_min,
            sigma_max,
        ).item()

        soft_shape = (
            kornia.filters
            .gaussian_blur2d(
                shape,
                kernel_size=(
                    kernel,
                    kernel,
                ),
                sigma=(
                    sigma,
                    sigma,
                ),
            )
        )

        soft_shape = soft_shape.clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # 3. Severity determines opaque-core threshold
        # --------------------------------------------------------------
        #
        # Low severity:
        #   only the strongest central pixels become opaque.
        #
        # High severity:
        #   much larger part of the blob becomes opaque.
        #
        # At severity=1:
        #   threshold is very low -> almost entire shape becomes core.

        threshold = (
            0.92
            - 0.87 * severity
        )

        threshold = max(
            0.05,
            min(
                0.92,
                threshold,
            ),
        )

        opaque_core = (
            soft_shape >= threshold
        ).to(
            dtype=soft_shape.dtype
        )

        # --------------------------------------------------------------
        # 4. Slightly expand the opaque core at high severity
        # --------------------------------------------------------------

        if severity > 0.5:

            expansion_kernel = self._sample_odd_kernel(
                image_size=image_size,
                low=3,
                high=9,
                device=device,
            )

            expansion_sigma = (
                0.8
                + 1.5 * severity
            )

            expanded_core = (
                kornia.filters
                .gaussian_blur2d(
                    opaque_core,
                    kernel_size=(
                        expansion_kernel,
                        expansion_kernel,
                    ),
                    sigma=(
                        expansion_sigma,
                        expansion_sigma,
                    ),
                )
            )

            # Convert the blurred expansion back into a mostly hard
            # region. This creates larger contiguous opaque chunks.
            expansion_threshold = (
                0.35
                - 0.20 * severity
            )

            expanded_core = (
                expanded_core
                >= expansion_threshold
            ).to(
                dtype=soft_shape.dtype
            )

            opaque_core = torch.maximum(
                opaque_core,
                expanded_core,
            )

        # --------------------------------------------------------------
        # 5. Soft boundary
        # --------------------------------------------------------------

        boundary_alpha = soft_shape

        # Base opacity rises strongly with severity.
        #
        # This controls pixels that are not part of the hard opaque core.
        base_opacity = severity ** 1.35

        alpha = (
            boundary_alpha
            * base_opacity
        )

        # --------------------------------------------------------------
        # 6. Opaque core overrides everything
        # --------------------------------------------------------------

        alpha = torch.maximum(
            alpha,
            opaque_core,
        )

        # --------------------------------------------------------------
        # 7. Hard guarantee at severity=1
        # --------------------------------------------------------------
        #
        # If the user explicitly requests maximum severity, the
        # meaningful shape of the dirt must contain actual alpha=1.
        #
        # We don't make the entire canvas opaque; only the dirt shape.

        if severity >= 0.999:

            final_core_threshold = 0.15

            final_core = (
                soft_shape
                >= final_core_threshold
            ).to(
                dtype=soft_shape.dtype
            )

            alpha = torch.maximum(
                alpha,
                final_core,
            )

        return alpha.clamp(
            0.0,
            1.0
        )

    # ------------------------------------------------------------------
    # Dirty appearance
    # ------------------------------------------------------------------

    def _create_dirty_image(
        self,
        image: torch.Tensor,
        texture_rgb: torch.Tensor,
        severity: float,
    ) -> torch.Tensor:
        """
        Generates the visual appearance of dirt.

        Severity controls darkness.

        Texture RGB controls local structure but does not directly
        control opacity.
        """

        # --------------------------------------------------------------
        # 1. Extract texture structure
        # --------------------------------------------------------------

        structure = (
            texture_rgb[:, 0:1] * 0.299
            + texture_rgb[:, 1:2] * 0.587
            + texture_rgb[:, 2:3] * 0.114
        )

        structure = (
            structure
            + torch.rand_like(
                structure
            ) * 0.06
        ).clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # 2. Mud base color
        # --------------------------------------------------------------

        base_r = torch.empty(
            1,
            device=image.device,
        ).uniform_(
            0.10,
            0.22,
        ).item()

        base_g = torch.empty(
            1,
            device=image.device,
        ).uniform_(
            0.08,
            0.19,
        ).item()

        base_b = torch.empty(
            1,
            device=image.device,
        ).uniform_(
            0.05,
            0.16,
        ).item()

        base_color = torch.tensor(
            [[
                base_r,
                base_g,
                base_b,
            ]],
            device=image.device,
            dtype=image.dtype,
        ).view(
            1,
            3,
            1,
            1,
        )

        # --------------------------------------------------------------
        # 3. Severity controls darkness
        # --------------------------------------------------------------

        darkness = severity ** 1.15

        brightness = (
            1.0
            - 0.90 * darkness
        )

        brightness = max(
            0.08,
            brightness,
        )

        base_color = (
            base_color
            * brightness
        )

        # --------------------------------------------------------------
        # 4. Add surface structure
        # --------------------------------------------------------------
        #
        # Keep variation relatively subtle.
        #
        # Opacity is handled entirely by alpha.

        structure_factor = (
            0.55
            + 0.45 * structure
        )

        dirt_appearance = (
            base_color
            * structure_factor
        )

        return dirt_appearance.expand_as(
            image
        ).clamp(
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_car_mask(
        alpha: torch.Tensor,
        car_mask: torch.Tensor,
    ) -> torch.Tensor:

        return alpha * car_mask

    @staticmethod
    def _sample_odd_kernel(
        image_size: int,
        low: int,
        high: int,
        device: torch.device,
    ) -> int:

        max_allowed = max(
            3,
            min(
                high,
                image_size - 1,
            ),
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
            (
                max_allowed
                - min_allowed
            )
            // 2
        ) + 1

        index = torch.randint(
            0,
            count,
            (1,),
            device=device,
        ).item()

        return (
            min_allowed
            + index * 2
        )

    # ------------------------------------------------------------------
    # Texture conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _texture_to_tensor(
        item,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        if isinstance(
            item,
            Image.Image,
        ):

            item = item.convert(
                "RGBA"
            )

            array = (
                np.asarray(item)
                .copy()
            )

            texture = (
                torch.from_numpy(
                    array
                )
                .permute(
                    2,
                    0,
                    1,
                )
                .contiguous()
                .float()
                / 255.0
            )

            texture = texture.unsqueeze(
                0
            )

        elif torch.is_tensor(item):

            texture = item

            if texture.ndim == 3:
                texture = texture.unsqueeze(
                    0
                )

            if texture.ndim != 4:
                raise ValueError(
                    "Texture must have shape "
                    "[C,H,W] or [B,C,H,W], "
                    f"got {tuple(texture.shape)}"
                )

            # One texture per patch.
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
                    [
                        texture,
                        alpha,
                    ],
                    dim=1,
                )

            elif channels != 4:

                raise ValueError(
                    "Texture must contain "
                    "3 or 4 channels, "
                    f"got {channels}"
                )

            texture = texture.float()

            if (
                texture.numel() > 0
                and texture.max() > 1.0
            ):
                texture = (
                    texture
                    / 255.0
                )

        else:

            raise TypeError(
                "Unsupported dirt texture "
                f"type: {type(item)}"
            )

        return texture.to(
            device=device,
            dtype=dtype,
        ).clamp(
            0.0,
            1.0,
        )