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

    The module separates two concepts:

        intensity
            Controls the amount of contamination:
            number and size of dirt patches.

        severity
            Controls the visual strength of contamination:
            opacity, blur, darkening, desaturation and opaque-mud
            probability.

    Dirt textures are used primarily as spatial alpha maps. Their
    RGB values are intentionally not pasted directly onto the
    source image. This avoids the "sticker" appearance and makes
    the generated degradation behave more like optical soiling.

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

        intensity = self._clamp01(
            float(cfg.intensity)
        )

        severity = self._get_severity(cfg)

        result = image[:, :3].clamp(
            0.0,
            1.0,
        ).clone()

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

            full_alpha = self._place_patch(
                patch_alpha=patch["alpha"],
                center=patch["center"],
                height=height,
                width=width,
                device=device,
                dtype=dtype,
            )

            full_alpha = self._apply_car_mask(
                alpha=full_alpha,
                car_mask=effective_car_mask,
            )

            dirty_image = self._create_dirty_image(
                image=result,
                severity=severity,
                patch_strength=patch["strength"],
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

        result = result.clamp(
            0.0,
            1.0,
        )

        soil_mask = soil_mask.clamp(
            0.0,
            1.0,
        )

        return result, soil_mask

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _is_disabled(
        cfg: SoilingConfig,
    ) -> bool:

        return (
            not cfg.enabled
            or float(cfg.intensity) <= 0.0
        )

    @staticmethod
    def _get_severity(
        cfg: SoilingConfig,
    ) -> float:
        """
        Read severity from SoilingConfig.

        The getattr() fallback keeps the module backwards-compatible
        with older SoilingConfig definitions that do not yet contain
        a severity field.
        """

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
    def _clamp01(
        value: float,
    ) -> float:

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

        if dirt_buffer is None or len(dirt_buffer) == 0:
            raise RuntimeError(
                "SoilingModule requires a non-empty "
                "dirt_buffer."
            )

        return dirt_buffer

    @staticmethod
    def _get_num_patches(
        intensity: float,
        severity: float,
    ) -> int:
        """
        Intensity primarily controls the amount of contamination.

        Severity has only a small influence on patch count. This is
        intentional: severity should not simply mean "more patches".
        """

        base_count = int(
            round(
                2.0
                + intensity * 8.0
            )
        )

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
                "SoilingModule expects image with "
                "shape [B, C, H, W], got "
                f"{tuple(image.shape)}"
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
                "[B, 1, H, W] or [B, H, W], got "
                f"{tuple(car_mask.shape)}"
            )

        if car_mask.shape[0] != batch_size:
            raise ValueError(
                "car_mask batch size does not match image: "
                f"{car_mask.shape[0]} != {batch_size}"
            )

        car_mask = car_mask[:, :1]

        if car_mask.shape[-2:] != (
            height,
            width,
        ):
            car_mask = F.interpolate(
                car_mask.float(),
                size=(height, width),
                mode="nearest",
            )

        car_mask = car_mask.to(
            device=device,
            dtype=dtype,
        ).clamp(
            0.0,
            1.0,
        )

        # Temporary compatibility fallback.
        #
        # If upstream mask generation is currently broken and
        # produces an empty mask, do not silently disable soiling.
        if car_mask.max() <= 0:

            print(
                "[Soiling] WARNING: empty car_mask; "
                "falling back to full-frame application."
            )

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

        alpha = texture[:, 3:4]

        alpha = self._soften_alpha(
            alpha=alpha,
            device=device,
            severity=severity,
        )

        alpha, strength = (
            self._apply_random_opacity(
                alpha=alpha,
                intensity=intensity,
                severity=severity,
                device=device,
            )
        )

        center = self._sample_patch_center(
            width=image_width,
            height=image_height,
            device=device,
        )

        return {
            "alpha": alpha,
            "center": center,
            "strength": strength,
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

        # Stronger severity allows somewhat larger chunks.
        min_scale = (
            0.035
            + 0.015 * severity
        )

        max_scale = (
            0.25
            + 0.20 * intensity
            + 0.10 * severity
        )

        max_scale = min(
            max_scale,
            0.70,
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

        # Irregular aspect ratios prevent every dirt patch
        # from looking like a circular sticker.
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
                patch_height
                * aspect
            ),
        )

        patch_height = min(
            patch_height,
            int(image_height * 0.75),
        )

        patch_width = min(
            patch_width,
            int(image_width * 0.75),
        )

        return (
            patch_height,
            patch_width,
        )

    @staticmethod
    def _sample_patch_center(
        width: int,
        height: int,
        device: torch.device,
    ) -> tuple[int, int]:

        center_x = torch.randint(
            0,
            width,
            (1,),
            device=device,
        ).item()

        center_y = torch.randint(
            0,
            height,
            (1,),
            device=device,
        ).item()

        return (
            center_x,
            center_y,
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

        batch_size = texture.shape[0]
        height = texture.shape[-2]
        width = texture.shape[-1]

        center = torch.tensor(
            [
                [
                    width / 2.0,
                    height / 2.0,
                ]
            ],
            device=device,
            dtype=texture.dtype,
        )

        angle_tensor = torch.tensor(
            [angle],
            device=device,
            dtype=texture.dtype,
        )

        # Kornia requires [B, 2].
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
    # Alpha processing
    # ------------------------------------------------------------------

    def _soften_alpha(
        self,
        alpha: torch.Tensor,
        device: torch.device,
        severity: float,
    ) -> torch.Tensor:

        image_size = min(
            alpha.shape[-2],
            alpha.shape[-1],
        )

        kernel = self._sample_odd_kernel(
            image_size=image_size,
            low=7,
            high=31,
            device=device,
        )

        # High severity can have sharper internal boundaries,
        # while the outer edge remains softened.
        sigma_min = (
            2.0
            - 0.5 * severity
        )

        sigma_max = (
            7.0
            - 1.5 * severity
        )

        sigma = torch.empty(
            1,
            device=device,
        ).uniform_(
            sigma_min,
            sigma_max,
        ).item()

        alpha = (
            kornia.filters
            .gaussian_blur2d(
                alpha,
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

        alpha_max = alpha.amax(
            dim=(-2, -1),
            keepdim=True,
        )

        alpha = alpha / (
            alpha_max + 1e-6
        )

        return alpha.clamp(
            0.0,
            1.0,
        )

    @staticmethod
    def _apply_random_opacity(
        alpha: torch.Tensor,
        intensity: float,
        severity: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        """
        Generate per-patch opacity.

        Severity changes the distribution rather than simply
        multiplying alpha.

        At low severity most patches remain transparent.

        At high severity the distribution becomes strongly
        biased toward opaque contamination, making completely
        blocked regions possible.
        """

        # Base opacity range.
        min_opacity = (
            0.04
            + 0.16 * severity
            + 0.05 * intensity
        )

        max_opacity = (
            0.25
            + 0.75 * severity
        )

        min_opacity = min(
            min_opacity,
            1.0,
        )

        max_opacity = min(
            max_opacity,
            1.0,
        )

        random_value = torch.rand(
            1,
            device=device,
        ).item()

        # Severity controls the shape of the distribution.
        #
        # severity = 0 -> mostly weak
        # severity = 1 -> strongly biased toward high opacity
        exponent = (
            1.8
            - 1.45 * severity
        )

        exponent = max(
            0.35,
            exponent,
        )

        random_value = (
            random_value
            ** exponent
        )

        opacity = (
            min_opacity
            + (
                max_opacity
                - min_opacity
            )
            * random_value
        )

        # Very high severity occasionally creates a genuinely
        # opaque chunk.
        opaque_probability = (
            max(
                0.0,
                severity - 0.72,
            )
            * 0.65
        )

        if (
            severity > 0.72
            and torch.rand(
                1,
                device=device,
            ).item()
            < opaque_probability
        ):
            opacity = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.82,
                1.0,
            ).item()

        alpha = (
            alpha * opacity
        ).clamp(
            0.0,
            1.0,
        )

        # This is the patch's local severity.
        # It is passed to the appearance model.
        strength = float(
            opacity
        )

        return alpha, strength

    # ------------------------------------------------------------------
    # Patch placement
    # ------------------------------------------------------------------

    @staticmethod
    def _place_patch(
        patch_alpha: torch.Tensor,
        center: tuple[int, int],
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        center_x, center_y = center

        patch_height = (
            patch_alpha.shape[-2]
        )

        patch_width = (
            patch_alpha.shape[-1]
        )

        x1 = (
            center_x
            - patch_width // 2
        )

        y1 = (
            center_y
            - patch_height // 2
        )

        x2 = (
            x1
            + patch_width
        )

        y2 = (
            y1
            + patch_height
        )

        dst_x1 = max(
            0,
            x1,
        )

        dst_y1 = max(
            0,
            y1,
        )

        dst_x2 = min(
            width,
            x2,
        )

        dst_y2 = min(
            height,
            y2,
        )

        src_x1 = max(
            0,
            -x1,
        )

        src_y1 = max(
            0,
            -y1,
        )

        src_x2 = (
            src_x1
            + (dst_x2 - dst_x1)
        )

        src_y2 = (
            src_y1
            + (dst_y2 - dst_y1)
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

    @staticmethod
    def _apply_car_mask(
        alpha: torch.Tensor,
        car_mask: torch.Tensor,
    ) -> torch.Tensor:

        return (
            alpha
            * car_mask
        )

    # ------------------------------------------------------------------
    # Dirty appearance
    # ------------------------------------------------------------------

    def _create_dirty_image(
        self,
        image: torch.Tensor,
        severity: float,
        patch_strength: float,
    ) -> torch.Tensor:
        """
        Generate the appearance seen through / underneath
        a contaminated optical surface.

        The result transitions continuously from:

            blurred scene

        to:

            dark, desaturated, nearly opaque mud.
        """

        blurred = self._blur_image(
            image=image,
            severity=severity,
        )

        darkened = self._darken(
            image=blurred,
            severity=severity,
        )

        desaturated = self._desaturate(
            image=blurred,
            severity=severity,
        )

        optical_soiling = (
            0.55 * darkened
            + 0.45 * desaturated
        )

        optical_soiling = (
            self._apply_tint(
                optical_soiling,
                severity=severity,
            )
        )

        # --------------------------------------------------------------
        # Black mud component
        # --------------------------------------------------------------
        #
        # At low severity this is almost absent.
        #
        # At high severity strong patches can become nearly opaque.
        #
        # This is intentionally controlled by both global severity
        # and local patch strength.
        # --------------------------------------------------------------

        black_mud_strength = self._get_black_mud_strength(
            severity=severity,
            patch_strength=patch_strength,
        )

        black_mud = torch.zeros_like(
            optical_soiling
        )

        dirty = (
            optical_soiling
            * (1.0 - black_mud_strength)
            + black_mud
            * black_mud_strength
        )

        return dirty.clamp(
            0.0,
            1.0,
        )

    @staticmethod
    def _get_black_mud_strength(
        severity: float,
        patch_strength: float,
    ) -> float:
        """
        Determine how much a dirty patch loses visibility.

        The function deliberately has a nonlinear response:
        heavy contamination becomes dramatically darker.
        """

        # Nothing special at low severity.
        if severity <= 0.35:
            return 0.0

        normalized = (
            (severity - 0.35)
            / 0.65
        )

        normalized = max(
            0.0,
            min(
                1.0,
                normalized,
            ),
        )

        # Nonlinear ramp:
        # the transition becomes strong near high severity.
        global_strength = (
            normalized ** 2.2
        )

        local_strength = max(
            0.0,
            min(
                1.0,
                patch_strength,
            ),
        )

        # Local opacity matters, but the global severity
        # still determines whether black mud is possible.
        strength = (
            0.15 * global_strength
            + 0.85
            * global_strength
            * local_strength
        )

        return max(
            0.0,
            min(
                0.95,
                strength,
            ),
        )

    @staticmethod
    def _blur_image(
        image: torch.Tensor,
        severity: float,
    ) -> torch.Tensor:

        device = image.device

        image_size = min(
            image.shape[-2],
            image.shape[-1],
        )

        kernel = SoilingModule._sample_odd_kernel(
            image_size=image_size,
            low=9,
            high=31,
            device=device,
        )

        # More severe contamination means stronger blur.
        sigma_min = (
            1.5
            + 2.0 * severity
        )

        sigma_max = (
            4.0
            + 6.0 * severity
        )

        sigma = torch.empty(
            1,
            device=device,
        ).uniform_(
            sigma_min,
            sigma_max,
        ).item()

        return (
            kornia.filters
            .gaussian_blur2d(
                image,
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

    @staticmethod
    def _darken(
        image: torch.Tensor,
        severity: float,
    ) -> torch.Tensor:

        min_strength = (
            0.03
            + 0.07 * severity
        )

        max_strength = (
            0.12
            + 0.38 * severity
        )

        strength = torch.empty(
            1,
            device=image.device,
        ).uniform_(
            min_strength,
            max_strength,
        ).item()

        return image * (
            1.0 - strength
        )

    @staticmethod
    def _desaturate(
        image: torch.Tensor,
        severity: float,
    ) -> torch.Tensor:

        gray = (
            image[:, 0:1] * 0.299
            + image[:, 1:2] * 0.587
            + image[:, 2:3] * 0.114
        )

        min_strength = (
            0.02
            + 0.03 * severity
        )

        max_strength = (
            0.15
            + 0.50 * severity
        )

        strength = torch.empty(
            1,
            device=image.device,
        ).uniform_(
            min_strength,
            max_strength,
        ).item()

        return (
            image * (1.0 - strength)
            + gray * strength
        )

    @staticmethod
    def _apply_tint(
        image: torch.Tensor,
        severity: float,
    ) -> torch.Tensor:

        # Keep tint subtle. The goal is not to colorize
        # the whole image brown.
        tint_strength = (
            0.02
            + 0.10 * severity
        )

        base_tint = torch.tensor(
            [
                0.88,
                0.86,
                0.82,
            ],
            device=image.device,
            dtype=image.dtype,
        ).view(
            1,
            3,
            1,
            1,
        )

        tint = (
            1.0
            + (
                base_tint - 1.0
            ) * tint_strength
        )

        return image * tint

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
    def _sample_odd_kernel(
        image_size: int,
        low: int,
        high: int,
        device: torch.device,
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

        # --------------------------------------------------------------
        # PIL
        # --------------------------------------------------------------

        if isinstance(
            item,
            Image.Image,
        ):

            item = item.convert(
                "RGBA"
            )

            # copy() prevents the "NumPy array is not writable"
            # warning that occurs with some PIL-backed arrays.
            array = np.asarray(
                item
            ).copy()

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
            )

            texture = (
                texture.float()
                / 255.0
            )

            texture = texture.unsqueeze(
                0
            )

        # --------------------------------------------------------------
        # Tensor
        # --------------------------------------------------------------

        elif torch.is_tensor(item):

            texture = item

            if texture.ndim == 3:
                texture = texture.unsqueeze(
                    0
                )

            if texture.ndim != 4:
                raise ValueError(
                    "Texture must have shape "
                    "[C,H,W] or [B,C,H,W], got "
                    f"{tuple(texture.shape)}"
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
                    "Texture must contain 3 or 4 channels, "
                    f"got {channels}"
                )

            texture = texture.float()

            if (
                texture.numel() > 0
                and texture.max() > 1.0
            ):
                texture = (
                    texture / 255.0
                )

        else:

            raise TypeError(
                "Unsupported dirt texture type: "
                f"{type(item)}"
            )

        return texture.to(
            device=device,
            dtype=dtype,
        ).clamp(
            0.0,
            1.0,
        )