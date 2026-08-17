from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class DirtAppearanceParams:
    """
    Resolved parameters for one dirt patch.

    All values are in [0, 1] unless otherwise stated.

    coverage
        How much of the generated patch shape is actually occupied by dirt.

    density
        How dense the dirt is inside the occupied area.

    opacity
        Maximum optical opacity of the dirt body.

    darkness
        Darkness of the dirt material.

    structure
        Strength of texture-derived local variation.

    edge_softness
        Width/strength of the transition between dirt and clean image.

    opaque_core
        Fraction/strength of the patch that can become truly opaque.

    residue
        Amount of thin semi-transparent dirt around the main body.
    """

    coverage: float
    density: float
    opacity: float
    darkness: float
    structure: float
    edge_softness: float
    opaque_core: float
    residue: float


# ============================================================================
# Soiling Module
# ============================================================================


class SoilingModule(BaseOcclusionModule):
    """
    Camera/lens soiling augmentation.

    The module intentionally separates:

        intensity
            How much dirt is generated:
            number and size of patches.

        severity
            How aggressive the dirt becomes:
            coverage, density, opacity, darkness and opaque-core strength.

    The important architectural distinction is:

        texture alpha
            describes SHAPE

        generated alpha
            describes ACTUAL optical opacity

        texture RGB
            describes LOCAL STRUCTURE

        severity
            controls the overall physical strength of the contamination.

    This prevents dirt from degenerating into a simple pasted binary mask.

    Expected image:
        [B, 3, H, W], float, nominally in [0, 1]

    Expected dirt texture:
        PIL RGB/RGBA image or torch Tensor:
            [C, H, W]
            [B, C, H, W]

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
                torch.zeros_like(
                    image[:, 0:1]
                ),
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

        result = (
            image[:, :3]
            .clamp(0.0, 1.0)
            .clone()
        )

        effective_car_mask = (
            self._prepare_car_mask(
                car_mask=car_mask,
                batch_size=batch_size,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
            )
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

            params = self._resolve_dirt_params(
                intensity=intensity,
                severity=severity,
                device=device,
            )

            patch = self._create_patch(
                dirt_buffer=dirt_buffer,
                image_height=height,
                image_width=width,
                intensity=intensity,
                severity=severity,
                params=params,
                device=device,
                dtype=dtype,
            )

            full_alpha, full_texture = (
                self._place_patch(
                    patch=patch,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    device=device,
                    dtype=dtype,
                )
            )

            full_alpha = self._apply_car_mask(
                full_alpha,
                effective_car_mask,
            )

            dirty_image = (
                self._create_dirty_image(
                    image=result,
                    texture_rgb=full_texture,
                    params=params,
                )
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

        if (
            dirt_buffer is None
            or len(dirt_buffer) == 0
        ):
            raise RuntimeError(
                "SoilingModule requires "
                "a non-empty dirt_buffer."
            )

        return dirt_buffer

    @staticmethod
    def _get_num_patches(
        intensity: float,
        severity: float,
    ) -> int:

        # Intensity is the main amount control.
        base_count = int(
            round(
                1.0
                + intensity * 8.0
            )
        )

        # Severity slightly increases the amount of heavy contamination,
        # but does not directly control opacity.
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
    # Dirt parameter model
    # ------------------------------------------------------------------

    def _resolve_dirt_params(
        self,
        intensity: float,
        severity: float,
        device: torch.device,
    ) -> DirtAppearanceParams:
        """
        Converts high-level intensity/severity into independent dirt
        properties.

        The mapping is intentionally non-linear.

        Severity does NOT mean "make the whole mask opaque".

        Instead it gradually increases:
            - coverage
            - density
            - opacity
            - darkness
            - structure
            - opaque core
        """

        # --------------------------------------------------------------
        # Coverage
        # --------------------------------------------------------------

        coverage = (
            0.15
            + 0.70 * (severity ** 1.15)
        )

        # Intensity has a smaller effect on the area of each patch.
        coverage += (
            0.10 * intensity
        )

        coverage = self._clamp01(
            coverage
        )

        # --------------------------------------------------------------
        # Density
        # --------------------------------------------------------------

        density = (
            0.12
            + 0.78 * (severity ** 1.30)
        )

        density = self._clamp01(
            density
        )

        # --------------------------------------------------------------
        # Opacity
        # --------------------------------------------------------------

        opacity = (
            0.08
            + 0.90 * (severity ** 1.45)
        )

        opacity = self._clamp01(
            opacity
        )

        # --------------------------------------------------------------
        # Darkness
        # --------------------------------------------------------------

        darkness = (
            0.10
            + 0.85 * (severity ** 1.20)
        )

        darkness = self._clamp01(
            darkness
        )

        # --------------------------------------------------------------
        # Texture structure
        # --------------------------------------------------------------

        structure = (
            0.20
            + 0.80 * severity
        )

        structure = self._clamp01(
            structure
        )

        # --------------------------------------------------------------
        # Edge softness
        # --------------------------------------------------------------

        # Heavy dirt gets somewhat harder edges.
        edge_softness = (
            0.85
            - 0.60 * severity
        )

        edge_softness = self._clamp01(
            edge_softness
        )

        # --------------------------------------------------------------
        # Opaque core
        # --------------------------------------------------------------

        # This is deliberately separate from opacity.
        #
        # opacity=1 does not mean the whole patch is binary.
        #
        # opaque_core controls how much of the strongest part of the
        # patch can actually become alpha=1.

        opaque_core = (
            0.02
            + 0.78 * (severity ** 2.0)
        )

        opaque_core = self._clamp01(
            opaque_core
        )

        # --------------------------------------------------------------
        # Residue
        # --------------------------------------------------------------

        # Thin residue is strongest at medium/high severity.
        residue = (
            0.05
            + 0.25 * severity
        )

        residue = self._clamp01(
            residue
        )

        return DirtAppearanceParams(
            coverage=coverage,
            density=density,
            opacity=opacity,
            darkness=darkness,
            structure=structure,
            edge_softness=edge_softness,
            opaque_core=opaque_core,
            residue=residue,
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
            car_mask = car_mask.unsqueeze(
                1
            )

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
        params: DirtAppearanceParams,
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
                coverage=params.coverage,
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

        alpha = self._build_dirt_alpha(
            texture_alpha=texture_alpha,
            params=params,
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
    # Alpha generation
    # ------------------------------------------------------------------

    def _build_dirt_alpha(
        self,
        texture_alpha: torch.Tensor,
        params: DirtAppearanceParams,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generates a continuous optical opacity field.

        This is intentionally NOT a binary mask generator.

        The resulting alpha consists of:

            1. soft dirt body
            2. dense internal regions
            3. optional opaque core
            4. thin residue around the body
        """

        source = texture_alpha.clamp(
            0.0,
            1.0,
        )

        source_max = source.amax(
            dim=(-2, -1),
            keepdim=True,
        )

        shape = (
            source
            / (source_max + 1e-6)
        ).clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Base smoothing
        # --------------------------------------------------------------

        image_size = min(
            shape.shape[-2],
            shape.shape[-1],
        )

        kernel = self._sample_odd_kernel(
            image_size=image_size,
            low=3,
            high=9,
            device=device,
        )

        sigma = (
            0.5
            + 1.8 * params.edge_softness
        )

        smooth_shape = (
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

        smooth_shape = smooth_shape.clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Coverage
        # --------------------------------------------------------------

        # Coverage does not simply multiply alpha.
        # It changes how much of the shape is considered actual dirt.

        coverage_threshold = (
            0.95
            - 0.90 * params.coverage
        )

        coverage_threshold = max(
            0.05,
            min(
                0.95,
                coverage_threshold,
            ),
        )

        coverage_mask = (
            smooth_shape
            >= coverage_threshold
        ).to(
            dtype=smooth_shape.dtype
        )

        # Make coverage continuous instead of binary.
        coverage_field = (
            smooth_shape
            * coverage_mask
        )

        # --------------------------------------------------------------
        # Density
        # --------------------------------------------------------------

        # Density controls how much of the interior is filled.
        #
        # Instead of random per-pixel noise, use low-frequency noise
        # so the result looks like actual mud clumps.

        low_freq_noise = self._generate_low_frequency_noise(
            shape=shape,
            device=device,
            dtype=shape.dtype,
        )

        density_field = (
            0.65
            + 0.35 * low_freq_noise
        )

        density_field = (
            density_field
            * params.density
            + (1.0 - params.density)
        )

        body = (
            coverage_field
            * density_field
        )

        # --------------------------------------------------------------
        # Base opacity
        # --------------------------------------------------------------

        alpha = (
            body
            * params.opacity
        )

        # --------------------------------------------------------------
        # Opaque core
        # --------------------------------------------------------------

        if params.opaque_core > 0.0:

            core_threshold = (
                0.90
                - 0.75 * params.opaque_core
            )

            core_threshold = max(
                0.10,
                min(
                    0.90,
                    core_threshold,
                ),
            )

            core = (
                smooth_shape
                >= core_threshold
            ).to(
                dtype=smooth_shape.dtype
            )

            # Do not make the whole blob binary.
            #
            # opaque_core determines the strength of the core,
            # while the shape itself determines where it can exist.

            core_strength = (
                params.opaque_core
            )

            alpha = torch.maximum(
                alpha,
                core * core_strength,
            )

        # --------------------------------------------------------------
        # Thin residue
        # --------------------------------------------------------------

        if params.residue > 0.0:

            residue_threshold = (
                0.20
                + 0.30 * params.coverage
            )

            residue_mask = (
                smooth_shape
                > residue_threshold
            ).to(
                dtype=smooth_shape.dtype
            )

            residue_alpha = (
                smooth_shape
                * params.residue
                * 0.35
                * residue_mask
            )

            alpha = torch.maximum(
                alpha,
                residue_alpha,
            )

        return alpha.clamp(
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Low-frequency structure
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_low_frequency_noise(
        shape: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        height = shape.shape[-2]
        width = shape.shape[-1]

        noise_h = max(
            2,
            height // 12,
        )

        noise_w = max(
            2,
            width // 12,
        )

        noise = torch.rand(
            shape.shape[0],
            1,
            noise_h,
            noise_w,
            device=device,
            dtype=dtype,
        )

        noise = F.interpolate(
            noise,
            size=(
                height,
                width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        return noise.clamp(
            0.0,
            1.0,
        )

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
        coverage: float,
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
            0.80,
            0.20
            + 0.28 * intensity
            + 0.22 * coverage,
        )

        max_scale = max(
            min_scale,
            max_scale,
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
                    image_height * 0.80
                ),
            ),
            min(
                patch_width,
                int(
                    image_width * 0.80
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
    # Dirty appearance
    # ------------------------------------------------------------------

    def _create_dirty_image(
        self,
        image: torch.Tensor,
        texture_rgb: torch.Tensor,
        params: DirtAppearanceParams,
    ) -> torch.Tensor:
        """
        Generates the material/color of the dirt.

        Opacity is NOT handled here.

        This function only answers:

            "What does the dirt look like where it exists?"
        """

        # --------------------------------------------------------------
        # Luminance structure
        # --------------------------------------------------------------

        luminance = (
            texture_rgb[:, 0:1] * 0.299
            + texture_rgb[:, 1:2] * 0.587
            + texture_rgb[:, 2:3] * 0.114
        )

        luminance = luminance.clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Local contrast / detail
        # --------------------------------------------------------------

        blurred = (
            kornia.filters
            .gaussian_blur2d(
                luminance,
                kernel_size=(5, 5),
                sigma=(1.2, 1.2),
            )
        )

        detail = (
            luminance
            - blurred
        )

        detail = (
            detail
            * 2.0
            + 0.5
        ).clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Combine structure
        # --------------------------------------------------------------

        structure = (
            luminance * 0.65
            + detail * 0.35
        )

        structure = (
            0.5
            + params.structure
            * (structure - 0.5)
        ).clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Mud color
        # --------------------------------------------------------------

        base_r = torch.empty(
            1,
            device=image.device,
        ).uniform_(
            0.10,
            0.23,
        ).item()

        base_g = torch.empty(
            1,
            device=image.device,
        ).uniform_(
            0.075,
            0.19,
        ).item()

        base_b = torch.empty(
            1,
            device=image.device,
        ).uniform_(
            0.045,
            0.15,
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
        # Darkness
        # --------------------------------------------------------------

        brightness = (
            1.0
            - 0.88 * params.darkness
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
        # Surface variation
        # --------------------------------------------------------------

        dirt = (
            base_color
            * (
                0.60
                + 0.40 * structure
            )
        )

        return dirt.expand_as(
            image
        ).clamp(
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    @staticmethod
    def _place_patch(
        patch: dict[str, Any],
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        center_x, center_y = (
            patch["center"]
        )

        patch_h, patch_w = (
            patch["texture_rgb"]
            .shape[-2:]
        )

        x1 = (
            center_x
            - patch_w // 2
        )

        y1 = (
            center_y
            - patch_h // 2
        )

        x2 = x1 + patch_w
        y2 = y1 + patch_h

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
            batch_size,
            1,
            height,
            width,
            device=device,
            dtype=dtype,
        )

        full_texture = torch.zeros(
            batch_size,
            3,
            height,
            width,
            device=device,
            dtype=dtype,
        )

        if (
            dst_x2 <= dst_x1
            or dst_y2 <= dst_y1
        ):
            return (
                full_alpha,
                full_texture,
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

        full_texture[
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

        return (
            full_alpha,
            full_texture,
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

        return (
            alpha
            * car_mask
        )

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

            texture = (
                texture.unsqueeze(0)
            )

        elif torch.is_tensor(item):

            texture = item

            if texture.ndim == 3:
                texture = (
                    texture.unsqueeze(0)
                )

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