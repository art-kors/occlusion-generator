import torch

from .config import PipelineConfig
from .preprocessors import DepthEstimator, CarSegmentator
from .gt_generator import GTGenerator

from .modules.fog import FogModule
from .modules.reflection import ReflectionModule
from .modules.soiling import SoilingModule
from .modules.flare import FlareModule


class OcclusionPipeline:

    def __init__(
        self,
        config: PipelineConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        self.depth_estimator = DepthEstimator(
            device=device
        )

        self.car_segmentator = CarSegmentator(
            device=device
        )

        self.modules = {
            "fog": FogModule(),
            "reflection": ReflectionModule(),
            "soiling": SoilingModule(),
            "flare": FlareModule(),
        }

        self.gt_generator = GTGenerator(config)

    @torch.no_grad()
    def process(
        self,
        image: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ======================================================
        # 1. INPUT
        # ======================================================

        image = image.to(self.device)

        if image.ndim != 4:
            raise ValueError(
                f"Expected image [B,C,H,W], got {image.shape}"
            )

        b, c, h, w = image.shape

        print("\n" + "=" * 70)
        print("[PIPELINE] START")
        print("=" * 70)

        print(
            "[PIPELINE] input:",
            tuple(image.shape),
        )

        # ======================================================
        # 2. DEPTH
        # ======================================================

        depth = self.depth_estimator.estimate(
            image
        )

        if depth is None:
            depth = torch.ones(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            ) * 10.0

        depth = depth.to(
            device=self.device,
            dtype=image.dtype,
        )

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        if depth.shape[-2:] != (h, w):

            depth = torch.nn.functional.interpolate(
                depth,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        # ======================================================
        # 3. CAR MASK
        # ======================================================

        car_mask = self.car_segmentator.segment(
            image
        )

        if car_mask is None:

            print(
                "[PIPELINE] car_mask = None -> "
                "using full-image mask"
            )

            car_mask = torch.ones(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

        else:

            car_mask = car_mask.to(
                device=self.device,
                dtype=image.dtype,
            )

            if car_mask.ndim == 2:
                car_mask = (
                    car_mask
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

            elif car_mask.ndim == 3:
                car_mask = car_mask.unsqueeze(1)

            if car_mask.shape[-2:] != (h, w):

                car_mask = torch.nn.functional.interpolate(
                    car_mask,
                    size=(h, w),
                    mode="nearest",
                )

        car_mask = torch.clamp(
            car_mask,
            0.0,
            1.0,
        )

        # ======================================================
        # 4. DEPTH ON CAR
        # ======================================================

        depth = torch.where(
            car_mask > 0.5,
            torch.full_like(
                depth,
                0.3,
            ),
            depth,
        )

        print(
            "[PIPELINE] depth:",
            tuple(depth.shape),
        )

        print(
            "[PIPELINE] car_mask:",
            tuple(car_mask.shape),
            "mean=",
            car_mask.mean().item(),
        )

        # ======================================================
        # 5. CURRENT IMAGE
        # ======================================================

        current_image = image.clone()

        generated_masks = {}

        # ======================================================
        # 6. FOG
        # ======================================================

        if self.config.fog.enabled:

            print(
                "[PIPELINE] Applying fog..."
            )

            current_image, mask = (
                self.modules["fog"].apply(
                    current_image,
                    depth,
                    car_mask,
                    self.config.fog,
                    **kwargs,
                )
            )

            generated_masks["fog"] = mask

        else:

            print(
                "[PIPELINE] Fog: DISABLED"
            )

            generated_masks["fog"] = torch.zeros(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

        # ======================================================
        # 7. REFLECTION
        # ======================================================

        if self.config.reflection.enabled:

            print(
                "[PIPELINE] Applying reflection..."
            )

            current_image, mask = (
                self.modules["reflection"].apply(
                    current_image,
                    depth,
                    car_mask,
                    self.config.reflection,
                    **kwargs,
                )
            )

            generated_masks["reflection"] = mask

        else:

            print(
                "[PIPELINE] Reflection: DISABLED"
            )

            generated_masks["reflection"] = torch.zeros(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

        # ======================================================
        # 8. SOILING
        # ======================================================

        if self.config.soiling.enabled:

            print(
                "[PIPELINE] Applying soiling..."
            )

            current_image, mask = (
                self.modules["soiling"].apply(
                    current_image,
                    depth,
                    car_mask,
                    self.config.soiling,
                    **kwargs,
                )
            )

            generated_masks["soiling"] = mask

        else:

            print(
                "[PIPELINE] Soiling: DISABLED"
            )

            generated_masks["soiling"] = torch.zeros(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

        # ======================================================
        # 9. FLARE
        # ======================================================

        if self.config.flare.enabled:

            print(
                "[PIPELINE] Applying flare..."
            )

            current_image, mask = (
                self.modules["flare"].apply(
                    current_image,
                    depth,
                    car_mask,
                    self.config.flare,
                    soil_mask=generated_masks[
                        "soiling"
                    ],
                    **kwargs,
                )
            )

            generated_masks["flare"] = mask

        else:

            print(
                "[PIPELINE] Flare: DISABLED"
            )

            generated_masks["flare"] = torch.zeros(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

        # ======================================================
        # 10. GT
        # ======================================================

        gt_masks = self.gt_generator.generate(
            generated_masks
        )

        # ======================================================
        # 11. OUTPUT
        # ======================================================

        current_image = torch.clamp(
            current_image,
            0.0,
            1.0,
        )

        diff = (
            current_image - image
        ).abs()

        print(
            "[PIPELINE] output:",
            tuple(current_image.shape),
        )

        print(
            "[PIPELINE] mean abs difference:",
            diff.mean().item(),
        )

        print(
            "[PIPELINE] max abs difference:",
            diff.max().item(),
        )

        print("=" * 70)
        print("[PIPELINE] END")
        print("=" * 70)

        return current_image, gt_masks