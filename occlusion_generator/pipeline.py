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
        # 1. Input
        # ======================================================

        image = image.to(self.device)

        b, c, h, w = image.shape

        print("\n" + "=" * 70)
        print("[PIPELINE DEBUG] START")
        print("=" * 70)

        print(
            "[PIPELINE DEBUG] image:",
            image.shape,
            "min=",
            image.min().item(),
            "max=",
            image.max().item(),
        )

        # ======================================================
        # 2. Preprocessing
        #
        # IMPORTANT:
        # Depth + car segmentation are shared dependencies.
        # They must NOT depend on fog.enabled.
        # ======================================================

        depth = self.depth_estimator.estimate(image)

        car_mask = self.car_segmentator.segment(image)

        print(
            "[PIPELINE DEBUG] depth:",
            depth.shape,
            "min=",
            depth.min().item(),
            "max=",
            depth.max().item(),
            "mean=",
            depth.mean().item(),
        )

        print(
            "[PIPELINE DEBUG] car_mask:",
            car_mask.shape,
            "min=",
            car_mask.min().item(),
            "max=",
            car_mask.max().item(),
            "mean=",
            car_mask.mean().item(),
        )

        # ======================================================
        # 3. Depth adjustment
        #
        # Keep original behavior:
        # pixels belonging to car get near depth.
        # ======================================================

        depth = torch.where(
            car_mask > 0.5,
            torch.tensor(
                0.3,
                device=self.device,
                dtype=depth.dtype,
            ),
            depth,
        )

        # ======================================================
        # 4. Current image
        # ======================================================

        current_image = image.clone()

        generated_masks = {}

        # ======================================================
        # 5. Fog
        # ======================================================

        current_image, mask = self.modules["fog"].apply(
            current_image,
            depth,
            car_mask,
            self.config.fog,
            **kwargs,
        )

        generated_masks["fog"] = mask

        # ======================================================
        # 6. Reflection
        # ======================================================

        current_image, mask = self.modules["reflection"].apply(
            current_image,
            depth,
            car_mask,
            self.config.reflection,
            **kwargs,
        )

        generated_masks["reflection"] = mask

        # ======================================================
        # 7. Soiling
        # ======================================================

        current_image, mask = self.modules["soiling"].apply(
            current_image,
            depth,
            car_mask,
            self.config.soiling,
            **kwargs,
        )

        generated_masks["soiling"] = mask

        soil_mask_for_flare = mask

        # ======================================================
        # 8. Flare
        # ======================================================

        current_image, mask = self.modules["flare"].apply(
            current_image,
            depth,
            car_mask,
            self.config.flare,
            soil_mask=soil_mask_for_flare,
            **kwargs,
        )

        generated_masks["flare"] = mask

        # ======================================================
        # 9. Ground truth
        # ======================================================

        gt_masks = self.gt_generator.generate(
            generated_masks
        )

        # ======================================================
        # 10. Final image
        # ======================================================

        current_image = torch.clamp(
            current_image,
            0.0,
            1.0,
        )

        print(
            "[PIPELINE DEBUG] output difference:",
            (
                current_image - image
            ).abs().mean().item(),
        )

        print("=" * 70)
        print("[PIPELINE DEBUG] END")
        print("=" * 70 + "\n")

        return current_image, gt_masks