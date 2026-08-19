import torch

from .config import PipelineConfig
from .gt_generator import GTGenerator

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

        self.modules = {
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
        # 2. CURRENT IMAGE
        # ======================================================

        current_image = image.clone()

        generated_masks = {}

        # ======================================================
        # 3. REFLECTION
        # ======================================================

        if self.config.reflection.enabled:

            current_image, mask = self.modules[
                "reflection"
            ].apply(
                current_image,
                None,
                None,
                self.config.reflection,
                **kwargs,
            )

            generated_masks["reflection"] = mask

        else:

            generated_masks["reflection"] = torch.zeros(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

        # ======================================================
        # 4. SOILING
        # ======================================================

        if self.config.soiling.enabled:

            current_image, mask = self.modules[
                "soiling"
            ].apply(
                current_image,
                None,
                None,
                self.config.soiling,
                **kwargs,
            )

            generated_masks["soiling"] = mask

        else:

            generated_masks["soiling"] = torch.zeros(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

        # ======================================================
        # 5. FLARE
        # ======================================================

        if self.config.flare.enabled:

            current_image, mask = self.modules[
                "flare"
            ].apply(
                current_image,
                None,
                None,
                self.config.flare,
                soil_mask=generated_masks["soiling"],
                **kwargs,
            )

            generated_masks["flare"] = mask

        else:

            generated_masks["flare"] = torch.zeros(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

        # ======================================================
        # 6. GROUND TRUTH
        # ======================================================

        gt_masks = self.gt_generator.generate(
            generated_masks
        )

        # ======================================================
        # 7. OUTPUT
        # ======================================================

        current_image = torch.clamp(
            current_image,
            0.0,
            1.0,
        )

        print(
            "[PIPELINE] mean abs difference:",
            (
                current_image - image
            ).abs().mean().item(),
        )

        print("=" * 70)
        print("[PIPELINE] END")
        print("=" * 70)

        return current_image, gt_masks