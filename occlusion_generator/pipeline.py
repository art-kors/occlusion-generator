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
        print("[PIPELINE DEBUG] START")
        print("=" * 70)

        print(
            "[PIPELINE DEBUG] input:",
            tuple(image.shape),
        )

        print(
            "[PIPELINE DEBUG] input range:",
            image.min().item(),
            image.max().item(),
        )

        # ======================================================
        # 2. NO DEPTH
        # 3. NO CAR SEGMENTATION
        # 4. NO FOG
        #
        # Deliberately removed.
        #
        # This means:
        #
        #     image
        #       ↓
        #     reflection
        #       ↓
        #     soiling
        #       ↓
        #     flare
        #
        # No spatial mask can come from depth/car segmentation.
        # ======================================================

        current_image = image.clone()

        generated_masks = {}

        # ======================================================
        # 5. REFLECTION
        # ======================================================

        if self.config.reflection.enabled:

            print(
                "[PIPELINE DEBUG] Applying reflection..."
            )

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

            print(
                "[PIPELINE DEBUG] Reflection: DISABLED"
            )

        # ======================================================
        # 6. SOILING
        # ======================================================

        if self.config.soiling.enabled:

            print(
                "[PIPELINE DEBUG] Applying soiling..."
            )

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

            print(
                "[PIPELINE DEBUG] Soiling: DISABLED"
            )

        soil_mask_for_flare = generated_masks[
            "soiling"
        ]

        # ======================================================
        # 7. FLARE
        # ======================================================

        if self.config.flare.enabled:

            print(
                "[PIPELINE DEBUG] Applying flare..."
            )

            current_image, mask = self.modules[
                "flare"
            ].apply(
                current_image,
                None,
                None,
                self.config.flare,
                soil_mask=soil_mask_for_flare,
                **kwargs,
            )

            generated_masks["flare"] = mask

        else:

            generated_masks["flare"] = torch.zeros(
                (b, 1, h, w),
                device=self.device,
                dtype=image.dtype,
            )

            print(
                "[PIPELINE DEBUG] Flare: DISABLED"
            )

        # ======================================================
        # 8. GT
        # ======================================================

        gt_masks = self.gt_generator.generate(
            generated_masks
        )

        # ======================================================
        # 9. OUTPUT
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
            "[PIPELINE DEBUG] output range:",
            current_image.min().item(),
            current_image.max().item(),
        )

        print(
            "[PIPELINE DEBUG] mean abs difference:",
            diff.mean().item(),
        )

        print(
            "[PIPELINE DEBUG] changed > 1e-3:",
            (
                diff > 1e-3
            ).float().mean().item() * 100,
            "%",
        )

        print("=" * 70)
        print("[PIPELINE DEBUG] END")
        print("=" * 70 + "\n")

        return current_image, gt_masks