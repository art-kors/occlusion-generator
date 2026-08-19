import torch
import torch.nn.functional as F
import kornia
import kornia.filters as KF
from ..interfaces import BaseOcclusionModule
from ..config import RainDropConfig
from PIL import Image

from raindrops_generator.raindrop.dropgenerator import generate_label, generateDrops

class RainDropModule(BaseOcclusionModule):

    @torch.no_grad()
    def apply(
        self,
        image: torch.Tensor,
        depth,
        car_mask,
        cfg: RainDropConfig,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ---------------------------------------------------------
        # image: [C, H, W] or [B, C, H, W]
        # ---------------------------------------------------------

        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    "RainDropModule currently supports batch size 1"
                )
            image = image[0]

        if image.ndim != 3:
            raise ValueError(
                f"Expected image with shape [C, H, W], got {image.shape}"
            )

        _, height, width = image.shape

        # ---------------------------------------------------------
        # Convert torch.Tensor -> PIL.Image
        # ---------------------------------------------------------

        image_np = (
            image.detach()
            .cpu()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )

        image_pil = Image.fromarray(
            (image_np * 255).astype(np.uint8)
        )

        # ---------------------------------------------------------
        # Generate drop positions / label map
        # ---------------------------------------------------------

        List_of_Drops, label_map = generate_label(
            height,
            width,
            cfg,
        )

        # ---------------------------------------------------------
        # Generate raindrops
        # ---------------------------------------------------------

        output_image, output_label, mask = generateDrops(
            image_pil,
            cfg,
            List_of_Drops,
        )

        # ---------------------------------------------------------
        # PIL -> torch.Tensor
        # ---------------------------------------------------------

        output_np = np.asarray(output_image).astype(np.float32) / 255.0

        output_tensor = torch.from_numpy(
            output_np
        ).permute(2, 0, 1).to(
            device=image.device,
            dtype=image.dtype,
        )

        # ---------------------------------------------------------
        # Convert mask to tensor
        # ---------------------------------------------------------

        mask_np = np.asarray(mask)

        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        mask_tensor = torch.from_numpy(
            mask_np
        ).to(
            device=image.device,
            dtype=torch.bool,
        )

        return output_tensor, mask_tensor