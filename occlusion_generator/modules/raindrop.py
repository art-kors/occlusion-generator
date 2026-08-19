from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from PIL import Image

from ..config import RainDropConfig
from ..interfaces import BaseOcclusionModule

from raindrops_generator.raindrop.dropgenerator import (
    generate_label,
    generateDrops,
)


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

        if image.ndim != 4:
            raise ValueError(
                f"Expected image [B,C,H,W], got {image.shape}"
            )

        b, c, height, width = image.shape

        if b != 1:
            raise ValueError(
                "RainDropModule currently supports batch size 1"
            )

        # ======================================================
        # 1. Pydantic config -> dict
        # ======================================================

        cfg_dict = cfg.model_dump()

        # ======================================================
        # 2. Torch -> PIL
        # ======================================================

        image_np = (
            image[0]
            .detach()
            .cpu()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .numpy()
        )

        image_pil = Image.fromarray(
            (image_np * 255).astype(np.uint8)
        )

        # ======================================================
        # 3. Create temporary input file
        # ======================================================

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
            ) as tmp:

                temp_path = tmp.name

            image_pil.save(temp_path)

            # ==================================================
            # 4. Generate drops
            # ==================================================

            List_of_Drops, label_map = generate_label(
                height,
                width,
                cfg_dict,
            )

            # ==================================================
            # 5. Old generator expects FILE PATH
            # ==================================================

            output_image, output_label, mask = generateDrops(
                temp_path,
                cfg_dict,
                List_of_Drops,
            )

        finally:

            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

        # ======================================================
        # 6. Output image -> Tensor
        # ======================================================

        if not isinstance(output_image, Image.Image):
            raise TypeError(
                f"Expected output_image to be PIL.Image, "
                f"got {type(output_image)}"
            )

        output_np = np.asarray(
            output_image
        ).astype(np.float32) / 255.0

        output_tensor = (
            torch.from_numpy(output_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=image.device,
                dtype=image.dtype,
            )
        )

        # ======================================================
        # 7. Mask -> Tensor
        # ======================================================

        mask_np = np.asarray(mask)

        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        mask_tensor = torch.from_numpy(
            mask_np.astype(np.float32)
        )

        if mask_tensor.max() > 1:
            mask_tensor /= 255.0

        mask_tensor = (
            mask_tensor
            .clamp(0.0, 1.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(
                device=image.device,
                dtype=image.dtype,
            )
        )

        return output_tensor, mask_tensor