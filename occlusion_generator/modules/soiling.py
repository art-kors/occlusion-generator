import torch
import torch.nn.functional as F
import kornia
from PIL import Image
import numpy as np
from .interfaces import BaseOcclusionModule
from .config import SoilingConfig

class SoilingModule(BaseOcclusionModule):
    def apply(self, image: torch.Tensor, depth: torch.Tensor, 
              car_mask: torch.Tensor, cfg: SoilingConfig, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        
        if not cfg.enabled or cfg.intensity == 0.0:
            return image, torch.zeros_like(image[:, 0:1, :, :])

        b, c, h, w = image.shape
        device = image.device
        soil_mask = torch.zeros(b, 1, h, w, device=device)
        soil_texture = image.clone() 
        num_defects = max(1, int(cfg.intensity * 8))
        
        dirt_buffer = kwargs.get("dirt_textures")

        for i in range(num_defects):
            min_size = int(min(h, w) * 0.05)
            max_size = int(min(h, w) * 0.4)
            rand_h = torch.randint(min_size, max_size, (1,)).item()
            rand_w = torch.randint(min_size, max_size, (1,)).item()
            cx = torch.randint(0, w, (1,)).item()
            cy = torch.randint(0, h, (1,)).item()

            if dirt_buffer is not None and len(dirt_buffer) > 0:
                idx = torch.randint(0, len(dirt_buffer), (1,)).item()
                item = dirt_buffer[idx]
                
                if isinstance(item, Image.Image):
                    tex = torch.from_numpy(np.array(item)).permute(2, 0, 1).float() / 255.0
                else:
                    tex = item
                    
                tex = tex.unsqueeze(0).to(device) 
                tex_resized = kornia.geometry.transform.resize(tex, (rand_h, rand_w), antialias=True)
                
                patch_rgb = tex_resized[:, :3, :, :]
                patch_alpha = tex_resized[:, 3:4, :, :]
                
                soil_texture = soil_texture * (1 - patch_alpha) + patch_rgb * patch_alpha
                soil_mask = torch.clamp(soil_mask + patch_alpha, 0, 1)
            else:
                print("⚠️ ВНИМАНИЕ: Я РИСУЮ КРУГИ, ПОТОМУ ЧТО dirt_buffer ПУСТОЙ!")
                y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                dist_sq = (x - cx)**2 + (y - cy)**2
                r = rand_h / 2
                patch_mask = (dist_sq < r**2).float().unsqueeze(0).unsqueeze(0)
                patch_color = torch.tensor([[[0.15]], [[0.12]], [[0.10]]], device=device).expand(b, -1, h, w)
                soil_texture = torch.where(patch_mask > 0.5, patch_color, soil_texture)
                soil_mask = torch.clamp(soil_mask + patch_mask, 0, 1)

        return soil_texture, soil_mask