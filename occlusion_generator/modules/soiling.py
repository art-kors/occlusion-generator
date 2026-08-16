import torch
import torch.nn.functional as F
from ..interfaces import BaseOcclusionModule
from ..config import SoilingConfig

class SoilingModule(BaseOcclusionModule):
    def apply(self, image: torch.Tensor, depth: torch.Tensor, 
              car_mask: torch.Tensor, cfg: SoilingConfig) -> tuple[torch.Tensor, torch.Tensor]:
        
        if not cfg.enabled or cfg.intensity == 0.0:
            return image, torch.zeros_like(image[:, 0:1, :, :])

        b, c, h, w = image.shape
        device = image.device
        
        soil_mask = torch.zeros(b, 1, h, w, device=device)
        soil_texture = torch.zeros_like(image) 
        
        num_defects = max(1, int(cfg.intensity * 10))
        
        for _ in range(num_defects):
            size = int(torch.randint(30, int(h * 0.15), (1,)))
            cx = torch.randint(0, w, (1,)).item()
            cy = torch.randint(0, h, (1,)).item()
            
            y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            dist_sq = (x - cx)**2 + (y - cy)**2
            r = size / 2
            patch_mask = (dist_sq < r**2).float().unsqueeze(0).unsqueeze(0)
            
            # Реалистичный цвет грязи (темно-серый/коричневый)
            patch_color = torch.tensor([[[0.15]], [[0.12]], [[0.10]]], device=device).expand(b, -1, h, w)
            
            if cfg.apply_distortion and r > 20:
                # КАПЛЯ: Искажаем то, что находится ПОД каплей
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, h, device=device),
                    torch.linspace(-1, 1, w, device=device),
                    indexing='ij'
                )
                
                normalized_dist = torch.sqrt(dist_sq) / r
                normalized_dist = torch.clamp(normalized_dist, 0, 1)
                
                # Вектор искажения от центра капли
                dx = (x - cx) * normalized_dist * 0.04
                dy = (y - cy) * normalized_dist * 0.04
                
                dx_norm = dx / (w / 2)
                dy_norm = dy / (h / 2)
                
                grid = torch.stack([grid_x + dx_norm, grid_y + dy_norm], dim=-1)
                grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
                
                # Забираем искаженный кусок фона
                warped_patch = F.grid_sample(image, grid, align_corners=True, mode='bilinear', padding_mode='border')
                soil_texture = torch.where(patch_mask > 0.5, warped_patch, soil_texture)
            else:
                # ГРЯЗЬ: Просто накладываем темный цвет
                soil_texture = torch.where(patch_mask > 0.5, patch_color, soil_texture)
                
            soil_mask = torch.clamp(soil_mask + patch_mask, 0, 1)

        final_image = torch.where(soil_mask > 0.5, soil_texture, image)

        return final_image, soil_mask