import torch
import torch.nn.functional as F
from ..interfaces import BaseOcclusionModule
from ..config import SoilingConfig

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
        
        # Получаем батч с реальными текстурами (B_s, 4, H_s, W_s) - 4 канала это RGBA
        dirt_buffer = kwargs.get("dirt_textures")

        for i in range(num_defects):
            # 1. Случайный размер и позиция
            min_size = int(min(h, w) * 0.05)
            max_size = int(min(h, w) * 0.4)
            rand_h = torch.randint(min_size, max_size, (1,)).item()
            rand_w = torch.randint(min_size, max_size, (1,)).item()
            
            cx = torch.randint(0, w, (1,)).item()
            cy = torch.randint(0, h, (1,)).item()

            if dirt_buffer is not None and len(dirt_buffer) > 0:
                # --- ИСПОЛЬЗУЕМ РЕАЛЬНЫЕ ТЕКСТУРЫ ---
                idx = torch.randint(0, len(dirt_buffer), (1,)).item()
                tex = dirt_buffer[idx].unsqueeze(0).to(device) # (1, 4, H_s, W_s)
                
                # Ресайзим до случайного размера!
                tex_resized = kornia.geometry.transform.resize(tex, (rand_h, rand_w), antialias=True)
                
                patch_rgb = tex_resized[:, :3, :, :]
                patch_alpha = tex_resized[:, 3:4, :, :]
                
                # Определяем, капля это (квадратная форма) или грязь (вытянутая)
                is_drop = 0.7 < (rand_h / rand_w) < 1.3
                
                if is_drop and cfg.apply_distortion:
                    # --- ЛОГИКА ПРЕЛОМЛЕНИЯ КАПЛИ ---
                    grid_y, grid_x = torch.meshgrid(
                        torch.linspace(-1, 1, rand_h, device=device),
                        torch.linspace(-1, 1, rand_w, device=device), indexing='ij')
                    
                    # Центр капли
                    cy_n, cx_n = 0.0, 0.0
                    dist_sq = (grid_x - cx_n)**2 + (grid_y - cy_n)**2
                    r = 1.0
                    normalized_dist = torch.sqrt(dist_sq) / r
                    normalized_dist = torch.clamp(normalized_dist, 0, 1)
                    
                    # Вектор искажения (линза)
                    dx = (grid_x - cx_n) * normalized_dist * 0.05
                    dy = (grid_y - cy_n) * normalized_dist * 0.05
                    dx_norm = dx / (rand_w / 2)
                    dy_norm = dy / (rand_h / 2)
                    
                    grid = torch.stack([grid_x + dx_norm, grid_y + dy_norm], dim=-1)
                    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
                    
                    # Искажаем фон под каплей
                    warped_patch = F.grid_sample(image, grid, align_corners=True, mode='bilinear', padding_mode='border')
                    
                    # Накладываем искаженный фон туда, где альфа-канал капли > 0.5
                    soil_texture = torch.where(patch_alpha > 0.5, warped_patch, soil_texture)
                else:
                    # --- ЛОГИКА ГРЯЗИ (Простое наложение) ---
                    soil_texture = soil_texture * (1 - patch_alpha) + patch_rgb * patch_alpha
                    
                soil_mask = torch.clamp(soil_mask + patch_alpha, 0, 1)
            else:
                # --- ФОЛЛБЭК (Процедурная генерация) ---
                y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                dist_sq = (x - cx)**2 + (y - cy)**2
                r = rand_h / 2
                patch_mask = (dist_sq < r**2).float().unsqueeze(0).unsqueeze(0)
                patch_color = torch.tensor([[[0.15]], [[0.12]], [[0.10]]], device=device).expand(b, -1, h, w)
                soil_texture = torch.where(patch_mask > 0.5, patch_color, soil_texture)
                soil_mask = torch.clamp(soil_mask + patch_mask, 0, 1)

        return soil_texture, soil_mask