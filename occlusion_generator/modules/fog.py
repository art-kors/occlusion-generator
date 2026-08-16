import torch
import kornia
from ..interfaces import BaseOcclusionModule
from ..config import FogConfig

class FogModule(BaseOcclusionModule):
    def apply(self, image: torch.Tensor, depth: torch.Tensor, 
              car_mask: torch.Tensor, cfg: FogConfig) -> tuple[torch.Tensor, torch.Tensor]:
        
        if not cfg.enabled or cfg.intensity == 0.0:
            # Возвращаем оригинал и пустую маску тумана
            return image, torch.zeros_like(image[:, 0:1, :, :])

        b, _, h, w = image.shape
        device = image.device
        
        # 1. Маппинг слайдера [0, 1] в физический коэффициент бета
        # 0.0 -> 0.0 (нет тумана), 1.0 -> 0.2 (очень густой туман)
        beta = cfg.intensity * 0.2 
        
        # 2. Цвет тумана (Airlight)
        fog_color = torch.tensor(cfg.color, device=device).view(1, 3, 1, 1)
        
        # 3. Защита от артефактов на кузове (принудительно делаем глубину близкой)
        safe_depth = torch.where(car_mask > 0.5, 0.3, depth)
        
        # 4. Вычисление карты пропускания (Transmission Map) t = exp(-beta * d)
        t = torch.exp(-beta * safe_depth)
        
        # 5. Рендеринг по формуле атмосферного рассеяния
        foggy_image = image * t + fog_color * (1 - t)
        
        # 6. Формирование маски для GT (считаем туманом то, что потеряло >30% контраста)
        fog_gt_mask = (t < 0.7).float()
        
        return foggy_image, fog_gt_mask