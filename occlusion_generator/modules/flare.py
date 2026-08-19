import torch
import kornia
from ..interfaces import BaseOcclusionModule
from ..config import FlareConfig

class FlareModule(BaseOcclusionModule):
    # Note: Added soil_mask argument as discussed in the pipeline architecture
    def apply(self, image: torch.Tensor, depth: torch.Tensor, 
              car_mask: torch.Tensor, cfg: FlareConfig, 
              soil_mask: torch.Tensor | None = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        
        if not cfg.enabled or cfg.intensity == 0.0:
            return image, torch.zeros_like(image[:, 0:1, :, :])

        b, c, h, w = image.shape
        device = image.device
        
        flare_canvas = torch.zeros_like(image)
        flare_mask = torch.zeros(b, 1, h, w, device=device)

        # 1. Find light source (Mock: Random bright point, in real app: threshold + argmax)
        src_x, src_y = torch.randint(w//4, 3*w//4, (1,)).item(), torch.randint(h//4, 3*h//4, (1,)).item()
        
        # Center of image
        cx, cy = w // 2, h // 2

        # 2. Draw Streaks (Lines from source)
        # Using Kornia's affine transform to stretch a bright dot into a line
        angle = torch.rand(1).item() * 180
        streak_length = int(h * 0.8 * cfg.intensity)
        
        streak_tensor = torch.zeros(b, c, streak_length, streak_length, device=device)
        streak_tensor[:, :, streak_length//2, streak_length//2] = 1.0 # Single bright pixel
        
        # Scale it into a line
        scale = torch.tensor([[streak_length, 0, 0], [0, 2, 0]], dtype=torch.float32, device=device).unsqueeze(0)
        streak_tensor = kornia.geometry.transform.affine(streak_tensor, scale)
        
        # Rotate
        angle_rad = torch.tensor([angle], dtype=torch.float32, device=device)
        streak_tensor = kornia.geometry.transform.rotate(streak_tensor, angle_rad)
        
        # Crop/Paste back (Simplified for architecture: just blur the whole thing to simulate a wide glow)
        streak_tensor = kornia.filters.gaussian_blur2d(streak_tensor, kernel_size=(int(h*0.1), int(w*0.1)), sigma=(10.0, 10.0))
        
        # 3. The Magic Trick: Interaction with soil
        if soil_mask is not None:
            # Blur the soil mask heavily to create a "scattering area"
            scattered_soil = kornia.filters.gaussian_blur2d(soil_mask, kernel_size=(111, 111), sigma=(50.0, 50.0))
            # Reduce flare intensity where dirt is (light scatters on dirt, doesn't form clean ghosts)
            flare_multiplier = 1.0 - (scattered_soil * 0.8) 
            flare_canvas = streak_tensor * flare_multiplier * cfg.intensity
        else:
            flare_canvas = streak_tensor * cfg.intensity

        flare_mask = (flare_canvas.mean(dim=1, keepdim=True) > 0.05).float()

        # 4. Additive Blending (Flare adds light, it doesn't replace pixels)
        final_image = torch.clamp(image + flare_canvas, 0.0, 1.0)

        return final_image, flare_mask