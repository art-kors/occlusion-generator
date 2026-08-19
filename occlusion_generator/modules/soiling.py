from __future__ import annotations

from typing import Any

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import SoilingConfig
from ..interfaces import BaseOcclusionModule


class SoilingModule(BaseOcclusionModule):
    """
    Physical lens soiling:
        - depth-dependent scattering
        - light absorption
        - spatial mud contamination
    """

    @torch.no_grad()
    def apply(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        car_mask: torch.Tensor,
        cfg: SoilingConfig,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ---------------------------------------------------------
        # Input
        # ---------------------------------------------------------
        # image:    [C, H, W]
        # depth:    [H, W] or [1, H, W]
        # car_mask: [H, W] or [1, H, W]

        if not cfg.enabled or cfg.intensity == 0.0:
            return image, torch.zeros_like(image[:, 0:1, :, :])
        
        b, c, h, w = image.shape
        device = image.device

        device = image.device

        # Получаем текстуру из kwargs, если она была передана
        soil_texture = kwargs.get("soil_texture", None)

        if soil_texture is None:
            # Fallback: генерируем простую шумовую текстуру
            soil_texture = (
                torch.rand((b, 1, h, w), device=device) * 0.6 + 0.2
            )

        # если texture имеет размер [H, W]
        if soil_texture.ndim == 2:
            soil_texture = soil_texture[None, None]

        # [C, H, W]
        elif soil_texture.ndim == 3:
            soil_texture = soil_texture[None]

        # resize под image
        soil_texture = torch.nn.functional.interpolate(
            soil_texture.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # если texture RGB -> grayscale
        if soil_texture.shape[1] > 1:
            soil_texture = soil_texture.mean(dim=1, keepdim=True)

        if soil_texture is None:
            # Генерием простую серо-коричневую шумовую маску как заглушку
            soil_texture = torch.rand((b, 1, h, w), device=device) * 0.6 + 0.2
            
        # Приводим soil_texture к формату [B, C, H, W]
        if soil_texture.dim() == 2:
            soil_texture = soil_texture.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        elif soil_texture.dim() == 3:
            soil_texture = soil_texture.unsqueeze(0)             # [1, C, H, W]

        # Если текстура грязи одноканальная (маска), размазываем ее на 3 канала (RGB)
        if soil_texture.shape[1] == 1 and c == 3:
            soil_rgb = soil_texture.expand(-1, 3, -1, -1)
            # Используем саму текстуру как альфа-канал (где грязи больше - там непрозрачнее)
            soil_alpha = soil_texture
        else:
            # Если текстура уже цветная (RGB), используем ее как цвет грязи
            soil_rgb = soil_texture
            # Вычисляем альфу как среднюю яркость по каналам (черно-белый вариант)
            soil_alpha = soil_texture.mean(dim=1, keepdim=True)



        soil_texture = kwargs.get("soil_texture")
                # Умножаем альфу грязи на маску машины: грязь будет только там, где машина != 0
        final_alpha = soil_alpha * car_mask

        # Масштабируем интенсивность из конфига (от 0.0 до 1.0)
        final_alpha = final_alpha * cfg.intensity

        # Ограничиваем значения альфы строго от 0 до 1, чтобы не было артефактов
        final_alpha = torch.clamp(final_alpha, 0.0, 1.0)

        # ---------------------------------------------------------
        # 4. Наложение маски на изображение (Alpha Blending)
        # ---------------------------------------------------------
        # Формула: Result = Original * (1 - Alpha) + Overlay * Alpha
        # Раздвигаем final_alpha [B, 1, H, W] до размера изображения [B, C, H, W]
        final_alpha_3c = final_alpha.expand_as(image)

        # Смешиваем оригинал с цветом грязи в зависимости от альфы
        soiled_image = image * (1.0 - final_alpha_3c) + soil_rgb * final_alpha_3c

        # Ограничиваем финальное изображение в допустимый диапазон пикселей
        soiled_image = torch.clamp(soiled_image, 0.0, 1.0)

        # ---------------------------------------------------------
        # 5. Возврат результатов
        # ---------------------------------------------------------
        # Вторым тензором возвращаем саму маску наложения (final_alpha).
        # Обычно это нужно, чтобы нейросеть (дискретизатор или детектор) 
        # могла игнорировать эти пиксели при подсчете лосса.
        return soiled_image, final_alpha