from pydantic import BaseModel, Field
from typing import Tuple, Optional

class ModuleConfig(BaseModel):
    enabled: bool = False
    intensity: float = Field(0.0, ge=0.0, le=1.0, description="Сила эффекта от 0 до 1")

class FogConfig(ModuleConfig):
    color: Tuple[float, float, float] = (0.75, 0.78, 0.82) # RGB нормализованный

class ReflectionConfig(ModuleConfig):
    pass

class SoilingConfig(ModuleConfig):
    pass
class FlareConfig(ModuleConfig):
    pass

class PipelineConfig(BaseModel):
    fog: FogConfig = Field(default_factory=FogConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    soiling: SoilingConfig = Field(default_factory=SoilingConfig)
    flare: FlareConfig = Field(default_factory=FlareConfig)
    
    gt_format: str = "multi_channel" # или "priority_single"