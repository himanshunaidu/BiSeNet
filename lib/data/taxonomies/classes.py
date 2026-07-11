from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass(frozen=True)
class SourceClass:
    raw_id: int
    name: str
    
@dataclass(frozen=True)
class ProjectClass:
    key: str
    name: str
    description: str
    
class AccessibilityClass(Enum):
    ROAD = ProjectClass(key="road", name="Road")
    SIDEWALK = ProjectClass(key="sidewalk", name="Sidewalk")
    BUILDING = ProjectClass(key="building", name="Building")
    POLE = ProjectClass(key="pole", name="Pole")
    TRAFFIC_LIGHT = ProjectClass(key="traffic_light", name="Traffic Light")
    TRAFFIC_SIGN = ProjectClass(key="traffic_sign", name="Traffic Sign")
    VEGETATION = ProjectClass(key="vegetation", name="Vegetation")
    TERRAIN = ProjectClass(key="terrain", name="Terrain")
    WALL = ProjectClass(key="wall", name="Wall")
    FENCE = ProjectClass(key="fence", name="Fence")
    STATIC = ProjectClass(key="static", name="Static Object")
    DYNAMIC = ProjectClass(key="dynamic", name="Dynamic Object")
    BACKGROUND = ProjectClass(key="background", name="Background")

IGNORE_LABEL = 255

@dataclass
class MappingSpec:
    name: str
    source_enum: type[Enum]
    target_classes: tuple[AccessibilityClass, ...]
    assignments: dict[Enum, AccessibilityClass]
    ignore_label: int = IGNORE_LABEL
    
    @property
    def n_classes(self) -> int:
        return len(self.target_classes)
    
    @property
    def target_to_train_id(self) -> dict[AccessibilityClass, int]:
        """
        Returns a dictionary mapping each target class to its corresponding train ID.
        """
        return {cls: idx for idx, cls in enumerate(self.target_classes)}
    
    @property
    def class_names(self) -> list[str]:
        return [cls.value.name for cls in self.target_classes]
    
    def validate(self):
        valid_targets = set(self.target_classes)
        for source_cls, target_cls in self.assignments.items():
            if not isinstance(source_cls, self.source_enum):
                raise ValueError(f"Source class {source_cls} is not a valid member of {self.source_enum.__name__}")
            if target_cls not in valid_targets:
                raise ValueError(f"Target class {target_cls} is not a valid member of target_classes")
    
    def compile_label_map(self, max_raw_id: int = 256) -> np.ndarray:
        self.validate()
        
        lb_map = np.full(max_raw_id, self.ignore_label, dtype=np.uint8)
        target_to_id = self.target_to_train_id
        for source_class, target_class in self.assignments.items():
            raw_id = source_class.value.raw_id
            train_id = target_to_id[target_class]
            lb_map[raw_id] = train_id
        return lb_map