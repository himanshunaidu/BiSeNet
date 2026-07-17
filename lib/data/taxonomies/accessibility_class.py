from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from lib.data.taxonomies.common import SourceClass, ProjectClass

class AccessibilityClass(Enum):
    ROAD = ProjectClass(key="road", name="Road")
    SIDEWALK = ProjectClass(key="sidewalk", name="Sidewalk")
    BUILDING = ProjectClass(key="building", name="Building")
    POLE = ProjectClass(key="pole", name="Pole")
    TRAFFIC_LIGHT = ProjectClass(key="traffic_light", name="Traffic Light")
    TRAFFIC_SIGN = ProjectClass(key="traffic_sign", name="Traffic Sign")
    CURB = ProjectClass(key="curb", name="Curb")
    CURB_RAMP = ProjectClass(key="curb_ramp", name="Curb Ramp")
    VEGETATION = ProjectClass(key="vegetation", name="Vegetation")
    TERRAIN = ProjectClass(key="terrain", name="Terrain")
    WALL = ProjectClass(key="wall", name="Wall")
    FENCE = ProjectClass(key="fence", name="Fence")
    STATIC = ProjectClass(key="static", name="Static Object")
    DYNAMIC = ProjectClass(key="dynamic", name="Dynamic Object")
    BACKGROUND = ProjectClass(key="background", name="Background")

ACCESSIBILITY_11_CLASSES = (
    AccessibilityClass.BACKGROUND,
    AccessibilityClass.ROAD,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.BUILDING,
    AccessibilityClass.POLE,
    AccessibilityClass.TRAFFIC_LIGHT,
    AccessibilityClass.TRAFFIC_SIGN,
    AccessibilityClass.VEGETATION,
    AccessibilityClass.TERRAIN,
    AccessibilityClass.STATIC,
    AccessibilityClass.DYNAMIC
)

ACCESSIBILITY_12_CLASSES = (
    AccessibilityClass.BACKGROUND,
    AccessibilityClass.ROAD,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.BUILDING,
    AccessibilityClass.POLE,
    AccessibilityClass.TRAFFIC_LIGHT,
    AccessibilityClass.TRAFFIC_SIGN,
    AccessibilityClass.CURB_RAMP,
    AccessibilityClass.VEGETATION,
    AccessibilityClass.TERRAIN,
    AccessibilityClass.STATIC,
    AccessibilityClass.DYNAMIC
)

ACCESSIBILITY_13_CLASSES = (
    AccessibilityClass.BACKGROUND,
    AccessibilityClass.ROAD,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.BUILDING,
    AccessibilityClass.POLE,
    AccessibilityClass.TRAFFIC_LIGHT,
    AccessibilityClass.TRAFFIC_SIGN,
    AccessibilityClass.CURB,
    AccessibilityClass.CURB_RAMP,
    AccessibilityClass.VEGETATION,
    AccessibilityClass.TERRAIN,
    AccessibilityClass.STATIC,
    AccessibilityClass.DYNAMIC
)

ACCESSIBILITY_3_CURB_RAMP_CLASSES = (
    AccessibilityClass.BACKGROUND,
    AccessibilityClass.SIDEWALK,
    AccessibilityClass.CURB_RAMP
)