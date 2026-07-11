from dataclasses import dataclass
from enum import Enum

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from lib.data.taxonomies.common import SourceClass, ProjectClass, MappingSpec, MappingBuilder
from lib.data.taxonomies.accessibility_class import AccessibilityClass, ACCESSIBILITY_11_CLASSES, ACCESSIBILITY_12_CLASSES


