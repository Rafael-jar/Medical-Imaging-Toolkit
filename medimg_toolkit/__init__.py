from .enhancement import ImageEnhancer
from .segmentation import ImageSegmenter
from .registration import ImageRegistrar
from .metrics import SimilarityMetrics
from .io_operations import MedicalImageIO

__version__ = "1.0.0"
__author__ = "Rafael Palma Santos"
__all__ = [
    'MedicalImageProcessor',
    'ImageEnhancer',
    'ImageSegmenter',
    'ImageRegistrar',
    'SimilarityMetrics',
    'MedicalImageIO'
]


class MedicalImageProcessor(ImageEnhancer, ImageSegmenter, ImageRegistrar, 
                           SimilarityMetrics, MedicalImageIO):
    
    def __init__(self):
        """
        Initialize the Medical Image Processor
        """
        print("MedicalImageProcessor initialized.")
