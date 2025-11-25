import numpy as np

class MedicalImageBase:
    """
    Base class for medical image processing operations.
    Provides common utilities and dependency checking.
    """
    
    @staticmethod
    def compute_histogram(img: np.ndarray, bins: int = 256) -> tuple:
        """
        Compute image histogram
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        bins : int
            Number of histogram bins
            
        Returns:
        --------
        tuple
            (histogram, bin_edges)
        """
        return np.histogram(img.ravel(), bins=range(bins + 1))