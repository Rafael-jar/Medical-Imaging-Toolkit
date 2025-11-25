import numpy as np

class MedicalImageBase:
    """
    Base class for medical image processing operations.
    Provides common utilities and dependency checking.
    """
    
    @staticmethod
    def ensure_uint8(img: np.ndarray) -> np.ndarray:
        """
        Ensure image is in uint8 format
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        np.ndarray
            Image in uint8 format
        """
        if img.dtype == np.uint8:
            return img
        
        # Normalize to 0-255 range
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_normalized = (img - img_min) / (img_max - img_min) * 255
        else:
            img_normalized = img
        
        return img_normalized.astype(np.uint8)
    
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
        return np.histogram(img.ravel(), bins=bins, range=(0, bins-1))