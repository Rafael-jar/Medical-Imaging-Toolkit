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
    
    @staticmethod
    def ensure_uint8(img: np.ndarray) -> np.ndarray:
        """
        Normalizes and converts image to 0-255 uint8 range.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        np.ndarray
            Image in uint8 format (0-255)
        """
        if img.dtype == np.uint8:
            return img
        
        # Handle 0-1 float range or arbitrary float range
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_normalized = (img - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(img, dtype=float)
            
        return (img_normalized * 255).astype(np.uint8)
    
    @staticmethod
    def ensure_float(img: np.ndarray) -> np.ndarray:
        """
        Normalizes image to 0.0-1.0 float range.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        np.ndarray
            Image in float format (0.0-1.0)
        """
        if img.dtype == np.uint8:
            return img.astype(float) / 255.0
        if img.max() > 1.0:
            return img / 255.0
        return img
    
    @staticmethod
    def convert_to_float(img: np.ndarray) -> np.ndarray:
        """
        Convert image to flat float array.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        np.ndarray
            Flattened float array
        """
        return img.ravel().astype(float) if img.dtype == bool else img.ravel()