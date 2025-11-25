import numpy as np
from typing import Tuple, Optional
from skimage.filters.rank import median
from skimage.morphology import disk
from .core import MedicalImageBase


class ImageEnhancer(MedicalImageBase):
    """
    Image enhancement operations for medical images.
    Includes filtering, contrast adjustment, and color correction.
    """
    
    def auto_level(self, img: np.ndarray, percentile_low: float = 2, 
                   percentile_high: float = 98) -> np.ndarray:
        """
        Apply auto-level (histogram stretching) to enhance image contrast
        
        Parameters:
        -----------
        img : np.ndarray
            Input image (grayscale or RGB)
        percentile_low : float
            Lower percentile for clipping (default: 2)
        percentile_high : float
            Upper percentile for clipping (default: 98)
            
        Returns:
        --------
        np.ndarray
            Enhanced image with stretched histogram
        """
        if img.ndim == 3:
            result = np.zeros_like(img)
            for c in range(img.shape[2]):
                channel = img[:, :, c]
                hist = np.bincount(channel.ravel(), minlength=256)
                cumhist = np.cumsum(hist)
                total_pixels = channel.size
                
                t_min = np.where(cumhist >= total_pixels * (percentile_low / 100))[0][0]
                t_max = np.where(cumhist >= total_pixels * (percentile_high / 100))[0][0]
                
                lut = np.zeros(256)
                if t_max > t_min:
                    step = 255 / (t_max - t_min)
                    for i in range(256):
                        if i <= t_min:
                            lut[i] = 0
                        elif i >= t_max:
                            lut[i] = 255
                        else:
                            lut[i] = step * (i - t_min)
                
                result[:, :, c] = lut[channel].astype(np.uint8)
            return result
        else:
            hist = np.bincount(img.ravel(), minlength=256)
            cumhist = np.cumsum(hist)
            total_pixels = img.size
            
            t_min = np.where(cumhist >= total_pixels * (percentile_low / 100))[0][0]
            t_max = np.where(cumhist >= total_pixels * (percentile_high / 100))[0][0]
            
            lut = np.zeros(256)
            if t_max > t_min:
                step = 255 / (t_max - t_min)
                for i in range(256):
                    if i <= t_min:
                        lut[i] = 0
                    elif i >= t_max:
                        lut[i] = 255
                    else:
                        lut[i] = step * (i - t_min)
            
            return lut[img].astype(np.uint8)
    
    def gamma_correction(self, img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction to adjust image brightness
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        gamma : float
            Gamma value (< 1 brightens, > 1 darkens)
            
        Returns:
        --------
        np.ndarray
            Gamma-corrected image
        """
        img_normalized = img / 255.0
        img_corrected = np.power(img_normalized, gamma)
        return (img_corrected * 255).astype(np.uint8)
    
    def auto_gamma(self, img: np.ndarray) -> float:
        """
        Automatically determine optimal gamma value based on image brightness
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        float
            Optimal gamma value
        """
        mean_brightness = np.mean(img)
        target_brightness = 127
        
        if mean_brightness < target_brightness:
            gamma = np.log(target_brightness / 255) / np.log(mean_brightness / 255)
        else:
            gamma = 1.0
        
        return np.clip(gamma, 0.5, 2.0)
    
    def increase_saturation(self, img: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """
        Increase color saturation of RGB image
        
        Parameters:
        -----------
        img : np.ndarray
            Input RGB image
        factor : float
            Saturation multiplier (1.0 = no change)
            
        Returns:
        --------
        np.ndarray
            Image with increased saturation
        """
        if img.ndim != 3:
            return img
        
        img_float = img.astype(np.float32)
        luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
        
        result = np.zeros_like(img_float)
        for c in range(3):
            result[:, :, c] = luminance + factor * (img_float[:, :, c] - luminance)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def denoise(self, img: np.ndarray, radius: int = 2) -> np.ndarray:
        """
        Apply median filter for noise reduction
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        radius : int
            Radius of median filter disk
            
        Returns:
        --------
        np.ndarray
            Denoised image
        """
        if img.ndim == 3:
            result = np.zeros_like(img)
            for c in range(img.shape[2]):
                result[:, :, c] = median(img[:, :, c], disk(radius))
            return result
        else:
            return median(img, disk(radius))
    
    def histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        np.ndarray
            Equalized image
        """
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
        return equalized.reshape(img.shape).astype(np.uint8)
    
    def enhance_image(self, img: np.ndarray, denoise_radius: int = 1,
                     saturation_factor: float = 1.3, gamma: Optional[float] = None,
                     auto_level_percentiles: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Comprehensive image enhancement pipeline
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        denoise_radius : int
            Radius for median filter (0 to skip)
        saturation_factor : float
            Multiplier for color saturation (1.0 = no change)
        gamma : float or None
            Gamma correction value (None = auto-detect)
        auto_level_percentiles : tuple
            (low, high) percentiles for histogram stretching
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        result = img.copy()
        
        if denoise_radius > 0:
            result = self.denoise(result, radius=denoise_radius)
        
        result = self.auto_level(result, percentile_low=auto_level_percentiles[0],
                                percentile_high=auto_level_percentiles[1])
        
        if gamma is None:
            gamma = self.auto_gamma(result)
        result = self.gamma_correction(result, gamma=gamma)
        
        if result.ndim == 3 and saturation_factor != 1.0:
            result = self.increase_saturation(result, factor=saturation_factor)
        
        return result
    
    def add_watermark(self, img: np.ndarray, watermark: np.ndarray,
                     position: Tuple[int, int] = None, alpha: float = 0.5,
                     auto_position: bool = True, auto_color: bool = True) -> np.ndarray:
        """
        Add a watermark to an image with automatic positioning and color adjustment
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        watermark : np.ndarray
            Watermark image (binary or grayscale)
        position : tuple
            (row, col) position for watermark placement (if auto_position=False)
        alpha : float
            Transparency factor (0-1)
        auto_position : bool
            Automatically position watermark in corner with most contrast
        auto_color : bool
            Automatically choose dark/light watermark based on background
            
        Returns:
        --------
        np.ndarray
            Image with watermark applied
        """
        result = img.copy()
        watermark_height, watermark_width = watermark.shape[:2]
        
        # Auto-position: find best corner
        if auto_position:
            corners = [
                (50, 50),  # Top-left
                (50, img.shape[1] - watermark_width - 50),  # Top-right
                (img.shape[0] - watermark_height - 50, 50),  # Bottom-left
                (img.shape[0] - watermark_height - 50, img.shape[1] - watermark_width - 50)  # Bottom-right
            ]
            
            # Find corner with highest contrast
            best_contrast = -1
            best_position = corners[0]
            
            for pos in corners:
                if pos[0] >= 0 and pos[1] >= 0 and \
                   pos[0] + watermark_height < img.shape[0] and \
                   pos[1] + watermark_width < img.shape[1]:
                    region = result[pos[0]:pos[0]+watermark_height, pos[1]:pos[1]+watermark_width]
                    if len(region.shape) == 2:
                        contrast = region.std()
                    else:
                        contrast = region.std()
                    
                    if contrast > best_contrast:
                        best_contrast = contrast
                        best_position = pos
            
            position = best_position
        
        # Validate position
        if position is None:
            position = (50, 50)
        
        if position[0] < 0 or position[1] < 0 or \
           position[0] + watermark_height >= img.shape[0] or \
           position[1] + watermark_width >= img.shape[1]:
            raise ValueError("Watermark position is out of image bounds.")
        
        # Extract region where watermark will be placed
        end_y = min(position[0] + watermark_height, img.shape[0])
        end_x = min(position[1] + watermark_width, img.shape[1])
        region = result[position[0]:end_y, position[1]:end_x]
        
        # Calculate average luminosity for auto-color
        if len(region.shape) == 2:
            avg_luminosity = region.mean()
        else:
            # Perceived luminosity: 0.299*R + 0.587*G + 0.114*B
            avg_luminosity = (0.299 * region[:,:,0] + 
                            0.587 * region[:,:,1] + 
                            0.114 * region[:,:,2]).mean()
        
        # Determine watermark color
        if auto_color:
            threshold = 128
            dark_watermark = avg_luminosity > threshold
            watermark_val = 0 if dark_watermark else 255
        else:
            watermark_val = 0  # Default to dark
        
        # Apply watermark
        for i in range(min(watermark_height, end_y - position[0])):
            for j in range(min(watermark_width, end_x - position[1])):
                if watermark.ndim == 2:
                    wm_pixel = watermark[i, j]
                else:
                    wm_pixel = watermark[i, j, 0]
                
                # This blends the watermark in the image with the alpha factor
                if wm_pixel > 128:
                    pos_x = position[0] + i
                    pos_y = position[1] + j
                    result[pos_x, pos_y] = alpha * watermark_val + (1 - alpha) * result[pos_x, pos_y]
        
        return result
