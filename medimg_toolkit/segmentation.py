import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

# Scikit-image imports
from skimage.filters import sobel, threshold_otsu
from skimage.morphology import disk, opening, closing, binary_closing
from skimage.segmentation import watershed, random_walker
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny, local_binary_pattern
from skimage.measure import label, regionprops, find_contours, approximate_polygon

# Scipy imports
from scipy.ndimage import binary_fill_holes
from scipy import ndimage as ndi

# Local imports
from .core import MedicalImageBase

class ImageSegmenter(MedicalImageBase):
    """
    Image segmentation operations for medical images.
    Includes custom implementations of histogram-based and region-growing algorithms,
    alongside wrappers for standard library methods.
    """

    # --- Helper Methods to Reduce "Mess" ---

    def _ensure_uint8(self, img: np.ndarray) -> np.ndarray:
        """Helper: Normalizes and converts image to 0-255 uint8 range."""
        if img.dtype == np.uint8:
            return img
        
        # Handle 0-1 float range or arbitrary float range
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_normalized = (img - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(img, dtype=float)
            
        return (img_normalized * 255).astype(np.uint8)

    def _ensure_float(self, img: np.ndarray) -> np.ndarray:
        """Helper: Normalizes image to 0.0-1.0 float range."""
        if img.dtype == np.uint8:
            return img.astype(float) / 255.0
        if img.max() > 1.0:
            return img / 255.0
        return img

    def _load_io_helper(self):
        """Lazy loader for IO operations to avoid circular imports."""
        try:
            from .io_operations import MedicalImageIO
            return MedicalImageIO()
        except ImportError:
            raise ImportError("Cannot load medical volume without .io_operations module")

    # --- Custom Algorithm Implementations ---

    def otsu_threshold(self, histogram: np.ndarray) -> int:
        """
        Calculates optimal threshold using a custom implementation of Otsu's method.
        Minimizes intra-class variance.
        """
        total_pixels = sum(histogram)
        max_t = len(histogram) - 1
        best_threshold = 1
        max_variance = 0
        
        # Pre-calculate arrays for vectorization (makes the loop cleaner)
        indices = np.arange(len(histogram))
        
        for t in range(1, max_t):
            # Split histogram at threshold t
            bg_hist = histogram[:t+1]
            fg_hist = histogram[t+1:]
            
            # Weights (probabilities)
            w_bg = np.sum(bg_hist) / total_pixels
            w_fg = np.sum(fg_hist) / total_pixels
            
            if w_bg == 0 or w_fg == 0:
                continue
                
            # Means
            mean_bg = np.sum(indices[:t+1] * bg_hist) / np.sum(bg_hist)
            mean_fg = np.sum(indices[t+1:] * fg_hist) / np.sum(fg_hist)
            
            # Between-class variance
            var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
            
            if var_between > max_variance:
                max_variance = var_between
                best_threshold = t
        
        return best_threshold
    
    def optimal_threshold(self, histogram: np.ndarray, initial_t: int = 128) -> int:
        """
        Calculates threshold using an iterative averaging method.
        """
        t = initial_t
        indices = np.arange(len(histogram))
        
        while True:
            # Slices
            bg_hist = histogram[:t+1]
            fg_hist = histogram[t+1:]
            
            # Calculate means safely
            sum_bg = np.sum(bg_hist)
            sum_fg = np.sum(fg_hist)
            
            mean_bg = np.sum(indices[:t+1] * bg_hist) / sum_bg if sum_bg > 0 else 0
            mean_fg = np.sum(indices[t+1:] * fg_hist) / sum_fg if sum_fg > 0 else 0
            
            # New threshold is the average of the means
            new_t = int((mean_bg + mean_fg) / 2)
            
            if new_t == t:
                break
            t = new_t
        
        return t
    
    def region_growing(self, img: np.ndarray, seed: Tuple[int, int], 
                      threshold: float = 50) -> np.ndarray:
        """
        Segments a region by iteratively adding connected neighbors 
        within a specific intensity threshold.
        """
        rows, cols = img.shape
        segmented = np.zeros_like(img, dtype=bool)
        seed_value = float(img[seed])
        
        # Using a stack for Depth-First Search (DFS)
        stack = [seed]
        segmented[seed] = True
        
        # Define 4-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while stack:
            y, x = stack.pop()
            
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                
                # Check bounds
                if 0 <= ny < rows and 0 <= nx < cols:
                    if not segmented[ny, nx]:
                        # Check intensity similarity
                        pixel_value = float(img[ny, nx])
                        if abs(pixel_value - seed_value) < threshold:
                            segmented[ny, nx] = True
                            stack.append((ny, nx))
        
        return segmented

    # --- High-Level Segmentation Methods ---

    def histogram_thresholding(self, img: np.ndarray, method: str = 'otsu', 
                             morphological_clean: bool = True) -> np.ndarray:
        """Segments image using histogram analysis (Otsu or Iterative Optimal)."""
        img_uint8 = self._ensure_uint8(img)
        
        # Calculate histogram (ignore black background 0)
        hist, _ = np.histogram(img_uint8, bins=256, range=(0, 256))
        hist[0] = 0 
        
        if method == 'otsu':
            threshold = self.otsu_threshold(hist)
        elif method == 'optimal':
            threshold = self.optimal_threshold(hist)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        binary_img = img_uint8 >= threshold
        
        if morphological_clean:
            selem = disk(3)
            binary_img = closing(opening(binary_img, selem), selem)
        
        return binary_img
    
    def watershed_segmentation(self, img: np.ndarray, morphological_clean: bool = True) -> np.ndarray:
        """Applies Watershed algorithm using Sobel edges and intensity markers."""
        img_uint8 = self._ensure_uint8(img)
        elevation_map = sobel(img_uint8)
        
        # Generate markers based on Otsu
        hist, _ = np.histogram(img_uint8, bins=256, range=(0, 256))
        hist[0] = 0
        thresh = self.otsu_threshold(hist)
        
        markers = np.zeros_like(img_uint8, dtype=int)
        markers[img_uint8 < thresh * 0.7] = 1    # Background
        markers[img_uint8 > thresh * 1.3] = 2    # Foreground
        
        segmentation = watershed(elevation_map, markers)
        
        # Extract foreground (label 2)
        mask = (segmentation == 2)
        mask = ndi.binary_fill_holes(mask)
        
        if morphological_clean:
            selem = disk(3)
            mask = closing(opening(mask, selem), selem)
            
        return mask

    def edge_based_segmentation(self, img: np.ndarray, sigma: float = 2.0, 
                                fill_holes: bool = True) -> np.ndarray:
        """Canny edge detection with optional hole filling."""
        img_float = self._ensure_float(img)
        edges = canny(img_float, sigma=sigma)
        
        if not fill_holes:
            return edges

        # Refine with Otsu for filling
        threshold = threshold_otsu(img_float)
        base_mask = img_float > threshold
        filled = binary_fill_holes(base_mask)
        return binary_closing(filled, disk(3))
    
    def random_walker_segmentation(self, img: np.ndarray, beta: int = 130) -> np.ndarray:
        """Graph-based Random Walker segmentation optimized for Brain MRI."""
        img_norm = self._ensure_float(img)
        
        # Initialize markers
        markers = np.zeros_like(img_norm, dtype=np.int32)
        
        # Heuristic: Center of image is likely brain (label 1)
        cx, cy = img_norm.shape[0]//2, img_norm.shape[1]//2
        center_region = img_norm[cx-15:cx+15, cy-15:cy+15]
        
        if center_region.mean() > 0.2:
            markers[cx-10:cx+10, cy-10:cy+10] = 1
        
        # Heuristic: Borders are background (label 2)
        markers[:5, :] = 2
        markers[-5:, :] = 2
        markers[:, :5] = 2
        markers[:, -5:] = 2
        
        # Execute Random Walker
        if np.max(markers) > 1:
            seg = random_walker(img_norm, markers, beta=beta, mode='bf')
            return (seg == 1)
                
                
        # Fallback
        return img_norm > threshold_otsu(img_norm)
    
    def texture_based_segmentation(self, img: np.ndarray, P: int = 8, R: float = 1.0,
                                  clean_result: bool = True) -> np.ndarray:
        """Segments based on Local Binary Patterns (LBP)."""
        img_uint8 = self._ensure_uint8(img)
        
        # Compute LBP
        lbp = local_binary_pattern(img_uint8, P=P, R=R, method='uniform')
        
        # Threshold LBP
        binary_result = lbp > threshold_otsu(lbp)
        
        if clean_result:
            # Refine using intensity mask
            intensity_mask = img_uint8 > self.otsu_threshold(np.histogram(img_uint8, bins=256)[0])
            combined = binary_result & intensity_mask
            combined = binary_fill_holes(binary_closing(combined, disk(3)))
            return combined
            
        return binary_result

    # --- Main Interface ---

    def segment_image(self, img: np.ndarray, method: str = 'watershed',
                     remove_background: bool = False, 
                     background_is_dark: bool = True,
                     morphological_operations: bool = True,
                     preserve_intensities: bool = False,
                     seed_region_growing: Tuple[int, int] = None) -> np.ndarray:
        """
        Master dispatcher for segmentation.
        
        Args:
            method: 'otsu', 'optimal', 'watershed', 'region_growing', 'canny', 'random_walker', 'texture'
            preserve_intensities: If True, returns the original image masked. If False, returns binary mask.
        """
        original_img = img.copy()
        
        # Pre-processing
        if img.ndim == 3:
            img = rgb2gray(img)
        
        # Pre-masking (Background Removal)
        bg_mask = np.ones_like(img, dtype=bool)
        if remove_background:
            img_uint8 = self._ensure_uint8(img)
            hist = np.histogram(img_uint8, bins=256)[0]
            thresh = self.otsu_threshold(hist)
            
            raw_mask = (img_uint8 >= thresh) if background_is_dark else (img_uint8 <= thresh)
            
            selem = disk(5)
            bg_mask = closing(opening(raw_mask, selem), selem)
            img = img * bg_mask

        # Dispatcher
        if method in ['otsu', 'histogram']:
            result = self.histogram_thresholding(img, 'otsu', morphological_operations)
        elif method == 'optimal':
            result = self.histogram_thresholding(img, 'optimal', morphological_operations)
        elif method == 'watershed':
            result = self.watershed_segmentation(img, morphological_operations)
        elif method == 'region_growing':
            # Use provided seed or auto-detect brightest point
            if seed_region_growing is not None:
                seed_y, seed_x = seed_region_growing
            else:
                print("using bad ones")
                seed_y, seed_x = np.unravel_index(np.argmax(img), img.shape)
            result = self.region_growing(img, (seed_y, seed_x))
        elif method in ['edge', 'canny']:
            result = self.edge_based_segmentation(img)
        elif method in ['random_walker', 'graph']:
            result = self.random_walker_segmentation(img)
        elif method in ['texture', 'lbp']:
            result = self.texture_based_segmentation(img)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Final composition
        final_mask = result.astype(bool)
        
        if preserve_intensities:
            # Ensure types match before multiplication
            if original_img.ndim == 3 and final_mask.ndim == 2:
                final_mask = np.stack([final_mask]*3, axis=-1)
            return original_img * final_mask
            
        return final_mask.astype(np.uint8) * 255 # Return standard binary mask (0 or 255)

    def identify_shapes(self, img: np.ndarray, tolerance: float = 5) -> Dict[str, Any]:
        """Identifies geometric shapes in a binary image."""
        # Normalize inputs
        if img.ndim == 3:
            if img.shape[2] == 4: img = rgba2rgb(img)
            img = rgb2gray(img)
        
        thresh = threshold_otsu(img)
        binary = img > thresh
        
        # Ensure object is white
        if binary[0, 0]: binary = ~binary
        
        label_img = label(binary)
        regions = regionprops(label_img)
        
        results = {
            'label_image': label_img,
            'regions': regions,
            'classifications': [],
            'corners': [],
            'centroids': []
        }
        
        for props in regions:
            results['centroids'].append(props.centroid)
            
            # Find contours
            contours = find_contours(label_img == props.label, 0.5)
            if not contours:
                results['classifications'].append("Unknown")
                results['corners'].append([])
                continue
                
            contour = max(contours, key=len)
            coords = approximate_polygon(contour, tolerance=tolerance)
            results['corners'].append(coords)
            
            # Classify
            n_corners = len(coords) - 1
            if n_corners < 3: shape = "Circle/Ellipse"
            elif n_corners == 3: shape = "Triangle"
            elif n_corners == 4: shape = "Square" if props.eccentricity < 0.2 else "Rectangle"
            elif n_corners == 5: shape = "Pentagon"
            elif n_corners == 6: shape = "Hexagon"
            else: shape = f"{n_corners}-sided Polygon"
            
            results['classifications'].append(shape)
            
        return results

    def segment_volume(self, file_path: str, method: str = 'watershed', 
                              is_nifti: bool = True, **kwargs) -> np.ndarray:
        """
        Wrapper for 3D volumes (DICOM/NIfTI).
        Handles slicing logic centrally.
        """
        io = self._load_io_helper()
        
        if is_nifti:
            volume = io.load_nifti(file_path)
            # Default NIfTI args
            slice_axis = kwargs.get('slice_axis', 2)
            segment_all = kwargs.get('segment_all_slices', False)
        else:
            # Dicom handling
            path_obj = Path(file_path)
            if path_obj.is_dir():
                volume, _ = io.load_dicom_series(file_path)
            else:
                volume = io.load_dicom(file_path)
                # If single file is 2D, return immediately
                if volume.ndim == 2:
                    return self.segment_image(volume, method=method, remove_background=False)
            
            slice_axis = 0 # DICOM series usually stack on axis 0
            segment_all = kwargs.get('segment_all_slices', False)

        # 3D Processing Loop
        if segment_all and volume.ndim == 3:
            result_vol = np.zeros_like(volume, dtype=volume.dtype)
            
            # Iterate based on axis
            # Note: We swap axes to make iteration uniform, then swap back
            vol_permuted = np.swapaxes(volume, 0, slice_axis)
            res_permuted = np.swapaxes(result_vol, 0, slice_axis)
            
            for i in range(vol_permuted.shape[0]):
                res_permuted[i] = self.segment_image(
                    vol_permuted[i], 
                    method=method, 
                    remove_background=False, 
                    morphological_operations=False,
                    preserve_intensities=kwargs.get('preserve_intensities', False)
                )
            
            return result_vol
        
        # Single Slice Extraction
        else:
            idx = kwargs.get('slice_idx')
            if idx is None: 
                idx = volume.shape[slice_axis] // 2
                
            if slice_axis == 0: slice_img = volume[idx, :, :]
            elif slice_axis == 1: slice_img = volume[:, idx, :]
            else: slice_img = volume[:, :, idx]
            
            return self.segment_image(slice_img, method=method, remove_background=False)