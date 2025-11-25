import numpy as np
from typing import Dict, Tuple
from skimage.metrics import structural_similarity as ssim
from .core import MedicalImageBase


class SimilarityMetrics(MedicalImageBase):
    """
    Compute various similarity metrics between images.
    Used for evaluating registration and segmentation quality.
    """

    @staticmethod
    def convert_to_float(img: np.ndarray) -> np.ndarray:
        """Convert image to flat float array"""
        return img.ravel().astype(float) if img.dtype == bool else img.ravel()
    
    def compute_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Mean Squared Error between two images
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Input images (same shape)
            
        Returns:
        --------
        float
            MSE value (lower is better, 0 = identical)
        """
        return np.mean((self.convert_to_float(img1) - self.convert_to_float(img2)) ** 2)
    
    def compute_mae(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Mean Absolute Error between two images
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Input images (same shape)
            
        Returns:
        --------
        float
            MAE value (lower is better, 0 = identical)
        """
        return np.mean(np.abs(self.convert_to_float(img1) - self.convert_to_float(img2)))
    
    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between two images
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Input images (same shape)
            
        Returns:
        --------
        float
            PSNR value in dB (higher is better, inf = identical)
        """
        img1_numeric = self.convert_to_float(img1)
        img2_numeric = self.convert_to_float(img2)
        
        mse = self.compute_mse(img1, img2)
        if mse == 0:
            return float('inf')
        
        max_pixel = max(img1_numeric.max(), img2_numeric.max())
        if max_pixel == 0:
            return float('inf')
        
        return 20 * np.log10(max_pixel / np.sqrt(mse))
    
    def compute_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Normalized Cross-Correlation between two images
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Input images (same shape)
            
        Returns:
        --------
        float
            NCC value (closer to 1 is better, range: -1 to 1)
        """
        flat1 = self.convert_to_float(img1)
        flat2 = self.convert_to_float(img2)
        
        norm1 = flat1 - np.mean(flat1)
        norm2 = flat2 - np.mean(flat2)
        
        denom = np.std(flat1) * np.std(flat2) * len(flat1)
        if denom == 0:
            return 0
        
        return np.sum(norm1 * norm2) / denom
    
    def compute_mutual_information(self, img1: np.ndarray, img2: np.ndarray, 
                                   bins: int = 64) -> Tuple[float, float]:
        """
        Compute Mutual Information and Normalized Mutual Information
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Input images (same shape)
        bins : int
            Number of histogram bins
            
        Returns:
        --------
        tuple
            (MI, NMI) values (higher is better)
        """
        flat1 = self.convert_to_float(img1)
        flat2 = self.convert_to_float(img2)
        
        hist_2d, _, _ = np.histogram2d(flat1, flat2, bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
        hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
        
        mi = hx + hy - hxy
        nmi = (hx + hy) / hxy if hxy != 0 else 0
        
        return mi, nmi
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Input images (same shape)
            
        Returns:
        --------
        float
            SSIM value (closer to 1 is better, range: -1 to 1)
        """
        img1_numeric = self.convert_to_float(img1)
        img2_numeric = self.convert_to_float(img2)
        
        data_range = img1_numeric.max() - img1_numeric.min()
        return ssim(img1_numeric, img2_numeric, data_range=data_range)
    
    def compute_dice_coefficient(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Dice Similarity Coefficient (for binary masks)
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Binary images (same shape)
            
        Returns:
        --------
        float
            Dice coefficient (0 to 1, 1 = perfect match)
        """
        img1_bool = img1.astype(bool)
        img2_bool = img2.astype(bool)
        
        intersection = np.logical_and(img1_bool, img2_bool).sum()
        size_i1 = img1_bool.sum()
        size_i2 = img2_bool.sum()
        
        if size_i1 + size_i2 == 0:
            return 1.0
        
        return 2. * intersection / (size_i1 + size_i2)
    
    def compute_jaccard_index(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Jaccard Index / IoU (Intersection over Union)
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Binary images (same shape)
            
        Returns:
        --------
        float
            Jaccard index (0 to 1, 1 = perfect match)
        """
        img1_bool = img1.astype(bool)
        img2_bool = img2.astype(bool)
        
        intersection = np.logical_and(img1_bool, img2_bool).sum()
        union = np.logical_or(img1_bool, img2_bool).sum()
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    def compute_all_metrics(self, img1: np.ndarray, img2: np.ndarray,
                           binary: bool = False) -> Dict[str, float]:
        """
        Compute all similarity metrics between two images
        
        Parameters:
        -----------
        img1, img2 : np.ndarray
            Input images (same shape)
        binary : bool
            If True, also compute binary metrics (Dice, Jaccard)
            
        Returns:
        --------
        dict
            Dictionary of all computed metrics
        """
        mi, nmi = self.compute_mutual_information(img1, img2)
        
        metrics = {
            'MSE': self.compute_mse(img1, img2),
            'MAE': self.compute_mae(img1, img2),
            'PSNR': self.compute_psnr(img1, img2),
            'NCC': self.compute_ncc(img1, img2),
            'MI': mi,
            'NMI': nmi,
            'SSIM': self.compute_ssim(img1, img2)
        }
        
        if binary:
            metrics['Dice'] = self.compute_dice_coefficient(img1, img2)
            metrics['Jaccard'] = self.compute_jaccard_index(img1, img2)
        
        return metrics
    
    def interpret_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Provide interpretation of computed metrics
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metric values
            
        Returns:
        --------
        str
            Human-readable interpretation
        """
        interpretation = []
        interpretation.append("Metric Interpretation:")
        interpretation.append("-" * 40)
        
        if 'MSE' in metrics:
            mse = metrics['MSE']
            if mse < 50:
                quality = "Excellent"
            elif mse < 200:
                quality = "Good"
            elif mse < 500:
                quality = "Fair"
            else:
                quality = "Poor"
            interpretation.append(f"MSE = {mse:.2f} ({quality} - lower is better)")
        
        if 'NCC' in metrics:
            ncc = metrics['NCC']
            if ncc > 0.9:
                quality = "Excellent"
            elif ncc > 0.7:
                quality = "Good"
            elif ncc > 0.5:
                quality = "Fair"
            else:
                quality = "Poor"
            interpretation.append(f"NCC = {ncc:.3f} ({quality} - closer to 1 is better)")
        
        if 'SSIM' in metrics:
            ssim_val = metrics['SSIM']
            if ssim_val > 0.9:
                quality = "Excellent"
            elif ssim_val > 0.7:
                quality = "Good"
            elif ssim_val > 0.5:
                quality = "Fair"
            else:
                quality = "Poor"
            interpretation.append(f"SSIM = {ssim_val:.3f} ({quality} - closer to 1 is better)")
        
        if 'Dice' in metrics:
            dice = metrics['Dice']
            if dice > 0.8:
                quality = "Excellent"
            elif dice > 0.6:
                quality = "Good"
            elif dice > 0.4:
                quality = "Fair"
            else:
                quality = "Poor"
            interpretation.append(f"Dice = {dice:.3f} ({quality} overlap)")
        
        return "\n".join(interpretation)
