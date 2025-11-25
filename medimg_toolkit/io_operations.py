
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from .core import MedicalImageBase
import nibabel as nib
import pydicom

class MedicalImageIO(MedicalImageBase):
    """
    File I/O operations for medical imaging formats.
    Supports DICOM and NIfTI formats.
    """
    
    def load_dicom(self, filepath: str, return_metadata: bool = False):
        """
        Load DICOM image file
        
        Parameters:
        -----------
        filepath : str
            Path to DICOM file
        return_metadata : bool
            If True, return (image, metadata) tuple
            
        Returns:
        --------
        np.ndarray or tuple
            Image array, or (image, metadata) if return_metadata=True
        """
        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array
        
        if return_metadata:
            metadata = self.extract_dicom_metadata(dcm)
            return img, metadata
        
        return img
    
    def load_dicom_series(self, directory: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a series of DICOM files as a 3D volume
        
        Parameters:
        -----------
        directory : str
            Directory containing DICOM files
            
        Returns:
        --------
        tuple
            (volume, metadata) where volume is 3D numpy array
        """ 
        dir_path = Path(directory)
        dicom_files = sorted(list(dir_path.glob("*.dcm")))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {directory}")

        # Load first file to get dimensions
        first_dcm = pydicom.dcmread(str(dicom_files[0]))
        img_shape = first_dcm.pixel_array.shape
        
        # Initialize volume
        volume = np.zeros((len(dicom_files), *img_shape), dtype=first_dcm.pixel_array.dtype)
        
        # Load all slices
        for i, filepath in enumerate(dicom_files):
            dcm = pydicom.dcmread(str(filepath))
            volume[i] = dcm.pixel_array
        
        metadata = self.extract_dicom_metadata(first_dcm)
        metadata['num_slices'] = len(dicom_files)
        
        return volume, metadata
    
    def extract_dicom_metadata(self, dcm) -> Dict:
        """
        Extract useful metadata from DICOM file
        
        Parameters:
        -----------
        dcm : pydicom.dataset.FileDataset
            DICOM dataset
            
        Returns:
        --------
        dict
            Dictionary of metadata
        """
        metadata = {}
        
        # Patient information
        metadata['patient_id'] = str(dcm.get('PatientID', 'Unknown'))
        metadata['patient_age'] = str(dcm.get('PatientAge', 'Unknown'))
        metadata['patient_sex'] = str(dcm.get('PatientSex', 'Unknown'))
        
        # Study information
        metadata['study_date'] = str(dcm.get('StudyDate', 'Unknown'))
        metadata['modality'] = str(dcm.get('Modality', 'Unknown'))
        metadata['study_description'] = str(dcm.get('StudyDescription', 'Unknown'))
        
        # Image information
        metadata['rows'] = int(dcm.get('Rows', 0))
        metadata['columns'] = int(dcm.get('Columns', 0))
        metadata['pixel_spacing'] = dcm.get('PixelSpacing', [1.0, 1.0])
        metadata['slice_thickness'] = float(dcm.get('SliceThickness', 1.0))
        
        # Acquisition parameters
        metadata['kvp'] = float(dcm.get('KVP', 0))
        metadata['exposure'] = float(dcm.get('Exposure', 0))
        
        return metadata
    
    def load_nifti(self, filepath: str, return_metadata: bool = False):
        """
        Load NIfTI image file
        
        Parameters:
        -----------
        filepath : str
            Path to NIfTI file (.nii or .nii.gz)
        return_metadata : bool
            If True, return (image, metadata) tuple
            
        Returns:
        --------
        np.ndarray or tuple
            Image array, or (image, metadata) if return_metadata=True
        """
        nii = nib.load(filepath)
        img = nii.get_fdata()
        
        if return_metadata:
            metadata = self.extract_nifti_metadata(nii)
            return img, metadata
        
        return img
    
    def extract_nifti_metadata(self, nii) -> Dict:
        """
        Extract metadata from NIfTI file
        
        Parameters:
        -----------
        nii : nibabel.nifti1.Nifti1Image
            NIfTI image object
            
        Returns:
        --------
        dict
            Dictionary of metadata
        """
        header = nii.header
        
        metadata = {
            'shape': nii.shape,
            'affine': nii.affine.tolist(),
            'voxel_size': header.get_zooms(),
            'data_type': str(header.get_data_dtype()),
            'dimension': header['dim'][0],
            'description': header.get('descrip', b'')
        }
        
        return metadata
    
    def save_nifti(self, img: np.ndarray, filepath: str, affine: Optional[np.ndarray] = None):
        """
        Save image as NIfTI file
        
        Parameters:
        -----------
        img : np.ndarray
            Image data
        filepath : str
            Output file path
        affine : np.ndarray or None
            Affine transformation matrix (uses identity if None)
        """
        if affine is None:
            affine = np.eye(4)
        
        nii = nib.Nifti1Image(img, affine)
        nib.save(nii, filepath)
    
    def convert_dicom_to_nifti(self, dicom_dir: str, output_path: str):
        """
        Convert DICOM series to NIfTI file
        
        Parameters:
        -----------
        dicom_dir : str
            Directory containing DICOM files
        output_path : str
            Output NIfTI file path
        """
        # Load DICOM series
        volume, metadata = self.load_dicom_series(dicom_dir)
        
        # Create affine matrix from DICOM metadata
        pixel_spacing = metadata.get('pixel_spacing', [1.0, 1.0])
        slice_thickness = metadata.get('slice_thickness', 1.0)
        
        affine = np.eye(4)
        affine[0, 0] = pixel_spacing[0]
        affine[1, 1] = pixel_spacing[1]
        affine[2, 2] = slice_thickness
        
        # Save as NIfTI
        self.save_nifti(volume, output_path, affine)
    
    def get_image_info(self, filepath: str) -> Dict:
        """
        Get basic information about a medical image file
        
        Parameters:
        -----------
        filepath : str
            Path to image file
            
        Returns:
        --------
        dict
            Dictionary containing image information
        """
        filepath_lower = filepath.lower()
        
        if filepath_lower.endswith('.dcm'):
            img, metadata = self.load_dicom(filepath, return_metadata=True)
            info = {
                'format': 'DICOM',
                'shape': img.shape,
                'dtype': str(img.dtype),
                'min_value': float(img.min()),
                'max_value': float(img.max()),
                'mean_value': float(img.mean()),
                **metadata
            }
        elif filepath_lower.endswith(('.nii', '.nii.gz')):
            img, metadata = self.load_nifti(filepath, return_metadata=True)
            info = {
                'format': 'NIfTI',
                'shape': img.shape,
                'dtype': str(img.dtype),
                'min_value': float(img.min()),
                'max_value': float(img.max()),
                'mean_value': float(img.mean()),
                **metadata
            }
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        return info
