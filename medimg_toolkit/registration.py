
import numpy as np
from typing import Dict, Optional
from .core import MedicalImageBase
import ants
import SimpleITK as sitk


class ImageRegistrar(MedicalImageBase):
    """
    Image registration operations for medical images.
    Supports both ANTsPy and SimpleITK backends.
    """
    
    def register_images_ants(self, fixed_img, moving_img, 
                            transform_type: str = 'Affine') -> Dict:
        """
        Register images using ANTsPy
        
        Parameters:
        -----------
        fixed_img : ANTsImage or str
            Fixed (reference) image or path
        moving_img : ANTsImage or str
            Moving image to align or path
        transform_type : str
            Type of transformation:
            - 'Rigid': Translation + rotation
            - 'Affine': Rigid + scaling + shearing
            - 'SyN': Non-rigid symmetric normalization
            - 'ElasticSyN': Elastic deformation
            - 'SyNRA': SyN with Rigid and Affine pre-alignment
            
        Returns:
        --------
        dict
            Registration result containing warped image and transforms
        """
        # Load images if paths provided
        if isinstance(fixed_img, str):
            fixed_img = ants.image_read(fixed_img)
        if isinstance(moving_img, str):
            moving_img = ants.image_read(moving_img)
        

        result = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform=transform_type
        )
        

        return result
    
    def register_images_sitk(self, fixed_img_path: str, moving_img_path: str,
                            transform_type: str = 'affine') -> 'sitk.Image':
        """
        Register images using SimpleITK (3D-to-3D registration)
        
        Parameters:
        -----------
        fixed_img_path : str
            Path to fixed image
        moving_img_path : str
            Path to moving image
        transform_type : str
            Type of transformation ('rigid', 'affine', 'bspline')
            
        Returns:
        --------
        sitk.Image
            Registered moving image
        """
        fixed_img = sitk.ReadImage(fixed_img_path)
        moving_img = sitk.ReadImage(moving_img_path)
        
        fixed_img = sitk.Cast(fixed_img, sitk.sitkFloat32)
        moving_img = sitk.Cast(moving_img, sitk.sitkFloat32)
        
        # Configure sitk using 50 histogram bins for MI
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Select transform type
        if transform_type.lower() == 'rigid':
            transform = sitk.Euler3DTransform()
        elif transform_type.lower() == 'affine':
            transform = sitk.AffineTransform(3)
        elif transform_type.lower() == 'bspline':
            transform_domain_mesh_size = [8] * moving_img.GetDimension()
            transform = sitk.BSplineTransformInitializer(fixed_img, transform_domain_mesh_size)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img,
            moving_img,
            transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        final_transform = registration_method.Execute(fixed_img, moving_img)
        
        resampled_img = sitk.Resample(
            moving_img,
            fixed_img,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_img.GetPixelID()
        )
        
        return resampled_img
    
    def register_multimodal(self, fixed_img, moving_img, 
                           backend: str = 'ants') -> Dict:
        """
        Perform multi-modal registration (e.g., CT to MRI)
        
        Parameters:
        -----------
        fixed_img : ANTsImage, str, or sitk.Image
            Fixed image (e.g., MRI)
        moving_img : ANTsImage, str, or sitk.Image
            Moving image (e.g., CT)
        backend : str
            Registration backend ('ants' or 'sitk')
            
        Returns:
        --------
        dict or sitk.Image
            Registration result
        """

        if backend.lower() == 'ants':
            # ANTs automatically uses MI for multi-modal
            return self.register_images_ants(fixed_img, moving_img, 
                                            transform_type='Affine')
        elif backend.lower() == 'sitk':
            # SimpleITK uses MI by default
            if isinstance(fixed_img, str) and isinstance(moving_img, str):
                return self.register_images_sitk(fixed_img, moving_img,
                                                transform_type='affine')
            else:
                raise ValueError("SimpleITK backend requires file paths")
        else:
            raise ValueError(f"Unknown backend: {backend}")
