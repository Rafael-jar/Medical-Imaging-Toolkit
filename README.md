# Medical Image Processing Toolkit 🏥

A comprehensive Python toolkit for medical image processing showcasing skills in medical imaging, computer vision, and scientific computing.

## 🚀 Features

- **Image Enhancement**: Auto-level, gamma correction, denoising, saturation adjustment, histogram equalization, watermarking
- **Segmentation**: Otsu/optimal thresholding, watershed, region growing, morphological operations, shape identification
- **Registration**: Rigid, affine, non-rigid (SyN) using ANTsPy/SimpleITK for mono-modal and multi-modal alignment
- **Metrics**: MSE, NCC, MI, NMI, SSIM, Dice coefficient, Jaccard index
- **Medical I/O**: DICOM and NIfTI file support with metadata extraction

## 🛠️ Installation
Just clone the repository and install the required dependencies using:
```bash
pip install -r requirements.txt
```
## 🔧 Design decisions
- I decided to implement some algorithms from scratch (e.g., Otsu's method, watershed segmentation) to demonstrate my understanding of the underlying principles.
- When implementing the segmentation of a Nifti image I chose to use a 2.5D approach (slice-by-slice) instead of a full 3D approach to balance performance and computational efficiency.
- I implemented a MedicalImageProcessor that inherits from functional classes (Mixins) to make the usage more intuitive and user-friendly.

## 📚 Examples
See the [Usage Examples Notebook](Usage_Examples.ipynb) for detailed demonstrations of the toolkit's capabilities.

### Image example sources
- The Brain MRI image .NII is from: https://github.com/neurolabusc/niivue-images/blob/main/fmri_pitch.nii.gz

- The astronaut comes from the library skimage: `from skimage import data; image = data.astronaut()`

- The mri_brain.jpg image is from: https://www.researchgate.net/figure/Sample-MRI-scans-marked-as-positive-brain-tumor-by-trained-clinicians_fig12_338115876


## 👤 Author

**Rafael Palma Santos**
