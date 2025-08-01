# SpaceObjectDetection
# Space Object Detection Toolkit

Lightweight framework for detecting sources in space / astronomical images using classical methods.

## Features

- Supports image formats: FITS, TIFF, PNG, JPEG, BMP.  
- Detectors:
  - **OtsuDetector**: Global Otsu threshold + connected component analysis.  
  - **SExtractor** (via `sep`): Background estimation and source extraction. 
  - **PoissonThresholding**: Minimum-error thresholding (Gaussian/Poisson) + region extraction.  
- Unified output per detection: `[cx, cy, area, w, h, gray_sum]`.  
- Built-in filtering (size and aspect ratio), visualization (PNG), and result saving (aligned TXT).


