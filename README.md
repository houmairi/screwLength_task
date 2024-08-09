# Screw Measurement Project

## Overview
This project was developed as part of a job application to demonstrate Computer Vision skills. It implements an automated system for measuring screw lengths in images using OpenCV and Python.

## Features
- Image preprocessing for enhanced screw detection
- Screw detection using contour analysis
- Accurate length measurement using Principal Component Analysis (PCA)
- Visualization of detected screws and measurements
- Statistical analysis of screw lengths across multiple images

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
- SciPy

## Installation
1. Clone this repository
2. Install the required packages:
   ```
   pip install opencv-python numpy scipy
   ```

## Usage
1. Place your screw images in the `Schrauben/` directory
2. Run the script:
   ```
   python screw_measurement.py
   ```
3. Processed images will be saved in the `processed_screws/` directory
4. Statistical analysis of the measurements will be printed to the console

## Customization
The script includes several adjustable parameters that can be modified to optimize performance for different types of screws or image conditions:
- Minimum length threshold
- CLAHE parameters for contrast enhancement
- Thresholding method
- Morphological operation kernel size
- Contour filtering criteria
- PCA axis length multiplier

Refer to the comments in the code for details on adjusting these parameters.

## Project Structure
- `ScrewMeasurement` class: Main class containing the image processing and measurement logic
- `preprocess_image`: Image preprocessing function
- `detect_screw`: Screw detection function
- `measure_length`: Length measurement function
- `process_image`: Single image processing function
- `process_folder`: Batch processing function for multiple images
- `analyze_results`: Statistical analysis function

## Future Improvements
- Implement a graphical user interface for easier use
- Add support for calibration to convert pixel measurements to real-world units
- Enhance robustness for different lighting conditions and screw types

## Author
houmairi

## License
This project is open-source and available under the MIT License.