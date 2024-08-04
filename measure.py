import cv2
import numpy as np
import os
from scipy.stats import describe

class ScrewMeasurement:
    """
    A class for measuring screw lengths in images.

    The main algorithm consists of the following steps:
    1. Image preprocessing
    2. Screw detection
    3. Length measurement
    4. Visualization and statistical analysis

    Attributes:
        min_length (float): Minimum length threshold for valid screw measurements.
    """

    def __init__(self, min_length=500):
        self.min_length = min_length

    def preprocess_image(self, image):
        """
        Preprocess the input image.

        Steps:
        1. Convert to grayscale
        2. Enhance contrast using CLAHE
        3. Apply binary thresholding
        4. Perform morphological operations to reduce noise

        Args:
            image (numpy.ndarray): Input image in BGR format.

        Returns:
            numpy.ndarray: Preprocessed binary image.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # 3. Apply threshold
        _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. Morphological operations
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return cleaned

    def detect_screw(self, preprocessed):
        """
        Detect the screw in the preprocessed image.

        Steps:
        1. Perform edge detection
        2. Find contours
        3. Filter contours based on area and aspect ratio

        Args:
            preprocessed (numpy.ndarray): Preprocessed binary image.

        Returns:
            numpy.ndarray: Contour of the detected screw, or None if no valid screw is found.
        """
        # 1. Find edges
        edges = cv2.Canny(preprocessed, 50, 150)
        
        # 2. Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Filter contours
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.1 < aspect_ratio < 10:
                    valid_contours.append(contour)
        
        return max(valid_contours, key=cv2.contourArea) if valid_contours else None

    def measure_length(self, contour):
        """
        Measure the length of the screw using its contour.

        Steps:
        1. Perform PCA to find the main axis
        2. Calculate endpoints of the main axis
        3. Find intersections with the contour
        4. Calculate the length

        Args:
            contour (numpy.ndarray): Contour of the screw.

        Returns:
            tuple: (length, start_point, end_point) if successful, (0, None, None) otherwise.
        """
        # 1. Perform PCA
        points = contour.reshape(-1, 2).astype(np.float32)
        if points.shape[0] <= 2:
            return 0, None, None

        mean, eigenvectors = cv2.PCACompute(points, mean=None)
        center = tuple(mean[0].astype(int))
        
        # 2. Calculate endpoints
        axis_length = max(contour.shape) * 2
        direction = eigenvectors[0]
        point1 = tuple((center + axis_length * direction).astype(int))
        point2 = tuple((center - axis_length * direction).astype(int))
        
        # 3. Find intersections
        intersections = []
        for t in np.linspace(0, 1, 2000):
            x = int(point1[0] * (1-t) + point2[0] * t)
            y = int(point1[1] * (1-t) + point2[1] * t)
            point = (x, y)
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                intersections.append(point)
        
        # 4. Calculate length
        if len(intersections) >= 2:
            start_point, end_point = intersections[0], intersections[-1]
            length = np.linalg.norm(np.array(end_point) - np.array(start_point))
            return length, start_point, end_point
        
        return 0, None, None

    def process_image(self, image_path, output_folder):
        """
        Process a single image to measure the screw length.

        Steps:
        1. Read and preprocess the image
        2. Detect the screw
        3. Measure the screw length
        4. Visualize the results

        Args:
            image_path (str): Path to the input image.
            output_folder (str): Path to the output folder for processed images.

        Returns:
            float: Measured length of the screw, or 0 if no valid screw is detected.
        """
        # 1. Read and preprocess
        img = cv2.imread(image_path)
        original = img.copy()
        preprocessed = self.preprocess_image(img)
        
        # Create a colored mask for visualization
        mask_colored = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        
        # 2. Detect screw
        screw_contour = self.detect_screw(preprocessed)
        
        length = 0
        if screw_contour is not None:
            # 3. Measure length
            length, start_point, end_point = self.measure_length(screw_contour)
            
            # 4. Visualize
            if length >= self.min_length:
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)
                cv2.line(mask_colored, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(img, f"Length: {length:.2f} px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(mask_colored, f"Length: {length:.2f} px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(img, f"Length below threshold: {length:.2f} px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(mask_colored, f"Length below threshold: {length:.2f} px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "No screw detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(mask_colored, "No screw detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save the processed image with mask
        comparison = np.hstack((original, img, mask_colored))
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"processed_{filename}")
        cv2.imwrite(output_path, comparison)
        
        return length if length >= self.min_length else 0

    def process_folder(self, input_folder, output_folder):
        """
        Process all images in a folder and collect screw lengths.

        Args:
            input_folder (str): Path to the input folder containing images.
            output_folder (str): Path to the output folder for processed images.

        Returns:
            list: List of valid screw lengths.
        """
        lengths = []
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                length = self.process_image(image_path, output_folder)
                if length > 0:
                    lengths.append(length)
                    print(f"Processed {filename}: Length = {length:.2f} pixels")
                else:
                    print(f"Processed {filename}: Length below threshold or no screw detected")
        
        return lengths

    @staticmethod
    def analyze_results(lengths):
        """
        Perform statistical analysis on the measured screw lengths.

        Args:
            lengths (list): List of screw lengths.

        Returns:
            None
        """
        if lengths:
            stats = describe(lengths)
            print("\nStatistical Analysis:")
            print(f"Number of screws measured: {stats.nobs}")
            print(f"Mean length: {stats.mean:.2f} pixels")
            print(f"Standard deviation: {np.sqrt(stats.variance):.2f} pixels")
            print(f"Minimum length: {stats.minmax[0]:.2f} pixels")
            print(f"Maximum length: {stats.minmax[1]:.2f} pixels")
            
            cv = (np.sqrt(stats.variance) / stats.mean) * 100
            print(f"Coefficient of Variation: {cv:.2f}%")
        else:
            print("\nNo valid screws detected in the dataset.")

def main():
    input_folder = "Schrauben/"
    output_folder = "processed_screws"
    min_length = 500 # in px

    screw_measurement = ScrewMeasurement(min_length)
    screw_lengths = screw_measurement.process_folder(input_folder, output_folder)
    ScrewMeasurement.analyze_results(screw_lengths)

if __name__ == "__main__":
    main()