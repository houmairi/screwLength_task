import cv2
import numpy as np
import os
from scipy.stats import describe

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def measure_screw_length(image_path, output_folder):
    img = cv2.imread(image_path)
    original = img.copy()
    preprocessed = preprocess_image(img)
    
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Get the rotated rectangle of the contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Project all points onto the fitted line
        projected_points = []
        for point in box:
            t = (point[0] - x) * vx + (point[1] - y) * vy
            projected_point = (int(x + t * vx), int(y + t * vy))
            projected_points.append(projected_point)
        
        # Find the two extreme points
        start_point = min(projected_points, key=lambda p: p[0] * vx + p[1] * vy)
        end_point = max(projected_points, key=lambda p: p[0] * vx + p[1] * vy)
        
        # Calculate the length
        length = np.linalg.norm(np.array(end_point) - np.array(start_point))
        
        # Draw the line on the original image
        cv2.line(img, start_point, end_point, (0, 255, 0), 2)
        
        # Put text with length information
        cv2.putText(img, f"Length: {length:.2f} px", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        length = 0
        cv2.putText(img, "No screw detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Create side-by-side comparison
    comparison = np.hstack((original, img))
    
    # Save the processed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"processed_{filename}")
    cv2.imwrite(output_path, comparison)
    
    return length

def process_folder(input_folder, output_folder):
    lengths = []
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            length = measure_screw_length(image_path, output_folder)
            if length > 0:
                lengths.append(length)
                print(f"Processed {filename}: Length = {length:.2f} pixels")
            else:
                print(f"Processed {filename}: No screw detected")
    
    return lengths

def analyze_results(lengths):
    if lengths:
        stats = describe(lengths)
        print("\nStatistical Analysis:")
        print(f"Number of screws measured: {stats.nobs}")
        print(f"Mean length: {stats.mean:.2f} pixels")
        print(f"Standard deviation: {np.sqrt(stats.variance):.2f} pixels")
        print(f"Minimum length: {stats.minmax[0]:.2f} pixels")
        print(f"Maximum length: {stats.minmax[1]:.2f} pixels")
        
        # Calculate coefficient of variation (CV)
        cv = (np.sqrt(stats.variance) / stats.mean) * 100
        print(f"Coefficient of Variation: {cv:.2f}%")
    else:
        print("\nNo valid screws detected in the dataset.")

if __name__ == "__main__":
    input_folder = "Schrauben/"  # Replace with the actual input path
    output_folder = "processed"  # This will create a 'processed' folder in your current directory
    screw_lengths = process_folder(input_folder, output_folder)
    analyze_results(screw_lengths)