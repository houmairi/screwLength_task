import cv2
import numpy as np
import os
from scipy.stats import describe

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    
    # Apply threshold
    _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned

def get_screw_contour(preprocessed):
    # Find edges
    edges = cv2.Canny(preprocessed, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this threshold based on your images
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.1 < aspect_ratio < 10:  # Adjust these thresholds as needed
                valid_contours.append(contour)
    
    if valid_contours:
        return max(valid_contours, key=cv2.contourArea)
    return None

def measure_screw_length(image_path, output_folder):
    img = cv2.imread(image_path)
    original = img.copy()
    preprocessed = preprocess_image(img)
    
    screw_contour = get_screw_contour(preprocessed)
    
    if screw_contour is not None:
        # Perform PCA to find the main axis
        points = screw_contour.reshape(-1, 2).astype(np.float32)
        
        if points.shape[0] > 2:  # Ensure we have enough points for PCA
            mean, eigenvectors = cv2.PCACompute(points, mean=None)
            
            # Get the center of the contour
            center = tuple(mean[0].astype(int))
            
            # Calculate the endpoints of the main axis
            axis_length = 1000  # Arbitrary large value
            direction = eigenvectors[0]
            point1 = tuple((center + axis_length * direction).astype(int))
            point2 = tuple((center - axis_length * direction).astype(int))
            
            # Find intersections with the contour
            intersections = []
            for t in np.linspace(0, 1, 1000):
                x = int(point1[0] * (1-t) + point2[0] * t)
                y = int(point1[1] * (1-t) + point2[1] * t)
                point = (x, y)
                if cv2.pointPolygonTest(screw_contour, point, False) >= 0:
                    intersections.append(point)
            
            if len(intersections) >= 2:
                start_point, end_point = intersections[0], intersections[-1]
                length = np.linalg.norm(np.array(end_point) - np.array(start_point))
                
                # Draw the line on the original image
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)
                
                # Put text with length information
                cv2.putText(img, f"Length: {length:.2f} px", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                length = 0
                cv2.putText(img, "Measurement failed", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            length = 0
            cv2.putText(img, "Not enough points for measurement", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        length = 0
        cv2.putText(img, "No screw detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
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