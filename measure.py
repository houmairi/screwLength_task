import cv2
import numpy as np
import os
from scipy.stats import describe

def measure_screw_length(image_path, output_folder):
    # Read the image
    img = cv2.imread(image_path)
    original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate the edges to connect them
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Adjust this threshold based on your images
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.1 < aspect_ratio < 10:  # Adjust these thresholds as needed
                valid_contours.append(cnt)
    
    # Find the largest valid contour
    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Get the rotated rectangle of the contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int_)
        
        # Calculate the length (maximum dimension of the rotated rectangle)
        length = max(rect[1])
        
        # Draw the rotated rectangle
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        
        # Put text with length information
        cv2.putText(img, f"Length: {length:.2f} pixels", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
            lengths.append(length)
            print(f"Processed {filename}: Length = {length:.2f} pixels")
    
    return lengths

def analyze_results(lengths):
    stats = describe(lengths)
    print("\nStatistical Analysis:")
    print(f"Number of screws: {stats.nobs}")
    print(f"Mean length: {stats.mean:.2f} pixels")
    print(f"Standard deviation: {np.sqrt(stats.variance):.2f} pixels")
    print(f"Minimum length: {stats.minmax[0]:.2f} pixels")
    print(f"Maximum length: {stats.minmax[1]:.2f} pixels")

if __name__ == "__main__":
    input_folder = "Schrauben/"  # Replace with the actual input path
    output_folder = "processed"  # This will create a 'processed' folder in your current directory
    screw_lengths = process_folder(input_folder, output_folder)
    analyze_results(screw_lengths)