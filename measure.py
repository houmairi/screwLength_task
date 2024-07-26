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
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the screw)
    largest_contour = max(contours, key=cv2.contourArea)
    
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