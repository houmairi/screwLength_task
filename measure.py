import cv2
import numpy as np
import os
from scipy.stats import describe

def measure_screw_length(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
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
    box = np.int0(box)
    
    # Calculate the length (maximum dimension of the rotated rectangle)
    length = max(rect[1])
    
    return length

def process_folder(folder_path):
    lengths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            length = measure_screw_length(image_path)
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
    folder_path = ""  # Paste folder location here
    screw_lengths = process_folder(folder_path)
    analyze_results(screw_lengths)