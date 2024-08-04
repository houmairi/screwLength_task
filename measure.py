import cv2
import numpy as np
import os
from scipy.stats import describe

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
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
    edges = cv2.Canny(preprocessed, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.1 < aspect_ratio < 10:
                valid_contours.append(contour)
    
    if valid_contours:
        return max(valid_contours, key=cv2.contourArea)
    return None

def find_screw_head(contour):
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Create a mask from the contour
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1, offset=(-x, -y))
    
    # Calculate the width profile
    width_profile = np.sum(mask, axis=1)
    
    # Find the point of maximum width
    max_width_index = np.argmax(width_profile)
    
    # Find the edge of the screw head (where width starts decreasing significantly)
    head_edge = max_width_index
    for i in range(max_width_index, h):
        if width_profile[i] < 0.9 * width_profile[max_width_index]:
            head_edge = i
            break
    
    return head_edge + y, mask

#def align_screw(image, contour):
#    # Fit an ellipse to the screw contour
#    ellipse = cv2.fitEllipse(contour)
#    angle = ellipse[2]
#
#    # Get the image dimensions
#    h, w = image.shape[:2]
#
#    # Calculate the rotation matrix
#    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
#
#    # Perform the rotation
#    rotated_image = cv2.warpAffine(image, M, (w, h))
#    rotated_contour = cv2.transform(contour.reshape(1, -1, 2), M).reshape(-1, 2)
#
#    # Check if the screw head is at the bottom
#    head_at_top, mask = find_screw_head(rotated_image, rotated_contour)
#
#    if not head_at_top:
#        # If the head is at the bottom, rotate by 180 degrees
#        M = cv2.getRotationMatrix2D((w//2, h//2), angle + 180, 1.0)
#        rotated_image = cv2.warpAffine(image, M, (w, h))
#        rotated_contour = cv2.transform(contour.reshape(1, -1, 2), M).reshape(-1, 2)
#        _, mask = find_screw_head(rotated_image, rotated_contour)
#
#    return rotated_image, rotated_contour, mask

def measure_screw(image_path, output_folder, pixels_per_mm):
    img = cv2.imread(image_path)
    original = img.copy()
    preprocessed = preprocess_image(img)

    screw_contour = get_screw_contour(preprocessed)
    
    if screw_contour is not None:
        # Find the screw head
        head_edge_y, mask = find_screw_head(screw_contour)
        
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(screw_contour)
        
        # Calculate screw length
        length_px = h - (head_edge_y - y)
        length_mm = length_px / pixels_per_mm
        
        # Draw the head edge line and length line
        cv2.line(img, (x, head_edge_y), (x + w, head_edge_y), (0, 255, 0), 2)
        cv2.line(img, (x + w//2, head_edge_y), (x + w//2, y + h), (0, 0, 255), 2)
        
        # Put text with length information
        cv2.putText(img, f"Length: {length_mm:.2f} mm", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        length_mm = 0
        cv2.putText(img, "No screw detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Create side-by-side comparison with mask
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((original, img, mask_colored))
    
    # Save the processed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"processed_{filename}")
    cv2.imwrite(output_path, comparison)
    
    return length_mm

def process_folder(input_folder, output_folder, pixels_per_mm):
    lengths = []
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            length = measure_screw(image_path, output_folder, pixels_per_mm)
            if length > 0:
                lengths.append(length)
                print(f"Processed {filename}: Length = {length:.2f} mm")
            else:
                print(f"Processed {filename}: No screw detected")
    
    return lengths

def analyze_results(lengths):
    if lengths:
        stats = describe(lengths)
        print("\nStatistical Analysis:")
        print(f"Number of screws measured: {stats.nobs}")
        print(f"Mean length: {stats.mean:.2f} mm")
        print(f"Standard deviation: {np.sqrt(stats.variance):.2f} mm")
        print(f"Minimum length: {stats.minmax[0]:.2f} mm")
        print(f"Maximum length: {stats.minmax[1]:.2f} mm")
        
        # Calculate coefficient of variation (CV)
        cv = (np.sqrt(stats.variance) / stats.mean) * 100
        print(f"Coefficient of Variation: {cv:.2f}%")
    else:
        print("\nNo valid screws detected in the dataset.")

if __name__ == "__main__":
    input_folder = "Schrauben/"  # Replace with the actual input path
    output_folder = "processed"  # This will create a 'processed' folder in your current directory
    
    # Calculate pixels per mm based on the metadata
    image_width_px = 1024
    x_resolution_dpi = 72.0
    inches_per_mm = 1 / 25.4
    pixels_per_mm = x_resolution_dpi * inches_per_mm
    
    screw_lengths = process_folder(input_folder, output_folder, pixels_per_mm)
    analyze_results(screw_lengths)