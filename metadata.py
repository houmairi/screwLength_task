import os
from PIL import Image
from PIL.ExifTags import TAGS

def get_image_metadata(image_path):
    try:
        image = Image.open(image_path)
        exif_data = {}
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
        return exif_data
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return {}

image_directory = r'' # Paste folder location here

if not os.path.exists(image_directory):
    print(f"The directory {image_directory} does not exist. Please check the path.")
else:
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            file_path = os.path.join(image_directory, filename)
            print(f"Metadata for {filename}:")
            metadata = get_image_metadata(file_path)
            if metadata:
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("  No metadata found or error occurred.")
            print("\n")