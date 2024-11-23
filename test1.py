import os
from Api.utilities.id import generate_id
from Api.classes.stitcher import Stitcher  # Assuming Sticher class is saved in Sticher.py

# Test images (make sure these paths are correct and point to your test images)
image_paths = [
    'images/e1.JPG',
    'images/e2.JPG',
    'images/e3.JPG',
    'images/e4.JPG',
    'images/e5.JPG'

    # Add more images if necessary
]

def test_sticher():
    try:
        # Initialize the Sticher class with image paths
        stitcher = Stitcher(image_paths=image_paths)

        # Perform stitching
        stitched_path = stitcher.stitch('Api/test.jpg')

        # Display the stitched image
        print(f"Stitched image saved at: {stitched_path}")
        stitcher.save()
        stitcher.show()

        # Optionally, save the stitched image
       

    except ValueError as ve:
        print(f"Error occurred during stitching: {ve}")
    except IOError as io:
        print(f"Error occurred while saving the image: {io}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_sticher()