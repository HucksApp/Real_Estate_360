import cv2
import numpy as np
import os
from Api.utilities.id import generate_id


class StitcherError(Exception):
    """Custom exception for errors in stitching."""
    pass


class Stitcher:
    stitched_count: int = 0

    def __init__(self, image_paths: list = None, resize_width: int = 800) -> None:
        self.__images_stitched: dict = {}
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        if image_paths:
            self.images_original = [cv2.imread(image_path) for image_path in image_paths]
            self.__check_image_loading()
            self.images_resized = [self.__resize_image(image, resize_width) for image in self.images_original]

    def __check_image_loading(self) -> None:
        """Check if all images are loaded correctly."""
        if any(image is None for image in self.images_original):
            raise StitcherError("One or more images could not be loaded. Please check the file paths.")

    def __resize_image(self, image: np.ndarray, target_width: int) -> np.ndarray:
        """Resize images to match the target width."""
        return cv2.resize(image, (target_width, int(image.shape[0] * target_width / image.shape[1])))

    def __handle_stitching_error(self, stitch_status: int) -> None:
        """Handle errors that occur during the stitching process."""
        if stitch_status != cv2.Stitcher_OK:
            raise StitcherError(f"Error during stitching. Status code: {stitch_status}")

    def __save_image(self, image: np.ndarray, path: str) -> bool:
        """Attempt to save the stitched image and handle errors."""
        if not cv2.imwrite(path, image):
            raise IOError(f"Failed to save stitched image to {path}")
        return True

    def stitch(self, stitched_path: str = None, max_width: int = 800) -> str:
        # Resize images only if their width exceeds the max_width
        resized_images = []
        for image in self.images_resized:
            if image.shape[1] > max_width:  # Check if the image width exceeds max_width
                target_width = max_width
                resized_image = cv2.resize(image, (target_width, int(image.shape[0] * target_width / image.shape[1])))
                resized_images.append(resized_image)
            else:
                resized_images.append(image)  # No resizing needed if the image is small enough

        self.stitcher.setPanoConfidenceThresh(0.5)  # Optional: fine-tune stitching confidence
        
        # Debug: Check number of keypoints in each image before stitching
        for image in resized_images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
            print(f"Number of keypoints: {len(kp)}")  # Print the number of keypoints for each image

        # Perform the stitching
        stitch_status, stitched_image = self.stitcher.stitch(resized_images)
        self.__handle_stitching_error(stitch_status)

        # Generate a unique ID for this stitched image
        id = generate_id()
        stitched_path = stitched_path or os.path.join(os.getcwd(), 'Api/bucket', f'stitched_{id}.jpg')

        # Save the result in the object's state
        self.obj = {'id': id, 'stitched_image': stitched_image, 'stitched_path': stitched_path}
        self.__images_stitched[str(id)] = self.obj
        Stitcher.stitched_count += 1

        return stitched_path


    def save(self) -> bool:
        """Save the stitched image to the path."""
        return self.__save_image(self.obj['stitched_image'], self.obj['stitched_path'])

    def show(self) -> None:
        """Display the stitched image using OpenCV."""
        if not hasattr(self, 'obj'):
            raise StitcherError("No stitched image to display.")

        cv2.imshow(self.obj['stitched_path'], self.obj['stitched_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_stitched(self, id: str) -> dict:
        """Retrieve the stitched image using its ID."""
        return self.__images_stitched.get(id, None)

    def __del__(self) -> None:
        """Ensure proper cleanup of resources."""
        if hasattr(self, 'obj'):
            self.__images_stitched.pop(str(self.obj['id']), None)
            del self.obj

    def __repr__(self) -> str:
        """Provide a textual representation of the stitched images."""
        rep = ""
        for obj in self.__images_stitched.values():
            rep += f"ID: {obj['id']} | Path: {obj['stitched_path']}\n"
        return rep





    



