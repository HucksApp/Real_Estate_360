import cv2


def stitch_images(image_paths):
    # Read images from file paths
    images = [cv2.imread(image_path) for image_path in image_paths]

    # Check if images were read successfully
    if any(image is None for image in images):
        print("Error: One or more images could not be loaded.")
        return None

    # Resize images to have the same width (optional but recommended for better results)
    target_width = 800
    resized_images = [cv2.resize(image, (target_width, int(image.shape[0] * target_width / image.shape[1]))) for image in images]

    # Create a stitcher object
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(0.5)
    # Perform the stitching process
    status, stitched_image = stitcher.stitch(resized_images)

    # Handle the possible statuses
    # Handle the possible statuses
    if status == cv2.Stitcher_OK:
        print("Image stitching succeeded!")
        # Save the stitched image
        stitched_image_path = '~/desktop/REAL-ESTATE-360/test_img/stitched.jpg'
        cv2.imwrite(stitched_image_path, stitched_image)
        cv2.imshow("result", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            #return stitched_image_path
    else:
        print(f"Image stitching failed with status {status}")
        return None

    

stitch_images(['images/e1.JPG', 'images/e2.JPG', 'images/e3.JPG', 'images/e4.JPG'])