import cv2
import os

def stitch_images(image_paths):
    images = [cv2.imread(image_path) for image_path in image_paths]
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(0.0)
    status, stitched_image = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Image stitching succeeded!")
        # Save the stitched image
        #stitched_image_path = os.path.join(os.path.expanduser('~'),'Desktop/REAL-ESTATE-360/','fig.jpg')
        stitched_image_path = os.path.join(os.getcwd(),'bucket', 'fig.jpg')
        m = cv2.imwrite(stitched_image_path, stitched_image)
        
        if not m:
            msg = ("Failed to save stitched image to {}"
                   .format(stitched_image_path))
            raise IOError(msg)
        cv2.imshow("result", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            #return stitched_image_path
    else:
        print(f"Image stitching failed with status {status}")
        return None
    

stitch_images(['images/e1.JPG', 'images/e2.JPG', 'images/e3.JPG', 'images/e4.JPG'])