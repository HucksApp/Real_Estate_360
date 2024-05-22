
import cv2
import numpy as np
import os
from Api.utilities.id import generate_id

class Sticher:
    stitched_count: int = 0
    def __init__(self, image_paths:list[str]=None) -> None:
        self.__images_stitched:dict = {}
        # Create a Stitcher object using OpenCV's create method
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        if image_paths and type(image_paths) is list:
            self.images_original = [cv2.imread(image_path)
                                    for image_path in image_paths]
            self.__path_error()
            # Resize images to have the same width 
            target_width = 800
            self.images_resized = [cv2.resize(image, 
                                         (target_width, int(image.shape[0] * target_width / image.shape[1])))
                                           for image in self.images_original]
    def __path_error(self) -> None:
        # Check if all images are loaded correctly
        if any(image is None for image in self.images_original):
            msg = ("One or more images could not be loaded."
                    "Please check the file paths.")
            raise ValueError(msg)
        
    def __stitch_error(self, stitch_status:int) -> None:
        if stitch_status != cv2.Stitcher_OK:
            msg = ("Error during stitching. Status code: {}"
                   .format(stitch_status))
            raise(msg)

    def __write_error(self, write_status:bool, stitched_path:str) -> None:
        if not write_status:
            msg = ("Failed to save stitched image to {}"
                   .format(stitched_path))
            raise IOError(msg)

    def stitch(self, stitched_path: str=None) -> str:
        # Perform the stitching process
        # The stitch method returns a status code and the stitched image
         # Create a stitcher object with advanced options
        self.stitcher.setPanoConfidenceThresh(0.0)
        stitched_status, stitched_image = self.stitcher.stitch(self.images_resized)
        self.__stitch_error(stitched_status)
        id = generate_id()
        if not stitched_path:
            stitched_path = (os.path.join(os.getcwd(), 'Api/bucket', 'stitched_{}.jpg'
                             .format(id)))
    
        self.obj = {'id': id, 'stitched_image': stitched_image, 'stitched_path': stitched_path}
        self.__images_stitched.update({str(id): self.obj})
        Sticher.stitched_count += 1
        return stitched_path
    
    def save(self) -> bool:
        write_status = cv2.imwrite(self.obj['stitched_path'], self.obj['stitched_image'])
        self.__write_error(write_status,self.obj['stitched_path'])
        return self.obj['id']
    
    def __del__(self) -> str:
        if hasattr(self, 'obj'):
            print()
            for key in self.__images_stitched.keys():
                if self.obj['id'] == key:
                    self.__images_stitched.pop(key)
            delattr(self, 'obj')

        if hasattr(self, '_image'):
            if os.path.isfile(self._image):
                os.remove(self._image)

    def show(self) -> None:
        if hasattr(self, 'obj'):
            cv2.imshow(self.obj['stitched_path'], self.obj['stitched_image'])
        elif hasattr(self, '_image'):
            stitched = cv2.imread(self._image)
            cv2.imshow(self._image, stitched)
        else:
            raise("Can't Show Image")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_stitched(self, id) -> str:
        for key, value in self.__images_stitched.items():
            if key == id:
                return  value
        return None
    
    def __repr__(self) -> str:
        rep =  ""
        for objs in self.__images_stitched.values():
            for key, value in objs.items():
                if key == 'stitched_image':
                    v1 = np.sum(value, axis=0)
                    v2 = np.sum(value, axis=1)
                    value = "numpy array np({}-axis 0 {}-axis 1)".format(v1, v2)
                rep += "{}: {}\n".format(key, value)
        return rep
"""
    @property
    def images_stitched(self) -> dict:
        return self.__images_stitched
    
    @property.setter
    def images_stitched(self, obj) -> dict:
        self.__images_stitched.update({obj.id : obj})
"""
  





    



