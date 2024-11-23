from Api.classes.stitcher import Sticher
from typing import TypedDict
from Api.classes.location import Location


class Frame(Sticher):

    def __init__(self,title: str, image_paths: str) -> None:

        init =image_paths if type(image_paths) is list else None
        super().__init__(init)
        
        if init:
            self._image = self.stitch()
        if type(image_paths) is str:
            self._image = image_paths
        self._title = title
            # location: {title, position, icon, link:Frame }
        self._locations = {}


    def add_location(self, obj:dict):
        location = Location(**obj)
        self._locations.update({location.id : location})

    def delete_location(self, id) ->'Location':
        for key, value in self._locations.items():
            if key == id:
                self._locations.pop(id)
                return value
        return None
    def __repr__(self) -> str:
        return "[Frame: title:{}\n image:{}\n locations:{}]".format(self._title, self._image, self._locations)

        
    
    



 
    



    