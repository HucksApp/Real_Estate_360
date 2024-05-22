from Api.classes.frame import Frame
from Api.classes.location import Location
from Api.classes.stitcher import Sticher


m1 = Frame('frame 1',['images/e1.JPG', 'images/e2.JPG', 'images/e3.JPG', 'images/e4.JPG'])
m2 = Frame('frame 2','./Api/bucket/stitched_b367c73b-be6a-465f-a6eb-b8bcaa0edee8.jpg')

#m1 = Sticher(['images/e1.JPG', 'images/e2.JPG', 'images/e3.JPG', 'images/e4.JPG'])

#print(dir(m1))
#print(m1.__dict__)
m1.add_location({'title':"move to 2", "position":155267,'icon': "mm/nn/cn", 'link':m2})
m1.stitch()
print(m1.save())
m1.show()
#del m1