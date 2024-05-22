import numpy as np
import imutils
import cv2



















obj:dict={}

t_list=[]

def drawMatches( imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # loop over the matches
    print(matches, status)
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    # return the visualization
    return vis









def stitch(images):
    (imageA, imageB) = images

    #check version of open cv
    obj["is_cv3"]= imutils.is_cv3(or_better=True)

    
    #gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    descriptor = cv2.xfeatures2d.SIFT_create()

    (kps, featuresA) = descriptor.detectAndCompute(imageA, None)
    kpsA = np.float32([kp.pt for kp in kps])
    t_list.append( (kpsA,featuresA) )


    (kps, featuresB) = descriptor.detectAndCompute(imageB, None)
    kpsB = np.float32([kp.pt for kp in kps])
    t_list.append( (kpsB,featuresB) )


    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches=[]
    ratio=0.75
    reprojThresh=1.0
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    
    if len(matches) > 4:
			# construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
    #mtc=(matches, H, status)

    result = cv2.warpPerspective(imageA, H,
                                 (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    print(result[0:imageB.shape[0]])
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    vis = drawMatches(imageA,imageB,kpsA, kpsB, matches, status)
    return (result, vis)





# read image 
imageA=cv2.imread('./test/test7.jpg')
imageB=cv2.imread('./test/test8.jpg')


# resize image with
imageA = imutils.resize(imageA,width=400)
imageB = imutils.resize(imageB,width=400)



(result, vis)=stitch([imageA,imageB])

#cv2.imshow("Image A", imageA)
#cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.imwrite("image1.jpg", result)
cv2.destroyAllWindows()

