# I have created a utility that does book cover identification

import numpy as np
import cv2

class CoverDiscriptor:
    def __init__(self, useSIFT = False):
        self.useSIFT = useSIFT

# this cover discriptor contains methods for finding keypoints in a
# image and then describe the area surrounding each keypoint using
# local invariant descriptors
    def describe(self, image):
        descriptor = cv2.BRISK_create()

        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()

        (kps, descs) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])

        return (kps, descs)
