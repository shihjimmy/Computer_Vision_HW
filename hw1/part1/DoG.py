import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        gaussian_images.append(image)
        # set kernel size to (0,0), so that OpenCV automatically determines the best kernel size based on the sigma
        gaussian_images.append( cv2.GaussianBlur(image, (0,0), self.sigma) )
        gaussian_images.append( cv2.GaussianBlur(image, (0,0), self.sigma**2) )
        gaussian_images.append( cv2.GaussianBlur(image, (0,0), self.sigma**3) )
        gaussian_images.append( cv2.GaussianBlur(image, (0,0), self.sigma**4) )

        resized_image = cv2.resize(gaussian_images[4], (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        gaussian_images.append(resized_image)
        gaussian_images.append( cv2.GaussianBlur(resized_image, (0,0), self.sigma) )
        gaussian_images.append( cv2.GaussianBlur(resized_image, (0,0), self.sigma**2) )
        gaussian_images.append( cv2.GaussianBlur(resized_image, (0,0), self.sigma**3) )
        gaussian_images.append( cv2.GaussianBlur(resized_image, (0,0), self.sigma**4) )

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        dog_images.append( cv2.subtract(gaussian_images[1],gaussian_images[0]) )
        dog_images.append( cv2.subtract(gaussian_images[2],gaussian_images[1]) )
        dog_images.append( cv2.subtract(gaussian_images[3],gaussian_images[2]) )
        dog_images.append( cv2.subtract(gaussian_images[4],gaussian_images[3]) )

        dog_images.append( cv2.subtract(gaussian_images[6],gaussian_images[5]) )
        dog_images.append( cv2.subtract(gaussian_images[7],gaussian_images[6]) )
        dog_images.append( cv2.subtract(gaussian_images[8],gaussian_images[7]) )
        dog_images.append( cv2.subtract(gaussian_images[9],gaussian_images[8]) )
        
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []

        for octave in range(self.num_octaves): 
            for i in range(1, self.num_DoG_images_per_octave-1):           
                img_prev = dog_images[i - 1 + (octave * 4)]
                img_cur  = dog_images[i + (octave * 4)]
                img_next = dog_images[i + 1 + (octave * 4)]

                h, w = img_cur.shape

                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        center_pixel = img_cur[y, x]
                        if ( abs(center_pixel) <= self.threshold ):
                            continue
                        
                        # Extract 3x3x3 neighborhood
                        patch = np.stack([img_prev[y-1:y+2, x-1:x+2], img_cur[y-1:y+2, x-1:x+2], img_next[y-1:y+2, x-1:x+2]])
                        max_index = np.unravel_index(np.argmax(patch), patch.shape)
                        min_index = np.unravel_index(np.argmin(patch), patch.shape)
                                             
                        if (
                            (center_pixel == patch.max() and max_index == (1, 1, 1) and np.count_nonzero(patch == patch.max()) == 1) or
                            (center_pixel == patch.min() and min_index == (1, 1, 1) and np.count_nonzero(patch == patch.min()) == 1)
                        ):
                            if octave == 1:
                                keypoints.append( [2*y, 2*x] ) # 2nd octave image is resized to 1/2
                            else:
                                keypoints.append( [y, x] )
                                
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)  
        # sort 2d-point
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]       
        return keypoints
