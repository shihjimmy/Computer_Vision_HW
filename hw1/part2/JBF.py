import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

        scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
        # Pixel values should be normalized to [0, 1] (divided by 255) to construct range kernel
        scaleFactor_r = 1 / (2 * sigma_r **2 * 255 ** 2)
        
        self.spatial_kernel = np.array(
                            [ [np.exp(-(i**2+j**2) * scaleFactor_s) \
                                for i in range(-(self.wndw_size // 2), (self.wndw_size +1) // 2)] \
                                for j in range(-(self.wndw_size // 2), (self.wndw_size +1) // 2) ]
                            )
        
        # Generate look up table for range kernel
        # Since the result of subtraction of intensity must fall in [0,255]
        # generate a list: (0^2 ~ 255^2) * scaleFactor_r
        self.LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)
        

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        output = np.zeros(img.shape)
        
        if len(guidance.shape) == 2:
            #gray scale
            for i in range(self.pad_w , padded_img.shape[0] - self.pad_w):
                for j in range(self.pad_w , padded_img.shape[1] - self.pad_w):
                    patch_img = padded_img[i - self.pad_w : i + self.pad_w + 1,  j - self.pad_w : j + self.pad_w + 1]
                    patch_guide = padded_guidance[i - self.pad_w : i + self.pad_w + 1,  j - self.pad_w : j + self.pad_w + 1]
                    center_val = padded_guidance[i, j]

                    Gr = self.LUT[ np.abs(patch_guide - center_val) ]
                    Gr_with_Gs = Gr * self.spatial_kernel
                    output[i - self.pad_w, j - self.pad_w] = np.sum(Gr_with_Gs[:,:,np.newaxis] * patch_img, axis=(0,1)) / np.sum(Gr_with_Gs)


        elif len(guidance.shape) == 3:
            #RGB scale
            for i in range(self.pad_w , padded_img.shape[0] - self.pad_w):
                for j in range(self.pad_w , padded_img.shape[1] - self.pad_w):
                    patch_img = padded_img[i - self.pad_w : i + self.pad_w + 1,  j - self.pad_w : j + self.pad_w + 1]
                    patch_guide_r = padded_guidance[i - self.pad_w : i + self.pad_w + 1,
                                                    j - self.pad_w : j + self.pad_w + 1, 0]
                    center_val_r = padded_guidance[i, j, 0]

                    patch_guide_g = padded_guidance[i - self.pad_w : i + self.pad_w + 1,
                                                    j - self.pad_w : j + self.pad_w + 1, 1]
                    center_val_g = padded_guidance[i, j, 1]

                    patch_guide_b = padded_guidance[i - self.pad_w : i + self.pad_w + 1,
                                                    j - self.pad_w : j + self.pad_w + 1, 2]
                    center_val_b = padded_guidance[i, j, 2]

                    Gr = self.LUT[ np.abs(patch_guide_r - center_val_r) ] * self.LUT[ np.abs(patch_guide_g - center_val_g) ] * self.LUT[ np.abs(patch_guide_b - center_val_b) ]
                    Gr_with_Gs = Gr * self.spatial_kernel
                    output[i - self.pad_w, j - self.pad_w] = np.sum(Gr_with_Gs[:,:,np.newaxis] * patch_img, axis=(0,1)) / np.sum(Gr_with_Gs)
 
        return np.clip(output, 0, 255).astype(np.uint8)
