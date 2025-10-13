import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv

class Morphological:
    """Class for morphological operations on images."""
    def __init__(self, image):
        #EXPECTS BINARY IMAGE
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]

    def erode(self, kernel_size=(3,3)):
        """Erosion operation on the image. A ⊖ B = {z|(B) z ⊆ A} or A ⊖ B = {z|[(Bˆ)z ∩ A] = ∅}"""
        kernel = np.ones(kernel_size, dtype=np.uint8)
        '''A way to handle :
        pad_h, pad_w = kernel_size[0] // 2, kernel_size[1] // 2
        
        output = np.zeros_like(self.image)

        # Iterate over every pixel in the image
        for y in range(pad_h, self.height - pad_h):
            for x in range(pad_w, self.width - pad_w):
                region = self.image[y - pad_h : y + pad_h + 1, x - pad_w : x + pad_w + 1]
                
                # Every pixel in the kernel must match (be 255) for erosion
                if np.all(region[kernel == 1] == 255):
                    output[y, x] = 255'''
        
        return cv.erode(self.image, kernel, iterations=1)
    
    def dilate(self, kernel_size=(3,3)):
        """Dilation operation on the image. A ⊕ B = {z|[(Bˆ)z ∩ A] ⊆ A}
        or A ⊕ B = {z|(B) z ∩ A ≠ ∅}"""
        kernel = np.ones(kernel_size, dtype=np.uint8)
        '''
        #ALTERNATIVE WAY TO HANDLE DILATION
        pad_h, pad_w = kernel_size[0] // 2, kernel_size[1] // 2

        # All black output image
        output = np.zeros_like(self.image)

        # iterate over every pixel in the image
        for y in range(self.height):
            for x in range(self.width):
                # if the pixel is white (255), we dilate
                if self.image[y, x] == 255:              
                    y_min = max(0, y - pad_h)
                    y_max = min(self.height, y + pad_h + 1)
                    x_min = max(0, x - pad_w)
                    x_max = min(self.width, x + pad_w + 1)
                    # Set the region to white (255)
                    output[y_min:y_max, x_min:x_max] = 255
        '''
        return cv.dilate(self.image, kernel, iterations=1)
    def open(self, kernel_size=(3,3)):
        """Opening operation on the image. A ◦ B = (A ⊖ B) ⊕ B erode then dilate"""
        kernel = np.ones(kernel_size, dtype=np.uint8)
        eroded = cv.erode(self.image, kernel, iterations=1)
        opened = cv.dilate(eroded, kernel, iterations=1)
        return opened

    def closing(self, kernel_size=(3,3)):
        """Closing operation on the image. A • B = (A ⊕ B) ⊖ B dilate then erode"""
        kernel = np.ones(kernel_size, dtype=np.uint8)
        dilated = cv.dilate(self.image, kernel, iterations=1)
        closed = cv.erode(dilated, kernel, iterations=1)
        return closed
    
    def HitOrMiss(self, hit_kernel, miss_kernel):
        """Hit-or-miss transform on the image. A ⊛ B = (A ⊖ B1) ∩ (Ac ⊖ B2), Ac is the complement of A (if A is ones , Ac is zeros and vice-versa)"""
        #First hit
        erosion_hit = cv.erode(self.image,hit_kernel, iterations=1)

        #Then we take the complement of the image
        complement_image = cv.bitwise_not(self.image)
        #Then miss the complement
        erosion_miss = cv.erode(complement_image, miss_kernel, iterations=1)

        #Finally we intersect the two results
        hit_or_miss = cv.bitwise_and(erosion_hit, erosion_miss)
        
        return hit_or_miss
    
    def BoundaryExtraction(self, kernel_size=(3,3)):
        """Boundary extraction on the image. β(A) = A - (A ⊖ B)"""
        kernel = np.ones(kernel_size, dtype=np.uint8)
        eroded = cv.erode(self.image, kernel, iterations=1)
        boundary = cv.subtract(self.image, eroded)
        return boundary
    
# Test
if __name__ == "__main__":
    original_image = cv.imread('./assets/horse2.png')

    if original_image is None:
        print("Erro: Img not found.")
    else:
        gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        _, binary_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY)

        morph = Morphological(binary_image)
    
        #eroded_image = morph.erode(kernel_size=(6,6))
        #dilated_image = morph.dilate(kernel_size=(6,6))
        #opened_image = morph.open(kernel_size=(6,6))
        #closing_image = morph.closing(kernel_size=(6,6))
        #Define hit and miss kernels
        hit_kernel = np.array([[1, 0, 0],
                               [1, 0, 0],
                               [1, 0, 0]], dtype=np.uint8) 
        miss_kernel = np.array([[0, 0, 1],
                                [0, 0, 1],
                                [0, 0, 1]], dtype=np.uint8)   
        hit_or_miss_image = morph.HitOrMiss(hit_kernel, miss_kernel)
        #boundary_image = morph.BoundaryExtraction(kernel_size=(3,3))

        cv.imshow('Binary img', binary_image)
        #cv.imshow('Erode image', eroded_image)
        #cv.imshow('Dilate image', dilated_image)
        #cv.imshow('Opening', opened_image)
        #cv.imshow('Closing', closing_image)
        #cv.imshow('Hit or Miss', hit_or_miss_image)
        #cv.imshow('Boundary Extraction', boundary_image)

        #Save images
        #cv.imwrite('./assets/outputs/erode.png', eroded_image)
        #cv.imwrite('./assets/outputs/dilate.png', dilated_image)
        #cv.imwrite('./assets/outputs/opening.png', opened_image)
        #cv.imwrite('./assets/outputs/closing.png', closing_image)
        #cv.imwrite('./assets/outputs/hit_or_miss.png', hit_or_miss_image)
        #cv.imwrite('./assets/outputs/boundary_extraction.png', boundary_image)

        cv.waitKey(0)
        cv.destroyAllWindows()