import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv

class Segmentation:
    '''Class for handling image segmentation tasks.'''
    def __init__(self, image):
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]

    def laplacian(self, kernel):
        '''Applies Laplacian filter to the image.
        mathematical definition: Z = f(x+1, y) + f(x-1, y) + f(x, y+1) + f(x, y-1) - 4f(x, y)'''
        laplacian_image = self.image.copy()
        pad = kernel.shape[0] // 2
        for i in range(pad, self.height - pad):
            for j in range(pad, self.width - pad):
                region = self.image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                laplacian_value = np.sum(region * kernel)
                laplacian_image[i, j] = np.clip(laplacian_value, 0, 255)
        
        return laplacian_image


if __name__ == "__main__":
    original_image = cv.imread('./assets/coins23.png', cv.IMREAD_GRAYSCALE)
  
    if original_image is None:
        print("Erro: Img not found.")
    else:
        seg =  Segmentation(original_image)
        #Laplacian kernel
        Hline_detection_kernel = np.array([[-1, -1, -1],
                                     [2, 2, 2],
                                     [-1, -1, -1]])
        Vline_detection_kernel = np.array([[-1, 2, -1],
                                         [-1, 2, -1],
                                         [-1, 2, -1]])

        PointDetection_kernel = np.array([[-1, -1, -1],
                                         [-1, 8, -1],
                                            [-1, -1, -1]])
        
        Hline_image = seg.laplacian(Hline_detection_kernel)
        Vline_image = seg.laplacian(Vline_detection_kernel)
        PointImage = seg.laplacian(PointDetection_kernel)
        #Display images
        cv.imshow('Original image', original_image)
        cv.imshow('Laplacian image', Hline_image)
        cv.imshow('Vertical Line Detection image', Vline_image)
        cv.imshow('Point Detection image', PointImage)

        #Save images

        #cv.imwrite('./assets/outputs/',)
       
        cv.waitKey(0)
        cv.destroyAllWindows()
