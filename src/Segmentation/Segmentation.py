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

    def convolve(self, kernel):
        '''Convolves the image for any kernel'''
        
        output_image = np.zeros_like(self.image, dtype=np.float64)
        pad = kernel.shape[0] // 2
        
        for i in range(pad, self.height - pad):
            for j in range(pad, self.width - pad):
                region = self.image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                conv_value = np.sum(region * kernel)
                output_image[i, j] = conv_value
        
        return output_image

    def laplacian(self, kernel):
        '''Applies Laplacian filter to the image.
        mathematical definition: Z = f(x+1, y) + f(x-1, y) + f(x, y+1) + f(x, y-1) - 4f(x, y)'''

        convolved_image = self.convolve(kernel)
        laplacian_image = np.clip(convolved_image, 0, 255).astype(np.uint8)
        
        return laplacian_image

    def sobel_operator(self):
        '''Applies the Sobel operator to find the gradient magnitude.'''
        
        #Defines the kernels
        
        sobel_gx = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float64)
        
        sobel_gy = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=np.float64)
        
        # Gx and Gy
        gx_image = np.zeros_like(self.image, dtype=np.float64)
        gy_image = np.zeros_like(self.image, dtype=np.float64)
        
        pad = sobel_gx.shape[0] // 2 # -->pad = 2

        # Apply Convolution to Gx and Gy
        for i in range(pad, self.height - pad):
            for j in range(pad, self.width - pad):
                region = self.image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                
                #Gx
                gx_value = np.sum(region * sobel_gx)
                gx_image[i, j] = gx_value
                
                # Gy
                gy_value = np.sum(region * sobel_gy)
                gy_image[i, j] = gy_value
    
        # M = sqrt(Gx^2 + Gy^2)
        sobel_magnitude = np.sqrt(np.square(gx_image) + np.square(gy_image))
    
        #Normalize the magnitude
        sobel_normalized = cv.normalize(sobel_magnitude, None, 0, 255, cv.NORM_MINMAX)
        
        return sobel_normalized.astype(np.uint8)
    
    def zero_crossing(self, log_image, threshold):
        '''Find zero-crossing in an image based on an specified threshold'''

        edges = np.zeros_like(log_image, dtype=np.uint8)
        
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                #Checks horizontally
                n_left = log_image[i, j-1]
                n_right = log_image[i, j+1]

                #Checks Vertically
                n_up = log_image[i+1, j]
                n_down = log_image[i-1, j]
                
                #Checks first diagonal 
                n_ul = log_image[i-1, j-1]
                n_lr = log_image[i+1, j+1]
                
                #Checks second diagonal 
                n_ur = log_image[i-1, j+1]
                n_ll = log_image[i+1, j-1]

                
                if (n_left * n_right < 0 and abs(n_left - n_right) > threshold) or \
                   (n_up * n_down < 0 and abs(n_up - n_down) > threshold) or \
                   (n_ul * n_lr < 0 and abs(n_ul - n_lr) > threshold) or \
                   (n_ur * n_ll < 0 and abs(n_ur - n_ll) > threshold):
                    
                    edges[i, j] = 255 
        
        return edges

    def marr_hildreth(self, gaussian_ksize = 5, st_deviant = 1.0, threshold = 4, use_kernel_aproximation = False):
        '''Computes marr_hildreth algorithm'''
        if use_kernel_aproximation == False: 
            #First step:  Smooths the image with an Gaussian filter
            kernel1D = cv.getGaussianKernel(gaussian_ksize, st_deviant)
            #Creates an 2d Kernel 
            gaussian_kernel = np.outer(kernel1D, kernel1D)

            smoothed_image = self.convolve(gaussian_kernel)

            #Second step:  compute the laplacian

            laplacian_kernel = np.array([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=np.float64)
            
            original_image = self.image
            self.image = smoothed_image
            log_image = self.convolve(laplacian_kernel)
            self.image = original_image

            #Third step: find zero-crossing
            edge_image = self.zero_crossing(log_image, threshold)
            return edge_image

        else: 
            #We can optimize steps one and two by convoluting the 5x5 kernel bellow as an aproximation of the laplacian of gaussian 
            log_kernel_5x5 = np.array([
            [ 0,  0, -1,  0,  0],
            [ 0, -1, -2, -1,  0],
            [-1, -2, 16, -2, -1],
            [ 0, -1, -2, -1,  0],
            [ 0,  0, -1,  0,  0]
        ], dtype=np.float64)
            log_image = self.convolve(log_kernel_5x5)
            edge_image = self.zero_crossing(log_image, threshold)
            return edge_image



        


if __name__ == "__main__":
    original_image = cv.imread('./assets/retinal.png', cv.IMREAD_GRAYSCALE)
  
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

        PointDetection_kernel = np.array([[0, 1, 0],
                                         [1, -4, 1],
                                            [0, 1, 0]])
        
        #Hline_image = seg.laplacian(Hline_detection_kernel)
        #Vline_image = seg.laplacian(Vline_detection_kernel)
        #PointImage = seg.laplacian(PointDetection_kernel)
        #Sobel_image = seg.sobel_operator()
        marr_hildreth_image = seg.marr_hildreth(
            gaussian_ksize=5, 
            st_deviant=1.4, 
            threshold=2.0,
            use_kernel_aproximation=False 
        )
        #Display images
        cv.imshow('Original image', original_image)
        #cv.imshow('Laplacian image', Hline_image)
        #cv.imshow('Vertical Line Detection image', Vline_image)
        #cv.imshow('Point Detection image', PointImage)
        #cv.imshow('Sobel edge detection image', Sobel_image)
        cv.imshow('Marr Hildreth', marr_hildreth_image)

        #Save images

        cv.imwrite('./assets/outputs/marr_hildereth.png',marr_hildreth_image)
        
        cv.waitKey(0)
        cv.destroyAllWindows()
