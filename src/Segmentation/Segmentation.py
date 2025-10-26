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
        
    def canny_gradient_step(self, ksize, sigma):
        '''
        Performs Canny steps 1 & 2 
        1. Smooth with Gaussian
        2. Calculate Gradient (Magnitude and Angle)
        '''
        
        # --- STEP 1: Gaussian Smoothing 
        # Create Gaussian kernel
        kernel_1d = cv.getGaussianKernel(ksize, sigma)
        gaussian_kernel = np.outer(kernel_1d, kernel_1d)
        
        # Convolve the ORIGINAL IMAGE with the Gaussian
        fs_smoothed_image = self.convolve(gaussian_kernel)

        # --- STEP 2: Calculate Gradient  ---
        sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_gy = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float64)

        # Convolve the SMOOTHED IMAGE with Sobel
        aux = self.image
        self.image = fs_smoothed_image
        gs_image_x = self.convolve( sobel_gx)
        gs_image_y = self.convolve( sobel_gy)
        self.image = aux

        # Calculate Magnitude 
        magnitude = np.sqrt(np.square(gs_image_x) + np.square(gs_image_y))
        
        # Calculate Angle 
        # Angle in radians, -pi to +pi
        angle = np.arctan2(gs_image_y, gs_image_x) 
        
        return magnitude, angle
    
    def non_maxima_suppression(self, magnitude, angle_rad):
        '''
        Performs Canny Step 3: Non-Maxima Suppression.
        Thins the "wide ridges" to 1-pixel edges.
        '''
        
        # Convert angles from radians to degrees [-180, 180]
        angle_deg = np.rad2deg(angle_rad)
        
        # Output image, initialized with zeros
        nms_image = np.zeros_like(magnitude, dtype=np.float64)
        
        # Scan the image (avoiding the 1-pixel border)
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                
                M = magnitude[i, j]
                A_deg = angle_deg[i, j]
                
                # --- Quantize the angle to one of 4 directions ---
                
                # Case 1: Horizontal Edge (Vertical Gradient)
                # Angle is between -22.5 & 22.5 OR (>= 157.5 or <= -157.5)
                if (-22.5 <= A_deg <= 22.5) or (A_deg <= -157.5) or (A_deg >= 157.5):
                    n1 = magnitude[i - 1, j] # Neighbor Above
                    n2 = magnitude[i + 1, j] # Neighbor Below
                
                # Case 2: +45° Edge (Gradient -45°)
                # Angle is between 22.5 & 67.5 OR -157.5 & -112.5
                elif (22.5 <= A_deg <= 67.5) or (-157.5 <= A_deg <= -112.5):
                    n1 = magnitude[i - 1, j - 1] # Neighbor Top-Left
                    n2 = magnitude[i + 1, j + 1] # Neighbor Bottom-Right

                # Case 3: Vertical Edge (Horizontal Gradient)
                # Angle is between 67.5 & 112.5 OR -112.5 & -67.5
                elif (67.5 <= A_deg <= 112.5) or (-112.5 <= A_deg <= -67.5):
                    n1 = magnitude[i, j - 1] # Neighbor Left
                    n2 = magnitude[i, j + 1] # Neighbor Right
                
                # Case 4: -45° Edge (Gradient +45°)
                # Angle is between 112.5 & 157.5 OR -67.5 & -22.5
                elif (112.5 <= A_deg <= 157.5) or (-67.5 <= A_deg <= -22.5):
                    n1 = magnitude[i - 1, j + 1] # Neighbor Top-Right
                    n2 = magnitude[i + 1, j - 1] # Neighbor Bottom-Left
                
                # If M is the local maximum, keep its value
                if (M >= n1) and (M >= n2):
                    nms_image[i, j] = M
                # Otherwise, suppress it (it's already 0 by default)
        
        return nms_image
    def hysteresis_thresholding(self, nms_image, low_thresh, high_thresh):
        '''
        Performs Canny Step 4: Double Thresholding and Hysteresis.
        Links strong and weak edges.
        '''
        
        
        # Create output image
        edges_image = np.zeros_like(nms_image, dtype=np.uint8)
        
        # 1. Classify strong and weak pixels
        strong_pixels = []
        weak_pixels = {} # Use a dict (hash map) for fast lookup of weak pixels

        for i in range(self.height):
            for j in range(self.width):
                mag = nms_image[i, j]
                if mag >= high_thresh:
                    edges_image[i, j] = 255  # Mark strong pixels
                    strong_pixels.append((i, j)) # Add to stack
                elif mag >= low_thresh:
                    weak_pixels[(i, j)] = True # Mark as weak
        
        # 2. Connectivity Analysis (using a stack - non-recursive)
        # Start stack with all strong pixels
        stack = strong_pixels 
        
        while stack:
            r, c = stack.pop()
            
            # Check 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue # Skip self
                    
                    nr, nc = r + dr, c + dc
                    
                    # Check bounds
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        # Check if this neighbor is a weak pixel
                        if (nr, nc) in weak_pixels:
                            # Promote it to an edge 
                            edges_image[nr, nc] = 255
                            # Remove from weak_pixels to avoid re-processing
                            del weak_pixels[(nr, nc)]
                            # Add it to the stack to check its neighbors
                            stack.append((nr, nc))
        
        # All remaining pixels in edges_image are 0 (suppressed) 
        return edges_image

    def canny_edge(self, k_size=5, sigma = 1.0, high_threshold = 30, low_threshold= 15):
        #First step:  get the magnitude and angle 
        magnitude, angle = self.canny_gradient_step(ksize=k_size, sigma=sigma)
        #Second we apply the nonmaxima fucntion
        nms_image = self.non_maxima_suppression(magnitude=magnitude, angle_rad=angle)
        #Step 3 we do the hysteresis thresholding
        canny_edge_image = self.hysteresis_thresholding(nms_image=nms_image, low_thresh=low_threshold, high_thresh=high_threshold)

        return canny_edge_image

        


if __name__ == "__main__":
    original_image = cv.imread('./assets/brain_tumor.jpeg', cv.IMREAD_GRAYSCALE)
  
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
        #marr_hildreth_image = seg.marr_hildreth(
         #   gaussian_ksize=5, 
        #    st_deviant=1.4, 
        #    threshold=2.0,
        #    use_kernel_aproximation=False 
        #)
        #canny_image = seg.canny_edge(k_size=5, 
            #                         sigma=1.0, 
           #                          high_threshold=90,
          #                           low_threshold=30
         #                       )
        #canny_image_opencv = cv.Canny(original_image, 100, 40)

        #Display images
        cv.imshow('Original image', original_image)
        #cv.imshow('Laplacian image', Hline_image)
        #cv.imshow('Vertical Line Detection image', Vline_image)
        #cv.imshow('Point Detection image', PointImage)
        #cv.imshow('Sobel edge detection image', Sobel_image)
        #cv.imshow('Marr Hildreth', marr_hildreth_image)
        #cv.imshow('Canny', canny_image)

        #Save images

        #cv.imwrite('./assets/outputs/canny.png',canny_image)
        
        cv.waitKey(0)
        cv.destroyAllWindows()
