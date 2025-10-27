import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
import heapq  # For watershed priority queue
from collections import deque # For connected components BFS
import math


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

    def hough_transform(self):
        """
        Implements the Standard Hough Transform for line detection from scratch.

        THE INPUT IMAGE MUST BE BINARY
        """
        # The maximum possible rho value is the diagonal of the image
        max_rho = int(np.ceil(np.hypot(self.height,self.width )))
        # This gives us (2 * max_rho) + 1 bins for rho
        rho_bins = np.arange(-max_rho, max_rho + 1, 1) # 1-pixel resolution
        theta_bins_deg = np.arange(0, 180, 1) # 1-degree resolution
        theta_bins_rad = np.deg2rad(theta_bins_deg)
        # Initialize the accumulator array with zeros
        num_rhos = len(rho_bins)
        num_thetas = len(theta_bins_rad)
        accumulator = np.zeros((num_rhos, num_thetas), dtype=np.uint64)

        # 2. Get Coordinates of All Edge Pixels
        # np.where returns two arrays: one for y-indices, one for x-indices
        y_indices, x_indices = np.where(self.image > 0)
        
        # 3. Cast Votes
        
        for i in range(len(x_indices)):
            x = x_indices[i]
            y = y_indices[i]
            
            # For this one (x, y) point, calculate rho for all possible thetas
            for t_idx in range(num_thetas):
                theta = theta_bins_rad[t_idx]
                
                # Calculate rho: rho = x*cos(theta) + y*sin(theta)
                rho = round(x * np.cos(theta) + y * np.sin(theta))
                
                # Map this rho value to its index in the accumulator
                # We add max_rho to offset the negative indices
                rho_idx = int(rho) + max_rho 
                
                # Increment the vote in the accumulator
                accumulator[rho_idx, t_idx] += 1
                
        return accumulator, theta_bins_rad, rho_bins
    
    def draw_hough_lines(self, rhos, thetas, color=(0, 255, 0), thickness=2):
        """
        Draws lines on an image given (rho, theta) pairs.
            """
        # Create a color copy of the original image to draw on
        copy_image = self.image
        # This assumes self.image is grayscale. If it might be color, use cv.cvtColor
        # For this class, self.image is the canny_image (binary), so we need to make it color
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
             self.image = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)

        for rho, theta in zip(rhos, thetas):
            a = np.cos(theta)
            b = np.sin(theta)
            
            # Find a point on the line
            x0 = a * rho
            y0 = b * rho
            
            # Get two other points on the line to draw it across the image
            # We just extend it by 1000 pixels in both directions
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            
            cv.line(copy_image, pt1, pt2, color, thickness)
        return copy_image
    
    def otsu_threshold(self):
        """
        Implements Otsu's method to find the optimal global threshold.
        Based on the algorithm described in the README_Otsu.md.
        
        This is an efficient, 1-pass implementation.
        """
        
        #Compute and Normalize Histogram
        # 'self.image.ravel()' flattens the 2D image to a 1D array
        hist_counts, _ = np.histogram(self.image.ravel(), bins=256, range=[0, 256])
        
        # Normalize to get probabilities (p_i)
        total_pixels = self.width * self.height
        hist_prob = hist_counts.astype(np.float64) / total_pixels
        
        # sum(i * p_i) for i = 0 to 255
        total_mean = np.dot(np.arange(256), hist_prob)
        
        # Iterate and Evaluate 
        max_sigma_b_squared = -1.0
        optimal_threshold = 0
        
        #Class 0 = Background, Class 1 = Foreground

        w0 = 0.0  # Weight of Class 0
        sum0 = 0.0 # (i * p_i) for Class 0

        
        for t in range(256):
            # Add this bin's values to the running sums for Class 0
            w0 += hist_prob[t]
            sum0 += t * hist_prob[t]
            
            #Calculate Metrics
            # Get weight for Class 1
            w1 = 1.0 - w0
            
            # Handle edge cases where a class has 0 pixels
            if w0 == 0.0 or w1 == 0.0:
                continue # It means that one of the classes is empty.

            # mu0 = (Sum of i*p_i for class 0) / w0
            mu0 = sum0 / w0
            
            # mu1 = (Total sum - Class 0 sum) / w1
            # Total sum is total_mean. Class 0 sum is sum0.
            mu1 = (total_mean - sum0) / w1
            
            #Calculate Objective Function
            # sigma_B^2(t) = w_0(t) * w_1(t) * (mu_1(t) - mu_0(t))^2
            sigma_b_squared = (w0 * w1) * ((mu1 - mu0) ** 2)
            
            #Find Maximum
            if sigma_b_squared > max_sigma_b_squared:
                max_sigma_b_squared = sigma_b_squared
                optimal_threshold = t
                
        # Apply the threshold 
        print(f"Otsu's optimal threshold: {optimal_threshold}")
        
        # Create the binary image
        # Class 0 (Background): <= t
        # Class 1 (Foreground): > t
        thresholded_image = np.zeros_like(self.image, dtype=np.uint8)
        thresholded_image[self.image > optimal_threshold] = 255
        
        return optimal_threshold, thresholded_image
    
    def connected_components(self, binary_image):
        """
        Finds and labels connected components (markers) in a binary image.
        Uses a Breadth-First Search (BFS) algorithm.
        Returns a label_image where 0 is background and 1, 2, 3... are labels.
        """
        labels = np.zeros_like(binary_image, dtype=np.int32)
        current_label = 1
        
        # 8-connectivity neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i in range(self.height):
            for j in range(self.width):
                # If this is a foreground pixel (255) and not yet labeled (0)
                if binary_image[i, j] == 255 and labels[i, j] == 0:
                    
                    # Start a new label and a new queue for BFS
                    q = deque()
                    q.append((i, j))
                    labels[i, j] = current_label
                    
                    while q:
                        r, c = q.popleft()
                        
                        # Check all 8 neighbors
                        for dr, dc in neighbors:
                            nr, nc = r + dr, c + dc
                            
                            # Check bounds
                            if 0 <= nr < self.height and 0 <= nc < self.width:
                                # If neighbor is foreground and not labeled
                                if binary_image[nr, nc] == 255 and labels[nr, nc] == 0:
                                    labels[nr, nc] = current_label
                                    q.append((nr, nc))
                                    
                    # Done with this component, increment label for next one
                    current_label += 1
                    
        return labels, current_label - 1

    def _watershed_flood(self, gradient_image, markers):
        """
        Performs the core watershed "flooding" algorithm.
        - gradient_image: The "topography" to be flooded.
        - markers: An image with labeled seeds (1=BG, 2,3...=FG objects)
        """
        
        # Constants
        WATERSHED_LINE = -1
        UNLABELED = 0
        
        # Output image for labels, initialized from markers
        labels = markers.astype(np.int32)
        
        # Priority Queue (min-heap)
        # Stores (priority, row, col)
        # Priority is the gradient intensity
        pq = []
        
        # 8-connectivity neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # 1. Initialize the queue
        # Find all labeled pixels (markers) and push their neibors
        # onto the queue. This is more efficient than pushing the markers themselves.
        for r in range(self.height):
            for c in range(self.width):
                if labels[r, c] != UNLABELED:
                    # This is a marker pixel. Check its neighbors.
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        
                        # Check bounds
                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            # If neighbor is unlabeled, it's part of the
                            # initial "shoreline" and should be processed.
                            if labels[nr, nc] == UNLABELED:
                                # Push (gradient_value, row, col)
                                heapq.heappush(pq, (gradient_image[nr, nc], nr, nc))
                                # Temporarily mark to avoid duplicates in queue
                                labels[nr, nc] = -2 
        
        # 2. Start the "flooding" process
        while pq:
            # Pop the pixel with the LOWEST gradient (elevation)
            priority, r, c = heapq.heappop(pq)
            
            # Check neighbors to find its label
            neighbor_labels = set()
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    label = labels[nr, nc]
                    if label != UNLABELED and label != WATERSHED_LINE and label != -2:
                        neighbor_labels.add(label)
            
            if not neighbor_labels:
                # Should not happen with this initialization, but as a safety
                # we'll just ignore this pixel.
                labels[r, c] = UNLABELED # Reset
                continue

            if len(neighbor_labels) == 1:
                # This pixel belongs to one basin. Assign its label.
                label = neighbor_labels.pop()
                labels[r, c] = label
                
                # Now, add its unlabeled neighbors to the queue
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        if labels[nr, nc] == UNLABELED:
                            heapq.heappush(pq, (gradient_image[nr, nc], nr, nc))
                            labels[nr, nc] = -2 # Mark as "in queue"
            else:
                # This pixel is where floods from 2+ basins meet.
                # It's a watershed line.
                labels[r, c] = WATERSHED_LINE
                
        # Clean up any -2 "in queue" pixels that were never processed
        # (this can happen at the very edges)
        labels[labels == -2] = UNLABELED 
        
        return labels


    def watershed_segmentation(self, original_image):
        """
        Performs the full Marker-Controlled Watershed workflow.
        Takes the original (preferably 3-channel for display) image
        as an argument to draw the final lines on.
        """
        
        # --- Step 1: Compute Gradient (Topography) ---
        # We run Sobel on 'self.image' (the grayscale version)
        gradient_seg = Segmentation(self.image)
        gradient_image = gradient_seg.sobel_operator()

        # --- Step 2: Generate Binary Mask (Otsu) ---
        otsu_thresh, binary_mask = self.otsu_threshold()

        # --- Step 3: Identify Foreground Markers (Erosion) ---
        
        # We "erode" the binary mask to get "certain" foreground pixels
        # Erode 3 times to get small, certain markers

        kernel = np.ones((3,3), dtype=np.uint8)
        fg_markers_img = cv.erode(binary_mask,kernel, iterations=1)
        
        # Label the separate foreground components
        fg_labels, num_objects = self.connected_components(fg_markers_img)
        print(f"    Found {num_objects} foreground objects.")

        # --- Step 4: Identify Background Markers (Dilation) ---
        # We "dilate" the binary mask to find the "certain" background
    
        bg_markers_img = cv.dilate(binary_mask,kernel, iterations=1)
        
        # Background is everything outside the dilated area
        # Label background as 1
        bg_label = 1
        markers = np.zeros_like(binary_mask, dtype=np.int32)
        markers[bg_markers_img == 0] = bg_label

        # --- Step 5: Combine Markers ---
        # Add foreground markers (labeled 2, 3, ...)
        # Add 1 to all fg_labels so they start at 2 (0 becomes 1, 1 becomes 2, etc.)
        # But wait, _connected_components starts labeling at 1. So we just add 1.
        fg_labels[fg_labels > 0] += bg_label # 1->2, 2->3, etc.
        
        # Combine markers: FG markers (2,3,...) overwrite BG marker (1)
        markers[fg_labels > 0] = fg_labels[fg_labels > 0]
        
        # --- Step 6: Execute Watershed Flood ---
      
        labels = self._watershed_flood(gradient_image, markers)

        # --- Visualization ---
        
        # Create a visual output
        # Draw watershed lines (label = -1) in red on the original image
        output_vis = original_image.copy()
        if len(output_vis.shape) == 2:
            output_vis = cv.cvtColor(output_vis, cv.COLOR_GRAY2BGR)
            
        output_vis[labels == -1] = [0, 0, 255] # BGR for Red

        return output_vis, labels


if __name__ == "__main__":
    original_image = cv.imread('./assets/water_coins.jpg', cv.IMREAD_GRAYSCALE)
  
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
         #                            sigma=1.0, 
          #                           high_threshold=50, 
           #                          low_threshold=25   
            #                    )


        otsu_threshold, otsu_image = seg.otsu_threshold()
        # ---------- Hough transform ------------

        #canny_image_opencv = cv.Canny(original_image, 100, 40)
        #hough_test = Segmentation(canny_image.copy()) # Use .copy() to avoid drawing on the Canny image itself
        #acc, thetha_bins, rho_bins = hough_test.hough_transform()
        
        # Find the edges in the image using canny detector
        #edges = cv.Canny(original_image,220 , 230)
        # Detect points that form a line
        #lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, None, minLineLength=50, maxLineGap=10)
        # Draw lines on the image
        #original  = cv.imread('./assets/street.jpg')
        #img = original.copy()
        # Draw the lines
        #if lines is not None:
           # for i in range(0, len(lines)):
          #      l = lines[i][0]
         #       cv.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        #cv.imshow("Result Image", img)

        original_image_color = cv.imread('./assets/water_coins.jpg', cv.IMREAD_COLOR)
        watershed_viz, watershed_labels = seg.watershed_segmentation(original_image_color)
        #Display images
        cv.imshow('Original image', original_image)
        #cv.imshow('Laplacian image', Hline_image)
        #cv.imshow('Vertical Line Detection image', Vline_image)
        #cv.imshow('Point Detection image', PointImage)
        #cv.imshow('Sobel edge detection image', Sobel_image)
        #cv.imshow('Marr Hildreth', marr_hildreth_image)
        #cv.imshow('Canny', edges)
        #cv.imshow('Hough lines', drawing_seg.image) # Show the image with lines drawn on it
        cv.imshow('Otsu', otsu_image)
        cv.imshow('Watershed Segmentation', watershed_viz)
        #Save images

        #cv.imwrite('./assets/outputs/otsu_water_coins.png',otsu_image)
        #cv.imwrite('./assets/outputs/watershed_coins.png',watershed_viz)
        #cv.imwrite('./assets/outputs/canny_for_hough.png',edges)
        
        cv.waitKey(0)
        cv.destroyAllWindows()
