import numpy as np
import cv2 as cv

class SpatialTransforms:
    """Class for performing spatial transformations on images.
    Spatial transformations involve changing the spatial arrangement of pixels in an image, such as translation, rotation, and scaling.
    Image resolution is the largest number of discernible line pairs per unit distance
    The arithmetic operations (+, −, ×, ÷) are performed
    pixelwise in array operations, between images of the same size.

    Attributes:
        image (numpy.ndarray): The input image.
        width (int): The width of the image.
        height (int): The height of the image.
    Methods:
    
    """
    def __init__(self, image):
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
    def scalling(self, scale_x=1.0, scale_y=1.0):
        """Scale the image by the given factors along x and y axes.
        Scaling is a spatial transformation that changes the size of an image.
        It can be used to enlarge or reduce the dimensions of an image.
        Im using Affine Transformation for this operation
        """
        #Create the scaling matrix
        scaling_matrix = np.array([[scale_x, 0, 0],
                                   [0, scale_y, 0],
                                   [0, 0, 1]])
        corners = np.array([[0, 0, 1],
                            [self.width, 0, 1],
                            [0, self.height, 1],
                            [self.width, self.height, 1]])
        transformed_corners = np.dot(corners,scaling_matrix.T)
        min_x = int(np.min(transformed_corners[:, 0]))
        max_x = int(np.max(transformed_corners[:, 0]))
        min_y = int(np.min(transformed_corners[:, 1]))
        max_y = int(np.max(transformed_corners[:, 1]))
        output_shape = (max_y - min_y, max_x - min_x)

        inv_matrix = np.linalg.inv(scaling_matrix)

        #if its an rgb image
        if len(self.image.shape) == 3:
            output_image = np.zeros((output_shape[0], output_shape[1], self.image.shape[2]), dtype=self.image.dtype)
        else:
            output_image = np.zeros((output_shape[0], output_shape[1]), dtype=self.image.dtype)
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                original_coords = np.dot(inv_matrix, np.array([j + min_x, i + min_y, 1]))
                original_x, original_y = int(original_coords[0]), int(original_coords[1])
                #performing bilinear interpolation
                if 0 <= original_x < self.width -1 and 0 <= original_y < self.height -1 :
                    x_floor, y_floor = np.floor(original_x).astype(int), np.floor(original_y).astype(int)
                    x_fraq, y_fraq = original_x - x_floor, original_y - y_floor
                    #Get the four neighboring pixels
                    top_left = self.image[y_floor, x_floor]
                    top_right = self.image[y_floor, min(x_floor + 1, self.width - 1)]    
                    bottom_left = self.image[min(y_floor + 1, self.height - 1), x_floor]
                    bottom_right = self.image[y_floor + 1, min(x_floor + 1, self.width - 1)]
                #Bilinear interpolation formula
                    #RGB
                    if len(self.image.shape) == 3:
                        top = (1 - x_fraq) * top_left + x_fraq * top_right
                        bottom = (1 - x_fraq) * bottom_left + x_fraq * bottom_right
                        pixel_value = (1 - y_fraq) * top + y_fraq * bottom
                        output_image[i, j] = np.clip(pixel_value, 0, 255)
                    else:
                        top = (1 - x_fraq) * top_left + x_fraq * top_right
                        bottom = (1 - x_fraq) * bottom_left + x_fraq * bottom_right
                        pixel_value = (1 - y_fraq) * top + y_fraq * bottom
                        output_image[i, j] = np.clip(pixel_value, 0, 255)
        return output_image
    
#FUCNTION TO TEST THE CLASS
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2 as cv

    # Load an image
    image = cv.imread('SanFrancisco.jpg', cv.IMREAD_GRAYSCALE)

    # Create an instance of the SpatialTransforms class
    spatial_transformer = SpatialTransforms(image)

    # Perform scaling
    scaled_image = spatial_transformer.scalling(scale_x=2.2, scale_y=1.4)

    # Display the original and scaled images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Scaled Image')
    plt.imshow(scaled_image, cmap='gray')
    plt.axis('off')

    plt.show()
                
                
    
