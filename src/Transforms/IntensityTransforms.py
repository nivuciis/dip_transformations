import numpy as np
import cv2 as cv

class IntensityTransforms:
    """Class for performing intensity transformations on images.
    Intensity transformations are point operations that change the intensity of each pixel in an image based on a specific function.
    Intensity resolution is the smallest discernible change in intensity level.
    Attributes:
        image (numpy.ndarray): The input image.
        width (int): The width of the image.
        height (int): The height of the image.
    
    Methods:
        negative(L=256): Compute the negative of the image.
        median_blur(kernel_size=3): Apply median blur to the image.
    
    """
    def __init__(self, image):
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]

    def negative(self, L=256):
        """Compute the negative of the image. s = T(p) = (L - 1) - p 
        in which L is the maximum intensity value. and p is the pixel value."""
        return (L-1) - self.image
    
    def median_blur(self, kernel_size=3):
        """Apply median blur to the image.
        Median blur is a non-linear filter used to reduce noise in an image. 
        It replaces each pixel's value with the median value of the intensities 
        in its neighborhood defined by the kernel size.
        Good for filtering high valued noises
        """
        padded_image = np.pad(self.image, pad_width=kernel_size//2, mode='edge')
        for i in range(self.height):
            for j in range(self.width):
                kernel_region = padded_image[i:i+kernel_size, j:j+kernel_size]
                self.image[i, j] = np.median(kernel_region)
        return self.image
    def laplacian_filter(self):
        """Apply Laplacian filter to the image.
        The Laplacian filter is a second-order derivative filter used to enhance edges in an image.
        It highlights regions of rapid intensity change, making it useful for edge detection.
        """
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        padded_image = np.pad(self.image, pad_width=1, mode='edge')
        output_image = np.zeros_like(self.image)
        for i in range(self.height):
            for j in range(self.width):
                region = padded_image[i:i+3, j:j+3]
                output_image[i, j] = np.clip(np.sum(region * laplacian_kernel), 0, 255)
        return output_image
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import cv2 as cv
    # Load an image
    image = cv.imread('SanFrancisco.jpg', cv.IMREAD_GRAYSCALE)
    # Create an instance of the IntensityTransforms class
    intensity_transformer = IntensityTransforms(image)
    # Perform negative transformation
    negative_image = intensity_transformer.negative()
    # Perform median blur
    median_blurred_image = intensity_transformer.median_blur(kernel_size=5)
    # Perform Laplacian filtering
    laplacian_image = intensity_transformer.laplacian_filter()
    # Display the original and transformed images
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title('Negative Image')
    plt.imshow(negative_image, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title('Median Blurred Image')
    plt.imshow(median_blurred_image, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title('Laplacian Filtered Image')
    plt.imshow(laplacian_image, cmap='gray')
    plt.axis('off')
    plt.show()

    