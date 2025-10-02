import matplotlib.pyplot as plt
import numpy as np
import os
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
        self.mean = np.mean(image)
        self.std = np.std(image)
        self.var = np.var(image)
    def getHistogram(self):
        """Compute the histogram of the image."""
        histogram = np.zeros(256, dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                pixel_value = self.image[i, j]
                histogram[pixel_value] += 1
        return histogram
    def plotHistogram(self, second_image=None):
        """Plot the histogram of the image."""
        histogram = self.getHistogram()
        plt.figure(figsize=(10, 5))
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.xlim([0, 256])
        plt.bar(range(256), histogram, width=1.0, color='black')
        if second_image is not None:
            second_histogram = IntensityTransforms(second_image).getHistogram()
            plt.bar(range(256), second_histogram, width=1.0, color='red', alpha=0.5)
            plt.legend(['First Image', 'Second Image'])
        plt.show()
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
    def log_transform(self, c=1):
        """Apply logarithmic transformation to the image.
        The logarithmic transformation is used to enhance the details in the darker regions of an image.
        It compresses the dynamic range of the pixel values, making it useful for images with a wide range of intensity levels.
        s = c * log(1 + r)
        where s is the output pixel value, r is the input pixel value, and c is a scaling constant.
        """
        # Ensure the image is in float format to prevent overflow
        image_float = self.image.astype(float)
        log_image = c * np.log1p(image_float)
        return log_image.astype(self.image.dtype)
    def gamma_correction(self, gamma=1.0, c=1.0):
        """Apply gamma correction to the image.
        Gamma correction is a non-linear operation used to adjust the brightness of an image.
        It is particularly useful for correcting the brightness of images for display on different devices.
        s = c * r^gamma
        where s is the output pixel value, r is the input pixel value, c is a scaling constant, and gamma is the gamma value.
        """
        # Normalize the image to the range [0, 1]
        normalized_image = self.image / 255.0
        gamma_corrected_image = c * np.power(normalized_image, gamma)
        # Scale back to [0, 255]
        gamma_corrected_image = np.clip(gamma_corrected_image * 255, 0, 255)
        return gamma_corrected_image.astype(self.image.dtype)
    def histogram_equalization(self):
        """Apply histogram equalization to the image.
        Histogram equalization is a technique used to improve the contrast of an image by redistributing the intensity values.
        It enhances the global contrast of the image, making it easier to distinguish between different intensity levels.
        """
        histogram, bins_source = np.histogram(self.image.flatten(), bins=256, range=[0,256])
        cdf = np.cumsum(histogram)
        cdf_normalized = cdf / cdf.max() * 255 # Normalize to 0-255
        equalized_image = np.interp(self.image.flatten(), range(256), cdf_normalized)
        return equalized_image.reshape(self.image.shape).astype(self.image.dtype)
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
    def piecewise_linear(self, r1, s1, r2, s2):
        """Apply piecewise linear transformation to the image.
        Piecewise linear transformation is used to enhance the contrast of an image by mapping input intensity values to output intensity values using linear segments.
        The transformation is defined by two points (r1, s1) and (r2, s2) that specify the mapping of input intensities to output intensities.
        """
        output_image = np.zeros_like(self.image)
        for i in range(self.height):
            for j in range(self.width):
                pixel_value = self.image[i, j]
                if 0 <= pixel_value < r1:
                    output_image[i, j] = (s1 / r1) * pixel_value
                elif r1 <= pixel_value < r2:
                    output_image[i, j] = ((s2 - s1) / (r2 - r1)) * (pixel_value - r1) + s1
                else:
                    output_image[i, j] = ((255 - s2) / (255 - r2)) * (pixel_value - r2) + s2 if r2<255 else pixel_value
        return output_image
    
#Test the class
if __name__ == "__main__":
    # Load an image
    image = cv.imread('moon.jpg', cv.IMREAD_GRAYSCALE)
    # Create an instance of the IntensityTransforms class
    intensity_transformer = IntensityTransforms(image)
    # Apply piecewise linear transformation
    piecewise_image = intensity_transformer.piecewise_linear(r1=70, s1=0, r2=140, s2=255)
    #histogram gamma correction
    hist_eq_image = intensity_transformer.piecewise_linear(r1=0, s1=0, r2=150, s2=200)
    #hist_eq_image = intensity_transformer.gamma_correction(gamma=1.8, c=1)
    # Display the original and piecewise linear transformed images
    cv.imshow('Original Image', image)
    cv.imshow('histogram equalized Image', hist_eq_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #plot histogram
    intensity_transformer.plotHistogram(second_image=hist_eq_image)
