import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
import time

class FrequencyFilters:
    """Class for applying frequency domain filters to images.
    Frequency domain filtering involves modifying the frequency components of an image to achieve effects such as blurring, sharpening, and noise reduction.
    The FFT (Fast Fourier Transform) is used to convert the image to the frequency domain, where filters can be applied.
    """
    def __init__(self, image):
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.mean = np.mean(image)
        self.std = np.std(image)
        self.var = np.var(image)
        self.FFT = np.fft.fft2(image)
    def FourierTransform(self):
        """Compute ,without using numpy fft, the 2D Fourier Transform (FT) of the image and shift the zero frequency component to the center."""
        #Line by Line 1DFT
        dft_lines = np.zeros((self.height, self.width), dtype=np.complex128)
        term = 0.0j
        for y in range(self.height):
            for u in range(self.width):
                complex_sum = 0.0j
                for x in range(self.width):
                    pixel_val = self.image[y][x]
                    term = pixel_val * np.exp(-2j * np.pi * u * x / self.width)
                    complex_sum += term
                dft_lines[y,u] = complex_sum
        
        #Column by Column 1DFT
        dft_final = np.zeros((self.height, self.width), dtype=np.complex128)
        for x in range(self.width):
            for v in range(self.height):
                complex_sum = 0.0j
                for y in range(self.height):
                    pixel_val = dft_lines[y][x]
                    term = pixel_val * np.exp(-2j * np.pi * v * y / self.height)
                    complex_sum += term
                dft_final[v,x] = complex_sum

        dft_final = np.fft.fftshift(dft_final)
        magnitude_spectrum = 20 * np.log(np.abs(dft_final) + 1)  
        magnitude_spectrum = cv.normalize(magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        return magnitude_spectrum
    def FFTTransform(self):
        """Compute the 2D Fast Fourier Transform (FFT) of the image and shift the zero frequency component to the center."""
        f_transform = np.fft.fft2(self.image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)  
        magnitude_spectrum = cv.normalize(magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        return magnitude_spectrum
    
    def InverseFFT(self, fft):
        """Compute the inverse 2D Fast Fourier Transform (IFFT) to convert back to the spatial domain."""
        f_ishift = np.fft.ifftshift(fft)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        return img_back

    def FrequencyFilter(self, filter_type='lowpass', cutoff=(30,0)):
        """Apply a frequency domain filter to the image.
        
        Parameters:
        filter_type (str): Type of filter to apply ('lowpass','highpass' or 'bandpass').
        cutoff (int): Cutoff frequency for the filter.
        
        Returns:
        np.ndarray: The filtered image in the spatial domain.
        """
        # Create a mask for the filter and calculate its center
        dft_shifted = np.fft.fftshift(self.FFT)
        center_row, center_col = self.width // 2, self.height // 2
        cutoff_lp, cutoff_hp = cutoff
        x = np.arange(self.width)
        y = np.arange(self.height)
        x_grid, y_grid = np.meshgrid(x, y)
        #Euclidean distance from the center
        dist_from_center = np.sqrt((x_grid - center_col)**2 + (y_grid - center_row)**2)
        #Compute masks
        mask_lp = np.zeros_like(self.image, dtype=np.uint8)
        mask_hp = np.zeros_like(self.image, dtype=np.uint8)
        mask_bp = np.zeros_like(self.image, dtype=np.uint8)

        if(filter_type == 'lowpass'):
            mask_lp = (dist_from_center <= cutoff_lp).astype(np.uint8)
            dft_lp_filtered = dft_shifted * mask_lp
            return self.InverseFFT(dft_lp_filtered)
        elif(filter_type == 'highpass'):
            mask_hp = (dist_from_center > cutoff_hp).astype(np.uint8)
            dft_hp_filtered = dft_shifted * mask_hp 
            return self.InverseFFT(dft_hp_filtered)
        elif(filter_type == 'bandpass'):
            mask_bp = ((dist_from_center <= cutoff_lp) & (dist_from_center > cutoff_hp)).astype(np.uint8)
            dft_bp_filtered = dft_shifted * mask_bp
            return self.InverseFFT(dft_bp_filtered)
        else:
            raise ValueError("Invalid filter type. Choose 'lowpass', 'highpass' or 'bandpass'.")


#Test
if __name__ == "__main__":

    # Load an image
    image = cv.imread('astronaut.png', cv.IMREAD_GRAYSCALE)
    #To compute manual DFT faster, resize the image to a smaller size
    #small_image = cv.resize(image, (340, 290))
    if image is None:
        print("Erro: Image not found")
    else:
        # Create an instance of FrequencyFilters
        freq_filter = FrequencyFilters(image)
        start_time = time.time()
        f_transform = freq_filter.FrequencyFilter(filter_type='highpass', cutoff=(10,70))
        end_time = time.time()
        print(f"Compute time: {end_time - start_time:.2f} s")
        output_filename = 'assets/highpass.png'
        #cv.imwrite(output_filename, f_transform)
        
        cv.imshow('Original Image', image)
        cv.imshow('Fourier Transform', f_transform)
        cv.waitKey(0)
        cv.destroyAllWindows()
    