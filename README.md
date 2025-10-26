# Digital Image Processing 

This repository contains a collection of Python scripts for a digital image processing (DIP) class. The focus is on demonstrating common image transformations, spatial filtering, segmentation, mathematical morphology, and frequency-domain transforms using the OpenCV library.

---

## Purpose

The primary goal of this repository is to serve as a practical resource for students learning the fundamentals of digital image processing. The code is intended to be simple, well-commented, and easy to understand, providing clear examples of key DIP concepts.

---

## Key Concepts Covered

The scripts in this repository explore several fundamental areas of image processing:

### 1. Spatial Filtering
Operations that directly manipulate the pixel values of an image.
* **Smoothing Filters:** (e.g., Gaussian Blur, Median Blur) used for noise reduction.
* **Sharpening Filters:** (e.g., Laplacian, Sobel) used for edge detection and enhancing details.
* **Image Gradients:** Finding the directional change in the intensity or color of an image.

### 2. Mathematical Morphology
Operations that probe an image with a small shape or template called a "structuring element."
* **Erosion:** Shrinks the boundaries of foreground objects.
* **Dilation:** Expands the boundaries of foreground objects.
* **Opening:** An erosion followed by a dilation, useful for removing small "salt" noise.
* **Closing:** A dilation followed by an erosion, useful for filling small holes or "pepper" noise.

### 3. Image Segmentation
The process of partitioning a digital image into multiple segments (sets of pixels, also known as super-pixels).
* **Thresholding:** (e.g., Simple, Adaptive, Otsu's) for separating objects from the background based on pixel intensity.
* **Edge-Based Segmentation:** Using detected edges (like from Canny or Sobel) to find object boundaries.
* **Watershed Algorithm:** A region-based segmentation approach.

### 4. Frequency Domain Transforms
Operations that first transform an image into its frequency representation.
* **Fourier Transform (FFT):** Used to analyze the frequency components of an image.
* **Frequency Domain Filtering:** Implementing low-pass and high-pass filters to smooth or sharpen images.

---

## Technologies Used

* **Python 3.x**
* **OpenCV:** The primary library used for all image processing tasks.
* **NumPy:** For numerical operations and array manipulation.
* **Matplotlib:** For displaying images and plots.

---

## Usage

It is highly recommended to use the **Anaconda** distribution to manage your Python environment and dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nivuciis/dip_transformations
    cd dip_transformations
    ```

2.  **Create and activate a new Conda environment:**
    (You can name the environment anything you like, e.g., `dip_env`)
    ```bash
    conda create -n dip_env python=3.9
    conda activate dip_env
    ```

3.  **Install the required libraries:**
    The simplest way is to use `pip` within your active conda environment:
    ```bash
    pip install opencv-python numpy matplotlib
    ```
    Alternatively, you can install them using the `conda-forge` channel:
    ```bash
    conda install -c conda-forge opencv numpy matplotlib
    ```

4.  **Run a script:**

    ```bash
    python Segmentation.py
    ```

---

