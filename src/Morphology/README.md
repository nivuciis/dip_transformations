# Morphology in Digital Image Processing
This document provides an overview of fundamental and compound morphological operations for image analysis, based on the principles of mathematical morphology. Morphology is a non-linear technique used to process images based on their shapes. It is a powerful tool for extracting image components that are useful for shape representation and description, such as boundaries, skeletons, and convex hulls. 

Morphology is a set of image processing operations that process images based on shapes. It is commonly applied to binary images but can also be used on grayscale images. Morphological techniques use a structuring element to probe and transform the input image.

## Common Morphological Operations

- **Erosion**: Removes pixels on object boundaries, making objects thinner and eliminating small details.
- **Dilation**: Adds pixels to object boundaries, making objects larger and filling small holes.
- **Opening**: Erosion followed by dilation; removes small objects or noise from the foreground.
- **Closing**: Dilation followed by erosion; fills small holes and gaps in the foreground objects.
- **Morphological Gradient**: The difference between dilation and erosion, highlighting object edges.
- **Top-hat Transform**: Extracts small elements and details from images.
- **Black-hat Transform**: Reveals small holes and dark regions on a bright background.

## Set theory

The foundation of mathematical morphology is set theory. In this context, objects within a binary image are treated as sets, where each element is a coordinate pair (x, y) corresponding to a foreground pixel.  All morphological operations are therefore based on set logic.

## Structuring Element

A structuring element is a small shape or template used to interact with the image. Common shapes include squares, disks, and crosses. The choice of structuring element affects the outcome of morphological operations.

## Fundamental operations
![Original Image](./assets/horse2.png)
### Dilation
Dilation is an operation that **expands or thickens** the foreground objects in an image. It is defined as:
$$A\oplus B=\{z|(\hat{B})_{z}\cap A\ne\emptyset\}$$
The structuring element `B` is passed over the image `A`. [cite_start]The result is the set of all points where the reflected SE has a non-empty intersection with the original object[cite: 690]. Intuitively, it adds pixels to the boundaries of objects.

![Dilate image](./assets/outputs/dilate.png)

**Primary Use Case**: Dilation is commonly used for **bridging gaps** in an object that may have resulted from poor resolution or noise.

### Erosion
Erosion is the dual operation to dilation; it **shrinks or thins** foreground objects. It is defined as:
$$A\ominus B=\{z|(B)_{z}\subseteq A\}$$

![Erode image](./assets/outputs/erode.png)

The result is the set of all points where the structuring element `B` can fit completely inside the original object `A`. Erosion effectively strips away a layer of pixels from the boundaries of objects.

**Primary Use Case**: Erosion is used to **remove small, irrelevant components** or noise from an image.

# Compound Operations

By combining dilation and erosion, we can create more powerful and less destructive operations for image cleaning and feature extraction.

### Opening
The opening operation is defined as an **erosion followed by a dilation** using the same structuring element.
$$A\circ B=(A\ominus B)\oplus B$$

![Opening image](./assets/outputs/opening.png)

Opening generally **smoothes the contours** of an object, **breaks thin connections**, and **eliminates small islands** of pixels (often called "salt" noise). It is considered **less destructive than a simple erosion** because the subsequent dilation helps to restore the shape of the larger objects that survived the erosion.

### Closing
The closing operation is the dual of opening, defined as a **dilation followed by an erosion**.
$$A\bullet B=(A\oplus B)\ominus B$$

![Closing image](./assets/outputs/closing.png)

Closing also tends to smooth contours but is primarily used to **fuse narrow breaks**, fill long, thin gulfs, and **eliminate small holes** within objects ("pepper" noise). **It is less destructive than a simple dilation**.


# Hit or miss
The morphological hit-or-miss transform (HMT) is a basic tool for shape detection.
Let I be a binary image composed of foreground (A) and background pixels, respectively.
the HMT utilizes two structuring elements: B1, for detecting shapes in the foreground, and B2 , for detect-
ing shapes in the background. The HMT of image I is defined as
$$A\otimes B=(A\ominus B_{1})\cap(A^{c}\ominus B_{2})$$

**B1 found a match in the foreground** (i.e., B1 is contained in A) and **B2 found a match in the background** (i.e., B2 is contained in Ac ). The word “simultaneous” implies that z is the same translation
of both structuring elements. The word “miss” in the HMT arises from the fact that B2 finding a match in Ac is the same as B2 not finding (missing) a match in A

**A match is found only at locations where the "Hit" kernel fits perfectly within the foreground of the image **AND** the "Miss" kernel fits perfectly within the background of the image at the same time.**
* In erosion only the 1's in the kernel should match the 0's doesn't matter

## Example border detection with hit or miss
For kernels:
```python
hit_kernel = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0]], dtype=np.uint8) 

miss_kernel = np.array([[0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1]], dtype=np.uint8) 
```

![Border detection](./assets/outputs/hit_or_miss.png)

# Boundary Extraction
The boundary of a set A of foreground pixels, denoted by $\beta(A)$ can be obtained by
**first eroding A by a suitable structuring element B, and then performing the set difference between A and its erosion**. That is:

$\beta(A) = A - (A \ominus B)$

![Boundary Extraction](./assets/outputs/boundary_extraction.png)


## Top-Hat Transform

The Top-Hat transform is defined as the difference between an original image and its morphological opening.

* **Formula**: `Top-hat = Original Image - Opening(Image)` 
* Primary Application: It is highly effective for **extracting light objects from an unevenly illuminated dark background**, a common problem known as shading correction.

## Bottom-Hat Transform

The Bottom-Hat transform is the dual of the Top-Hat. It is defined as the difference between the morphological closing of an image and the original image itself.

* **Formula**: `Bottom-hat = Closing(Image) - Original Image`
* Primary Application: **It is used to extract dark objects or details situated on a bright background**.



## Applications

- Noise removal
- Object extraction
- Image segmentation
- Shape analysis
- Edge detection
- Skeletonization
- Feature extraction



