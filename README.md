# dip_transformations
This repo contains some useful image transformations to understand spatial filtering and transforms with OpenCV


# Notes on digital image processing 
## Basic relationships between pixels

**The set of 4-neighbors of pixel p is defined as**

```math
n4(p) = (x + 1, y),(x − 1, y),(x, y + 1),(x, y − 1)
```
    
**The set of 4-diagonal neighbors of pixel p is defined as**
```math
nd4(p) = (x + 1, y + 1),(x − 1, y − 1),(x + 1, y − 1),(x − 1, y + 1)
```
Neighbor coordinates may fall outside the image range so we can set it as 0 in, for example, median blur filtering.
    
## Math tools
**---> Array operation is made pixel-by-pixel.**

**---> Matrix operations are carried out using matrix theory.**

**---> Henceforth assume array operation unless stated otherwise**

**---> The arithmetic operations (+, −, ×, ÷) are performed
pixelwise in array operations, between images of the same size.** 

**---> The image resulting from arithmetic operations should(often) be scaled to [0, 255]!**

## Affine transformations for Spatial transforms
**A general form that can scale, translate, rotate or sheer a set of
coordinate points is the Affine transform:**


**Scalling**

$\begin{bmatrix}Cx & 0 & 0 \\0 & Cy & 0\\0&0&1 \end{bmatrix}$