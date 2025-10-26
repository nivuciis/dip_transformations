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
coordinate points is the Affine transforms:**

First, we need to compute the transformation matrix, where the inputs are geometric parameters such as angles, scale factors, and the image (from which the dimensions are extracted), and the output is a 3×3 transformation matrix in homogeneous coordinates.

After obtained the transformation matrix we need to compute the inverse maping matrix ( instead of taking every pixel of the image and put it on a floating position in the result image we do the opposite)

**Scalling**

Changes the size of the image by a factor of sx along the x-axis and sy along the y-axis.

```python
    S = [[s_x, 0, 0],
          [0, s_y, 0],
          [0, 0, 1]]
```

**Translating**

Moves the image by tx pixels along the x-axis and ty pixels along the y-axis.
```python
    T = [[1, 0, t_x],
          [0, 1, t_y],
          [0, 0, 1]]
```

**Rotation**

Rotates the image by an angle θ around the origin (0, 0). To rotate around a different point, you first translate the image to bring the point to the origin, rotate, and then translate it back.

```python
    R = [[cos, -sin, 0],
          [sin, cos, 0],
          [0, 0, 1]]
```
**Sheer**

Slants the image. A horizontal shear shifts pixels in a row based on their y-coordinate, while a vertical shear shifts pixels in a column based on their x-coordinate. The factors sh_x and sh_y control the amount of shear.

Horizontal
```python
    Hs = [[1, sh_x, 0],
          [0, 1, 0],
          [0, 0, 1]]
```
Vertical
```python
    Vs = [[1, 0, 0],
          [sh_y, 1, 0],
          [0, 0, 1]]
```