## Funtamental of image segmentation 
Let R represent the entire spatial region occupied by an image. We may view image
segmentation as a process that partitions R into n subregions, the segmentation process can be based on **discontinuity or similarity**.

---
### Similarity or Region-based segmentation

*about the similarity* we have the region-base segmentation as it follows; $R_{1}, R_{2},..., R_{n}$ such
that: 

(a) 

$$ \bigcup_{i=1}^{n} R_i = R. $$

(b)  

$$ R_i, \text{is a connected set, for}\space  i = 0, 1, 2, \ldots, n $$

(c)  

$$ R_i \cap R_j = \varnothing \quad \text{For all } i \text{and } j, \, i \ne j. $$

(d) 

$$ Q(R_i) = \text{TRUE} \quad \text{for } i = 0, 1, 2, \ldots, n $$

(e)  

$$ Q(R_i \cup R_j) = \text{FALSE} \quad \text{For any adjacent regions } R_i \text{ and } R_j. $$

* If the set formed by the union of two
regions is not connected, the regions are said to **disjoint**.

Condition **(a)** indicates that the segmentation must be complete, in the sense that **every pixel must be in a region**. 

Condition **(b)** requires that **points in a region be connected in some predeﬁned sense** (e.g., the points must be 8-connected). 

Condition **(c)** says that **the regions must be disjoint.** 

Condition **(d)** deals with the properties that must be satisﬁed by the pixels in a segmented region for example, $Q(R_{i}) = TRUE$
if all pixels in $R_{i}$ have the same intensity. 

Finally, condition **(e)** indicates that **two adjacent regions $R_{i}$ and $R_{j}$ must be different in the sense of predicate** Q.† (Transpose)

* **VERY IMPORTANT DEFINITION**: The **logical predicate Q** is an homogeneous rule that defines what makes the pixels in a region similar based on similarities criteria such as " All the pixels in $R_{i}$ must have the same intensity value " 

---
### Discontinuity or Edge-base segmentation 
This type of segmentation follows implement a detection based on local discontinuities on an image.

Prior to that we need to introduce the types of discontinuities we can have in an image. The three types of image characteristics in which we are interested are isolated points, lines, and edges.

* **Edge** pixels are pixels at which
the intensity of an image changes abruptly.
* A **line** may be viewed as a (typically) thin edge segment in which the intensity of the background on either side of the line is either much higher or much lower than the intensity of the line
pixels.
* An **isolated point** may be viewed as a foreground (background) pixel surrounded by background (foreground) pixels. 

$ \rightarrow $ It is intuitive that **abrupt, local changes in intensity can be detected using derivatives.** For reasons that will become evident shortly, first- and
second-order derivatives are particularly well suited for this purpose.

In what follows, we compute *intensity differences* using just a few terms of the Taylor
series. For first-order derivatives we use only the linear terms, and we can form differences in
one of three ways.

---

**The forward difference**:

$$
\frac{\partial f(x)}{\partial x} = f'(x) = f(x + 1) - f(x)
$$
 
**The backward difference** :

$$
\frac{\partial f(x)}{\partial x} = f'(x) = f(x) - f(x - 1)
$$

**the central difference** :

$$
\frac{\partial f(x)}{\partial x} = f'(x) = \frac{f(x + 1) - f(x - 1)}{2}
$$

**The second order derivative** based on a central difference,  
$\dfrac{\partial^2 f(x)}{\partial x^2}$, is obtained by:

$$
\frac{\partial^2 f(x)}{\partial x^2} = f''(x) = f(x + 1) - 2f(x) + f(x - 1)
$$

**Points are detected based on the Laplacian:**

$$
\nabla^2 f(x, y) = \frac{\partial^2 f(x, y)}{\partial x^2} + \frac{\partial^2 f(x, y)}{\partial y^2}
$$

$$
\nabla^2 f(x, y) = f(x+1, y) + f(x-1, y) + f(x, y+1) + f(x, y-1) - 4f(x, y)
$$

---
$ \rightarrow $ **Conclusion**: In summary, we arrive at the following conclusions: 

(1) **First-order derivatives generally produce thicker edges**. 

(2) Second-order derivatives have a **stronger response to
ﬁne detail, such as thin lines, isolated points, and noise**. 

(3) Second-order derivatives produce a double-edge response at ramp and step transitions in intensity. 

(4) The **sign of the second derivative** can be used to determine whether a **transition into** an edge is from** light to dark or dark to light.**

To find derivatives we scan the image using a mask: The approach of choice for computing ﬁrst and second derivatives at every pixel location in an image is to use spatial convolution. For the 3x3 ﬁlter kernel, the procedure is to compute the sum of products of the kernel coefﬁcients with the intensity values in the region encompassed by the kernel

$$
Z = w_1 z_1 + w_2 z_2 + \dots + w_9 z_9 = \sum_{k=1}^{9} w_k z_k
$$

Considering the original image: 


![Sudoku](./assets/sudoku2.png)

###  **Point** Extraction kernels

These masks can be implemented as follows:

```python
A = [
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
]

B = [
    [ 1,  1,  1],
    [ 1, -8,  1],
    [ 1,  1,  1]
]

C = [
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
]

D = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]

```
**Descriptions:**
- (a) Filter mask used to implement.  
- (b) Mask used to implement an extension of this equation that includes the diagonal terms.  
- (c) and (d) Two other implementations of the Laplacian found frequently in practice.

### **Line** detection kernels  

Detection angles are with respect to the axis system,  
with positive angles measured counterclockwise with respect to the (vertical) *x*-axis.

```python
horizontal = [
    [-1, -1, -1],
    [ 2,  2,  2],
    [-1, -1, -1]
]

45_positive = [
    [ 2, -1, -1],
    [-1,  2, -1],
    [-1, -1,  2]
]

vertical = [
    [-1,  2, -1],
    [-1,  2, -1],
    [-1,  2, -1]
]

45_negative = [
    [-1, -1,  2],
    [-1,  2, -1],
    [ 2, -1, -1]
]
```
```python
#Laplacian kernel
        Hline_detection_kernel = np.array([[-1, -1, -1],
                                            [2, 2, 2],
                                           [-1, -1, -1]])

        Vline_detection_kernel = np.array([[-1, 2, -1],
                                           [-1, 2, -1],
                                           [-1, 2, -1]])
```

Example 1 - Horizontal line kernel extraction :

![Horizontal kernel](./assets/outputs/Hline_image.png)

Example 2 - Vertical line kernel extraction :

![Vertical kernel](./assets/outputs/Vline_image.png)

### Image gradient

The tool of choice for finding edge strength *and* direction at an arbitrary location $(x, y)$ of an image, $f$, is the *gradient*, denoted by $\nabla f$ and defined as the vector

$$
\nabla f(x, y) = \text{grad}[f(x, y)] = \begin{bmatrix} g_x(x, y) \\ g_y(x, y) \end{bmatrix} = \begin{bmatrix} \frac{\partial f(x, y)}{\partial x} \\ \frac{\partial f(x, y)}{\partial y} \end{bmatrix} $$

 When evaluated for all applicable values of $x$ and $y$, $\nabla f(x, y)$ becomes a *vector image*, each element of which is a vector. The *magnitude*, $M(x, y)$, of this gradient vector at a point $(x, y)$ is given by its Euclidean vector norm:
$$
M(x, y) = \|\nabla f(x, y)\| = \sqrt{g_x^2(x, y) + g_y^2(x, y)} 
$$

* IMPORTANT NOTE : A standard grayscale image can only display one value (a scalar) for each pixels intensity, from black to white. We can't directly plot the 2D vector $[g_x, g_y]$ or the complex number $R + jI$(Fourier for example) as a single, intuitive intensity value. **Therefore, the most common and visually informative way to "see" the result is to plot its magnitude**. **When we plot the gradient magnitude ($M$), we are visualizing the edge strength. Bright areas in this "gradient image" correspond to strong edges in the original image.**


The *direction* of the gradient vector at a point $(x, y)$ is given by

$$
\alpha(x, y) = \tan^{-1}\left[\frac{g_y(x, y)}{g_x(x, y)}\right] 
$$

Angles are measured in the counterclockwise direction with respect to the $x$-axis. This is also an image of the same size as $f$, created by the elementwise division of $g_y$ and $g_x$ over all applicable values of $x$ and $y$. 

## Steps for edge detection

Since the derivatives are sensitive to noise and there are noisy images that we cannot detect easily there are three steps to perfom **edge detection**:

1. **Image smoothing** for noise reduction(Low-pass filters such as median blur are useful here).
2. **Detection of edge points.** This is a local operation that extracts from an image all points that are potential edge-point candidates(Gradient $\nabla$).
3. **Edge localization.** The objective of this step is to select from the candidate points only the points that are members of the set of points comprising an edge.












