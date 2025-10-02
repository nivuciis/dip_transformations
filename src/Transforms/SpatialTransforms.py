import numpy as np
import cv2 as cv
from scipy.ndimage import map_coordinates

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
        self.mean = np.mean(image)
        self.std = np.std(image)
        self.var = np.var(image)
    
    @staticmethod
    def create_transform_matrix(tx=0, ty=0,  # Translation
                                     sx=1, sy=1,  # Scalling
                                     angle=0, # Rotation 
                                     cx=0, cy=0): # Sheer
        """
        Create and compute the transformation matrix
        """
        M = np.identity(3, dtype=np.float32)

        #Shear
        if cx != 0 or cy != 0:
            Mshear = np.array([
                [1, cx, 0],
                [cy, 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            M = Mshear @ M 

        #Scalling
        if sx != 1 or sy != 1:
            M_scalling = np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            M = M_scalling @ M

        #Rotation
        if angle != 0:
            theta = np.radians(angle)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            M_rot = np.array([
                [cos_t, -sin_t, 0],
                [sin_t,  cos_t, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            M = M_rot @ M

        # Translation
        if tx != 0 or ty != 0:
            M_trans = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ], dtype=np.float32)
            M = M_trans @ M

        return M

    def _apply_affine_transform(self, transform_matrix, output_shape = None):
        """
        Applies the 3x3 transformation matrix M_transform to the image 
        using inverse mapping and bilinear interpolation (SciPy).
        """
        if output_shape is None:
            output_shape = (self.height, self.width)
            
        rows_dest, cols_dest = output_shape
    
        M_inv = np.linalg.inv(transform_matrix)
        Y_dest, X_dest = np.indices((rows_dest, cols_dest), dtype=np.float32)

        #Necessary to matrix multiplication for M_inv
        homogeneous_coords = np.vstack([
            X_dest.ravel(), 
            Y_dest.ravel(), 
            np.ones_like(X_dest.ravel())
        ])
        homogeneous_coords = M_inv @ homogeneous_coords
        
        #Normalization 
        X_origin = homogeneous_coords[0] / homogeneous_coords[2]
        Y_origin = homogeneous_coords[1] / homogeneous_coords[2]

        coords_mapping = np.array([Y_origin, X_origin])
        
        # Remmapping
        img_transform_map = map_coordinates(
            input=self.image,
            coordinates=coords_mapping,
            order=1,                      
            mode='constant',              
            cval=0.0                      
        )
        
        return img_transform_map.reshape(rows_dest, cols_dest).astype(self.image.dtype)

    def scalling(self, scale_x=1.0, scale_y=1.0):
        """Scale the image by the given factors along x and y axes.
        Scaling is a spatial transformation that changes the size of an image.
        It can be used to enlarge or reduce the dimensions of an image.
        Im using Affine Transformation for this operation
        """
        scale_matrix = self.create_transform_matrix(sx=scale_x, sy=scale_y)
        
        new_height = int(self.height * scale_y)
        new_width = int(self.width * scale_x)
        
        return self._apply_affine_transform(
            scale_matrix, 
            output_shape=(new_height, new_width)
        )

    def rotation(self, angle=0):
        rotation_matrix = self.create_transform_matrix(angle=angle)
    
        return self._apply_affine_transform(rotation_matrix)

    def translation(self, tx=0, ty=0):
        translation_matrix = self.create_transform_matrix(tx=tx, ty=ty)
        
        return self._apply_affine_transform(translation_matrix)
    
#Test
if __name__ == "__main__":

    # Load an image
    image = cv.imread('SanFrancisco.jpg', cv.IMREAD_GRAYSCALE)

    if image is None:
        print("Erro: Imagem 'SanFrancisco.jpg' não encontrada. Verifique o caminho.")
    else:
        spatial_transformer = SpatialTransforms(image)

        scaled_image = spatial_transformer.scalling(scale_x=5.5, scale_y=5.8)
        rotated_image = spatial_transformer.scalling(scale_x=3.5, scale_y=3.8)
        rotated_image = spatial_transformer.rotation(-60)
        translated_image = spatial_transformer.translation(tx=20, ty=50)
        # OpenCV
        cv.imshow('1 - Original Image', image)
        cv.imshow('2 - Scaled Image (x1.5, y0.8)', scaled_image.astype(np.uint8)) 
        cv.imshow('3 - Rotated Image (+30 deg)', rotated_image.astype(np.uint8))
        cv.imshow('4 - Translated Image (+50, +50)', translated_image.astype(np.uint8))
        cv.waitKey(0) 
        cv.destroyAllWindows()
                
                
    
