import cv2
import numpy as np
import io
from PIL import Image

def process_mask(mask_image: Image.Image, feather_radius: int = 5) -> Image.Image:
    """
    Applies Gaussian blur to the mask to create a feathered edge.
    
    Args:
        mask_image (PIL.Image): The binary mask image (black/white).
        feather_radius (int): The radius of the Gaussian blur.
        
    Returns:
        PIL.Image: The feathered mask.
    """
    # Convert PIL Image to NumPy array
    mask_np = np.array(mask_image.convert("L"))
    
    # Apply Gaussian Blur
    # kernel size must be odd, so we do 2*r + 1
    ksize = (feather_radius * 2 + 1, feather_radius * 2 + 1)
    blurred_mask = cv2.GaussianBlur(mask_np, ksize, 0)
    
    # Convert back to PIL Image
    return Image.fromarray(blurred_mask)

def resize_for_model(image: Image.Image, size: tuple = (1024, 1024)) -> Image.Image:
    """
    Resizes the image to the target size using high-quality resampling.
    
    Args:
        image (PIL.Image): Input image.
        size (tuple): Target width and height.
        
    Returns:
        PIL.Image: Resized image.
    """
    return image.resize(size, Image.Resampling.LANCZOS)
