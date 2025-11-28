import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input

def preprocess_image_for_captioning(image_path, target_size=(299, 299)):
    """
    Preprocess image for captioning model (InceptionV3 input format)
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
    
    Returns:
        np.ndarray: Preprocessed image array
    """
    try:
        # Load image
        img = load_img(image_path, target_size=target_size)
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Expand dimensions to match batch format
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for InceptionV3
        img_array = preprocess_input(img_array)
        
        return img_array
    
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        # Return zeros as fallback
        return np.zeros((1, target_size[0], target_size[1], 3))

def preprocess_image_for_segmentation(image_path, target_size=(256, 256)):
    """
    Preprocess image for segmentation model (U-Net input format)
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
    
    Returns:
        np.ndarray: Preprocessed image array
    """
    try:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Expand dimensions to match batch format
        img = np.expand_dims(img, axis=0)
        
        return img
    
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        # Return zeros as fallback
        return np.zeros((1, target_size[0], target_size[1], 3))

def resize_image_maintain_aspect_ratio(image, target_size, background_color=(0, 0, 0)):
    """
    Resize image while maintaining aspect ratio by adding padding
    
    Args:
        image (np.ndarray): Input image
        target_size (tuple): Target (width, height)
        background_color (tuple): Color for padding
    
    Returns:
        np.ndarray: Resized image with padding
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scaling factor
    scale = min(target_width / width, target_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Create canvas with target size
    canvas = np.full((target_height, target_width, 3), background_color, dtype=image.dtype)
    
    # Calculate position to center the resized image
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return canvas

def augment_image(image, augmentation_params=None):
    """
    Apply data augmentation to image
    
    Args:
        image (np.ndarray): Input image
        augmentation_params (dict): Parameters for augmentation
    
    Returns:
        np.ndarray: Augmented image
    """
    if augmentation_params is None:
        augmentation_params = {
            'rotation_range': 10,
            'brightness_range': (0.8, 1.2),
            'zoom_range': 0.1,
            'horizontal_flip': True
        }
    
    # Make a copy to avoid modifying original
    aug_image = image.copy()
    
    # Random rotation
    if 'rotation_range' in augmentation_params:
        angle = np.random.uniform(-augmentation_params['rotation_range'], 
                                 augmentation_params['rotation_range'])
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_image = cv2.warpAffine(aug_image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    # Random brightness
    if 'brightness_range' in augmentation_params:
        brightness_factor = np.random.uniform(*augmentation_params['brightness_range'])
        aug_image = np.clip(aug_image * brightness_factor, 0, 255).astype(np.uint8)
    
    # Random horizontal flip
    if augmentation_params.get('horizontal_flip', False) and np.random.random() > 0.5:
        aug_image = cv2.flip(aug_image, 1)
    
    return aug_image

def preprocess_caption_text(caption, tokenizer, max_length):
    """
    Preprocess caption text for model input
    
    Args:
        caption (str): Raw caption text
        tokenizer: Fitted tokenizer
        max_length (int): Maximum sequence length
    
    Returns:
        np.ndarray: Tokenized and padded sequence
    """
    # Clean and format caption
    caption = caption.lower().strip()
    caption = f'<start> {caption} <end>'
    
    # Tokenize
    sequence = tokenizer.texts_to_sequences([caption])[0]
    
    # Pad sequence
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
    
    return padded_sequence

def postprocess_segmentation_mask(mask, original_size):
    """
    Postprocess segmentation mask to original image size
    
    Args:
        mask (np.ndarray): Predicted segmentation mask
        original_size (tuple): Original image size (width, height)
    
    Returns:
        np.ndarray: Resized segmentation mask
    """
    # Convert to class indices if it's probability map
    if len(mask.shape) == 3 and mask.shape[-1] > 1:
        mask = np.argmax(mask, axis=-1)
    
    # Resize to original size
    resized_mask = cv2.resize(mask.astype(np.uint8), original_size, 
                             interpolation=cv2.INTER_NEAREST)
    
    return resized_mask

def normalize_image(image, method='standard'):
    """
    Normalize image using different methods
    
    Args:
        image (np.ndarray): Input image
        method (str): Normalization method ('standard', 'minmax', 'imagenet')
    
    Returns:
        np.ndarray: Normalized image
    """
    image = image.astype(np.float32)
    
    if method == 'standard':
        # Normalize to [0, 1]
        return image / 255.0
    
    elif method == 'minmax':
        # Min-max normalization
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min > 0:
            return (image - img_min) / (img_max - img_min)
        return image
    
    elif method == 'imagenet':
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # First normalize to [0, 1]
        image = image / 255.0
        
        # Apply ImageNet normalization
        image = (image - mean) / std
        
        return image
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_image_patches(image, patch_size, stride=None):
    """
    Create overlapping patches from image for processing large images
    
    Args:
        image (np.ndarray): Input image
        patch_size (tuple): Size of patches (height, width)
        stride (tuple): Stride for patch extraction
    
    Returns:
        list: List of image patches and their coordinates
    """
    if stride is None:
        stride = (patch_size[0] // 2, patch_size[1] // 2)
    
    patches = []
    coordinates = []
    
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size
    stride_h, stride_w = stride
    
    for y in range(0, height - patch_height + 1, stride_h):
        for x in range(0, width - patch_width + 1, stride_w):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)
            coordinates.append((x, y))
    
    return patches, coordinates

def merge_patches(patches, coordinates, output_size, patch_size, method='average'):
    """
    Merge patches back into full image
    
    Args:
        patches (list): List of patches
        coordinates (list): Coordinates of patches
        output_size (tuple): Size of output image
        patch_size (tuple): Size of patches
        method (str): Merging method ('average', 'max')
    
    Returns:
        np.ndarray: Merged image
    """
    height, width = output_size
    patch_height, patch_width = patch_size
    
    # Initialize output arrays
    if method == 'average':
        output = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        
        for patch, (x, y) in zip(patches, coordinates):
            output[y:y + patch_height, x:x + patch_width] += patch
            counts[y:y + patch_height, x:x + patch_width] += 1
        
        # Avoid division by zero
        counts[counts == 0] = 1
        output = output / counts
        
    elif method == 'max':
        output = np.zeros((height, width), dtype=np.float32)
        
        for patch, (x, y) in zip(patches, coordinates):
            current = output[y:y + patch_height, x:x + patch_width]
            output[y:y + patch_height, x:x + patch_width] = np.maximum(current, patch)
    
    return output

class ImagePreprocessor:
    """
    Comprehensive image preprocessor class
    """
    
    def __init__(self, target_size=(256, 256), normalization='standard'):
        self.target_size = target_size
        self.normalization = normalization
        
    def __call__(self, image_path):
        """
        Process image from file path
        
        Args:
            image_path (str): Path to image
        
        Returns:
            np.ndarray: Processed image
        """
        return self.preprocess_from_path(image_path)
    
    def preprocess_from_path(self, image_path):
        """Load and preprocess image from file path"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.preprocess_array(img)
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return np.zeros((1, self.target_size[0], self.target_size[1], 3))
    
    def preprocess_array(self, image_array):
        """Preprocess image array"""
        # Resize
        resized = cv2.resize(image_array, self.target_size)
        
        # Normalize
        normalized = normalize_image(resized, method=self.normalization)
        
        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)
        
        return batch
    
    def preprocess_pil(self, pil_image):
        """Preprocess PIL image"""
        # Convert to numpy array
        img_array = np.array(pil_image)
        
        # Ensure RGB format
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        return self.preprocess_array(img_array)

# Utility functions for text preprocessing
def clean_caption(caption):
    """
    Clean caption text
    
    Args:
        caption (str): Raw caption
    
    Returns:
        str: Cleaned caption
    """
    import re
    
    # Convert to lowercase
    caption = caption.lower()
    
    # Remove special characters except spaces and periods
    caption = re.sub(r'[^a-z0-9\s\.]', '', caption)
    
    # Remove extra spaces
    caption = re.sub(r'\s+', ' ', caption)
    
    # Strip whitespace
    caption = caption.strip()
    
    return caption

def add_caption_tokens(caption, start_token='<start>', end_token='<end>'):
    """
    Add start and end tokens to caption
    
    Args:
        caption (str): Caption text
        start_token (str): Start token
        end_token (str): End token
    
    Returns:
        str: Caption with tokens
    """
    return f"{start_token} {caption} {end_token}"

def create_caption_sequences(captions, tokenizer, max_length):
    """
    Convert captions to padded sequences
    
    Args:
        captions (list): List of captions
        tokenizer: Fitted tokenizer
        max_length (int): Maximum sequence length
    
    Returns:
        np.ndarray: Padded sequences
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Tokenize captions
    sequences = tokenizer.texts_to_sequences(captions)
    
    # Pad sequences
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    return padded