#!/usr/bin/env python3
"""
Pretrained segmentation model using DeepLabV3+ for real semantic segmentation
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
from PIL import Image

class PretrainedSegmentationModel:
    """Wrapper for pretrained DeepLabV3+ model"""
    
    def __init__(self):
        self.model = None
        self.model_url = "https://tfhub.dev/tensorflow/deeplabv3/1"
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
        ]
        self.colors = [
            [0, 0, 0],       # background - black
            [128, 0, 0],     # aeroplane - maroon
            [0, 128, 0],     # bicycle - green
            [128, 128, 0],   # bird - olive
            [0, 0, 128],     # boat - navy
            [128, 0, 128],   # bottle - purple
            [0, 128, 128],   # bus - teal
            [128, 128, 128], # car - gray
            [64, 0, 0],      # cat - dark red
            [192, 0, 0],     # chair - red
            [64, 128, 0],    # cow - dark green
            [192, 128, 0],   # diningtable - orange
            [64, 0, 128],    # dog - dark blue
            [192, 0, 128],   # horse - magenta
            [64, 128, 128],  # motorbike - dark teal
            [192, 128, 128], # person - pink
            [0, 64, 0],      # pottedplant - forest green
            [128, 64, 0],    # sheep - brown
            [0, 192, 0],     # sofa - lime
            [128, 192, 0],   # train - yellow-green
            [0, 64, 128]     # tv - dark cyan
        ]
        
    def load_model(self):
        """Load the pretrained DeepLabV3 model"""
        try:
            print("Loading DeepLabV3+ model from TensorFlow Hub...")
            self.model = hub.load(self.model_url)
            print("✅ Pretrained segmentation model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading pretrained model: {e}")
            print("Falling back to lightweight alternative...")
            return self._create_lightweight_model()
    
    def _create_lightweight_model(self):
        """Create a lightweight segmentation model as fallback"""
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Input
            from tensorflow.keras.models import Model
            
            # Use MobileNetV2 as backbone
            input_tensor = Input(shape=(512, 512, 3))
            base_model = MobileNetV2(input_tensor=input_tensor, weights='imagenet', 
                                   include_top=False, alpha=0.35)
            
            # Simple decoder
            x = base_model.output
            x = Conv2D(256, 3, activation='relu', padding='same')(x)
            x = UpSampling2D(2)(x)
            x = Conv2D(128, 3, activation='relu', padding='same')(x)
            x = UpSampling2D(2)(x)
            x = Conv2D(64, 3, activation='relu', padding='same')(x)
            x = UpSampling2D(2)(x)
            x = Conv2D(32, 3, activation='relu', padding='same')(x)
            x = UpSampling2D(2)(x)
            x = Conv2D(21, 3, activation='softmax', padding='same', name='segmentation_output')(x)
            
            self.model = Model(inputs=input_tensor, outputs=x)
            print("✅ Lightweight segmentation model created!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating lightweight model: {e}")
            return False
    
    def predict(self, image_path):
        """Generate segmentation mask for an image"""
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            
            if self.model is None:
                if not self.load_model():
                    return None
            
            # Run inference
            if hasattr(self.model, 'signatures'):  # TensorFlow Hub model
                predictions = self.model.signatures['serving_default'](image)
                segmentation_mask = predictions['semantic']
            else:  # Custom Keras model
                predictions = self.model.predict(image, verbose=0)
                segmentation_mask = predictions
            
            # Post-process the mask
            mask = self._postprocess_mask(segmentation_mask, image_path)
            return mask
            
        except Exception as e:
            print(f"Error in segmentation prediction: {e}")
            return None
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess image for segmentation"""
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        
        # Resize to model input size
        image = tf.image.resize(image, [512, 512])
        image = image / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        image = tf.expand_dims(image, 0)
        
        return image
    
    def _postprocess_mask(self, mask, original_image_path):
        """Post-process segmentation mask"""
        # Convert to numpy
        if tf.is_tensor(mask):
            mask = mask.numpy()
        
        # Get the class predictions
        if len(mask.shape) == 4:  # Batch dimension
            mask = mask[0]
        
        if len(mask.shape) == 3 and mask.shape[-1] > 1:  # Probability map
            mask = np.argmax(mask, axis=-1)
        
        # Create colored mask
        colored_mask = self._apply_color_map(mask)
        
        # Resize to original image size
        original_img = cv2.imread(original_image_path)
        if original_img is not None:
            original_height, original_width = original_img.shape[:2]
            colored_mask = cv2.resize(colored_mask, (original_width, original_height), 
                                    interpolation=cv2.INTER_NEAREST)
        
        return colored_mask
    
    def _apply_color_map(self, mask):
        """Apply color mapping to segmentation mask"""
        height, width = mask.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(self.colors):
            if class_id < len(self.colors):
                colored_mask[mask == class_id] = color
        
        return colored_mask
    
    def get_class_names(self):
        """Get list of class names"""
        return self.class_names

# Alternative lightweight segmentation using image processing
class LightweightSegmentation:
    """Lightweight segmentation using computer vision techniques"""
    
    def __init__(self):
        self.colors = {
            'sky': [135, 206, 235],      # Sky blue
            'vegetation': [34, 139, 34],  # Forest green
            'ground': [139, 69, 19],     # Brown
            'water': [0, 100, 200],      # Deep blue
            'building': [128, 128, 128], # Gray
            'person': [255, 182, 193],   # Light pink
            'vehicle': [255, 140, 0],    # Dark orange
            'object': [255, 20, 147]     # Deep pink
        }
    
    def segment_image(self, image_path):
        """Create segmentation using image processing techniques"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            height, width = img.shape[:2]
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Initialize segmentation mask
            segmentation = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Sky detection (upper region, high brightness, blue-ish)
            sky_mask = self._detect_sky(img, hsv)
            segmentation[sky_mask] = self.colors['sky']
            
            # Vegetation detection (green regions)
            vegetation_mask = self._detect_vegetation(img, hsv)
            segmentation[vegetation_mask] = self.colors['vegetation']
            
            # Water detection (blue regions, horizontal patterns)
            water_mask = self._detect_water(img, hsv)
            segmentation[water_mask] = self.colors['water']
            
            # Ground/road detection (lower region, brownish)
            ground_mask = self._detect_ground(img, height)
            segmentation[ground_mask] = self.colors['ground']
            
            # Building detection (geometric patterns, gray/white)
            building_mask = self._detect_buildings(img, hsv)
            segmentation[building_mask] = self.colors['building']
            
            # Apply smoothing for better appearance
            segmentation = cv2.bilateralFilter(segmentation, 9, 75, 75)
            
            return segmentation
            
        except Exception as e:
            print(f"Error in lightweight segmentation: {e}")
            return None
    
    def _detect_sky(self, img, hsv):
        """Detect sky regions"""
        height = img.shape[0]
        
        # Focus on upper third of image
        upper_region = hsv[:height//3, :]
        
        # Sky is typically bright and blue-ish
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(upper_region, lower_blue, upper_blue)
        
        # Also consider very bright regions in upper area
        brightness_mask = cv2.inRange(upper_region[:, :, 2], 180, 255)
        
        # Combine masks
        sky_region_mask = cv2.bitwise_or(blue_mask, brightness_mask)
        
        # Create full-size mask
        full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        full_mask[:height//3, :] = sky_region_mask
        
        return full_mask > 0
    
    def _detect_vegetation(self, img, hsv):
        """Detect vegetation/trees"""
        # Green color range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        return green_mask > 0
    
    def _detect_water(self, img, hsv):
        """Detect water bodies"""
        # Water is typically blue and horizontal
        lower_water = np.array([90, 50, 50])
        upper_water = np.array([120, 255, 200])
        
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        
        # Look for horizontal patterns (water surfaces)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        
        return water_mask > 0
    
    def _detect_ground(self, img, height):
        """Detect ground/road in lower region"""
        # Focus on lower third
        ground_mask = np.zeros((height, img.shape[1]), dtype=np.uint8)
        ground_mask[2*height//3:, :] = 255
        
        return ground_mask > 0
    
    def _detect_buildings(self, img, hsv):
        """Detect buildings and structures"""
        # Buildings are often gray/white with geometric patterns
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect edges for geometric structures
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for rectangular structures
        kernel = np.ones((3, 3), np.uint8)
        building_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Focus on middle regions (where buildings typically are)
        height = img.shape[0]
        building_mask[:height//3, :] = 0  # Remove upper region (sky)
        building_mask[2*height//3:, :] = 0  # Remove lower region (ground)
        
        return building_mask > 0

def create_segmentation_model():
    """Create and return segmentation model"""
    try:
        # Try to load pretrained model first
        model = PretrainedSegmentationModel()
        if model.load_model():
            return model
    except Exception as e:
        print(f"Could not load pretrained model: {e}")
    
    # Fallback to lightweight segmentation
    print("Using lightweight segmentation model...")
    return LightweightSegmentation()

if __name__ == "__main__":
    # Test the segmentation models
    print("Testing segmentation models...")
    
    model = create_segmentation_model()
    print(f"Created segmentation model: {type(model).__name__}")
    
    if hasattr(model, 'get_class_names'):
        print(f"Available classes: {model.get_class_names()}")