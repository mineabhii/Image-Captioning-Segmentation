import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D, Concatenate, 
    Input, BatchNormalization, Activation, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def conv_block(inputs, num_filters, kernel_size=3, dropout_rate=0.2):
    """
    Convolutional block with batch normalization and dropout
    
    Args:
        inputs: Input tensor
        num_filters: Number of filters for conv layers
        kernel_size: Size of convolution kernel
        dropout_rate: Dropout rate
    
    Returns:
        Output tensor after conv block
    """
    x = Conv2D(num_filters, kernel_size, padding='same', 
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, kernel_size, padding='same',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    return x

def encoder_block(inputs, num_filters, dropout_rate=0.2):
    """
    Encoder block for U-Net
    
    Args:
        inputs: Input tensor
        num_filters: Number of filters
        dropout_rate: Dropout rate
    
    Returns:
        Tuple of (conv_output, pooled_output)
    """
    conv = conv_block(inputs, num_filters, dropout_rate=dropout_rate)
    pool = MaxPooling2D((2, 2))(conv)
    
    return conv, pool

def decoder_block(inputs, skip_connection, num_filters, dropout_rate=0.2):
    """
    Decoder block for U-Net
    
    Args:
        inputs: Input tensor from previous layer
        skip_connection: Skip connection from encoder
        num_filters: Number of filters
        dropout_rate: Dropout rate
    
    Returns:
        Output tensor after decoder block
    """
    upsample = UpSampling2D((2, 2))(inputs)
    concat = Concatenate()([upsample, skip_connection])
    conv = conv_block(concat, num_filters, dropout_rate=dropout_rate)
    
    return conv

def create_segmentation_model(input_shape=(256, 256, 3), num_classes=21, dropout_rate=0.2):
    """
    Create U-Net model for semantic segmentation
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of segmentation classes
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled U-Net model
    """
    
    inputs = Input(input_shape, name='image_input')
    
    # Encoder path
    conv1, pool1 = encoder_block(inputs, 64, dropout_rate)
    conv2, pool2 = encoder_block(pool1, 128, dropout_rate)
    conv3, pool3 = encoder_block(pool2, 256, dropout_rate)
    conv4, pool4 = encoder_block(pool3, 512, dropout_rate)
    
    # Bottleneck
    bottleneck = conv_block(pool4, 1024, dropout_rate=dropout_rate * 2)
    
    # Decoder path
    dec4 = decoder_block(bottleneck, conv4, 512, dropout_rate)
    dec3 = decoder_block(dec4, conv3, 256, dropout_rate)
    dec2 = decoder_block(dec3, conv2, 128, dropout_rate)
    dec1 = decoder_block(dec2, conv1, 64, dropout_rate)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax', name='segmentation_output')(dec1)
    
    # Create model
    model = Model(inputs, outputs, name='unet_segmentation')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_crossentropy']
    )
    
    return model

def create_deep_unet(input_shape=(256, 256, 3), num_classes=21, dropout_rate=0.2):
    """
    Create a deeper U-Net model with more skip connections
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes for segmentation
        dropout_rate: Dropout rate
    
    Returns:
        Compiled deep U-Net model
    """
    
    inputs = Input(input_shape, name='image_input')
    
    # Encoder
    conv1, pool1 = encoder_block(inputs, 32, dropout_rate)
    conv2, pool2 = encoder_block(pool1, 64, dropout_rate)
    conv3, pool3 = encoder_block(pool2, 128, dropout_rate)
    conv4, pool4 = encoder_block(pool3, 256, dropout_rate)
    conv5, pool5 = encoder_block(pool4, 512, dropout_rate)
    
    # Bottleneck
    bottleneck = conv_block(pool5, 1024, dropout_rate=dropout_rate * 2)
    
    # Decoder
    dec5 = decoder_block(bottleneck, conv5, 512, dropout_rate)
    dec4 = decoder_block(dec5, conv4, 256, dropout_rate)
    dec3 = decoder_block(dec4, conv3, 128, dropout_rate)
    dec2 = decoder_block(dec3, conv2, 64, dropout_rate)
    dec1 = decoder_block(dec2, conv1, 32, dropout_rate)
    
    # Output
    outputs = Conv2D(num_classes, 1, activation='softmax', name='segmentation_output')(dec1)
    
    model = Model(inputs, outputs, name='deep_unet_segmentation')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_crossentropy']
    )
    
    return model

def create_attention_unet(input_shape=(256, 256, 3), num_classes=21, dropout_rate=0.2):
    """
    U-Net with attention gates for better feature selection
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
        dropout_rate: Dropout rate
    
    Returns:
        U-Net model with attention mechanisms
    """
    
    def attention_gate(F_g, F_l, F_int):
        """
        Attention gate implementation
        F_g: gating signal from coarser scale
        F_l: feature map from finer scale
        F_int: intermediate channel number
        """
        W_g = Conv2D(F_int, 1, padding='same')(F_g)
        W_g = BatchNormalization()(W_g)
        
        W_x = Conv2D(F_int, 1, padding='same')(F_l)
        W_x = BatchNormalization()(W_x)
        
        psi = Activation('relu')(W_g + W_x)
        psi = Conv2D(1, 1, padding='same')(psi)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)
        
        return tf.keras.layers.multiply([F_l, psi])
    
    inputs = Input(input_shape)
    
    # Encoder
    conv1, pool1 = encoder_block(inputs, 64, dropout_rate)
    conv2, pool2 = encoder_block(pool1, 128, dropout_rate)
    conv3, pool3 = encoder_block(pool2, 256, dropout_rate)
    conv4, pool4 = encoder_block(pool3, 512, dropout_rate)
    
    # Bottleneck
    bottleneck = conv_block(pool4, 1024, dropout_rate=dropout_rate * 2)
    
    # Decoder with attention
    up4 = UpSampling2D((2, 2))(bottleneck)
    att4 = attention_gate(up4, conv4, 256)
    concat4 = Concatenate()([up4, att4])
    dec4 = conv_block(concat4, 512, dropout_rate=dropout_rate)
    
    up3 = UpSampling2D((2, 2))(dec4)
    att3 = attention_gate(up3, conv3, 128)
    concat3 = Concatenate()([up3, att3])
    dec3 = conv_block(concat3, 256, dropout_rate=dropout_rate)
    
    up2 = UpSampling2D((2, 2))(dec3)
    att2 = attention_gate(up2, conv2, 64)
    concat2 = Concatenate()([up2, att2])
    dec2 = conv_block(concat2, 128, dropout_rate=dropout_rate)
    
    up1 = UpSampling2D((2, 2))(dec2)
    att1 = attention_gate(up1, conv1, 32)
    concat1 = Concatenate()([up1, att1])
    dec1 = conv_block(concat1, 64, dropout_rate=dropout_rate)
    
    # Output
    outputs = Conv2D(num_classes, 1, activation='softmax')(dec1)
    
    model = Model(inputs, outputs, name='attention_unet')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_crossentropy']
    )
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for segmentation evaluation
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice loss function for training
    """
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """
    Combined loss: Cross-entropy + Dice loss
    """
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    dice_loss_val = dice_loss(y_true, y_pred)
    return ce_loss + dice_loss_val

class SegmentationCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for monitoring segmentation during training
    """
    
    def __init__(self, validation_data, validation_freq=5):
        super().__init__()
        self.validation_data = validation_data
        self.validation_freq = validation_freq
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.validation_freq == 0:
            print(f"\n--- Validation Results at Epoch {epoch + 1} ---")
            
            # Get a batch of validation data
            val_images, val_masks = self.validation_data[:3]  # Take first 3 samples
            
            # Predict
            predictions = self.model.predict(val_images, verbose=0)
            
            # Calculate metrics
            for i in range(len(val_images)):
                pred_mask = tf.argmax(predictions[i], axis=-1)
                true_mask = val_masks[i]
                
                # Calculate IoU or other metrics here
                print(f"Sample {i+1}: Prediction shape {pred_mask.shape}")

# Color map for visualization
CITYSCAPES_COLORS = [
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
    [0, 0, 0],        # void/background
    [255, 255, 255],  # unknown
]

def colorize_mask(mask):
    """
    Convert segmentation mask to colored image
    
    Args:
        mask: 2D numpy array with class indices
    
    Returns:
        3D colored mask
    """
    colored_mask = tf.zeros((mask.shape[0], mask.shape[1], 3), dtype=tf.uint8)
    
    for class_id, color in enumerate(CITYSCAPES_COLORS):
        colored_mask = tf.where(
            tf.expand_dims(mask == class_id, axis=-1),
            color,
            colored_mask
        )
    
    return colored_mask

if __name__ == "__main__":
    # Test model creation
    print("Creating standard U-Net model...")
    model = create_segmentation_model()
    print(f"Standard U-Net created: {model.input_shape} -> {model.output_shape}")
    
    print("\nCreating deep U-Net model...")
    deep_model = create_deep_unet()
    print(f"Deep U-Net created: {deep_model.input_shape} -> {deep_model.output_shape}")
    
    print("\nCreating attention U-Net model...")
    att_model = create_attention_unet()
    print(f"Attention U-Net created: {att_model.input_shape} -> {att_model.output_shape}")
    
    print("\nModel summary (Standard U-Net):")
    model.summary()