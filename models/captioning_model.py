import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout, Add, Input, 
    GlobalAveragePooling2D, RepeatVector, Concatenate,
    TimeDistributed, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_feature_extractor():
    """
    Create feature extractor using pre-trained InceptionV3
    Returns a model that extracts features from images
    """
    # Load pre-trained InceptionV3 model
    inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    
    # Freeze the layers
    for layer in inception.layers:
        layer.trainable = False
    
    # Add custom layers for feature extraction
    x = inception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Create the feature extractor model
    feature_extractor = Model(inputs=inception.input, outputs=x, name='feature_extractor')
    
    return feature_extractor

def create_captioning_model(vocab_size, max_length, embedding_dim=512, lstm_units=512):
    """
    Create the complete image captioning model
    
    Args:
        vocab_size (int): Size of vocabulary
        max_length (int): Maximum length of captions
        embedding_dim (int): Dimension of word embeddings
        lstm_units (int): Number of LSTM units
    
    Returns:
        Compiled Keras model for image captioning
    """
    
    # Image feature input
    image_features = Input(shape=(512,), name='image_features')
    
    # Text sequence input
    text_input = Input(shape=(max_length,), name='text_input')
    
    # Image feature processing
    image_dense = Dense(embedding_dim, activation='relu')(image_features)
    image_dropout = Dropout(0.5)(image_dense)
    
    # Text processing
    text_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
    text_dropout = Dropout(0.5)(text_embedding)
    
    # LSTM decoder
    lstm_out = LSTM(lstm_units, return_sequences=True, return_state=True)(text_dropout)
    lstm_output, lstm_h, lstm_c = lstm_out
    
    # Attention mechanism (simplified)
    # Repeat image features to match sequence length
    image_repeated = RepeatVector(max_length)(image_dropout)
    
    # Concatenate image features with LSTM output
    concat_features = Concatenate()([lstm_output, image_repeated])
    
    # Dense layers for output
    dense1 = TimeDistributed(Dense(lstm_units, activation='relu'))(concat_features)
    dense1_dropout = Dropout(0.5)(dense1)
    
    dense2 = TimeDistributed(Dense(vocab_size, activation='softmax'))(dense1_dropout)
    
    # Create the model
    model = Model(inputs=[image_features, text_input], outputs=dense2, name='captioning_model')
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_encoder_decoder_model(vocab_size, max_length, embedding_dim=512, lstm_units=512):
    """
    Alternative architecture: Encoder-Decoder with attention
    
    Args:
        vocab_size (int): Size of vocabulary
        max_length (int): Maximum length of captions
        embedding_dim (int): Dimension of word embeddings
        lstm_units (int): Number of LSTM units
    
    Returns:
        Compiled Keras model for image captioning
    """
    
    # Encoder (Image processing)
    encoder_input = Input(shape=(299, 299, 3), name='encoder_input')
    
    # Use InceptionV3 as encoder
    inception = InceptionV3(weights='imagenet', include_top=False)
    for layer in inception.layers:
        layer.trainable = False
    
    encoder_features = inception(encoder_input)
    encoder_features = GlobalAveragePooling2D()(encoder_features)
    encoder_output = Dense(lstm_units, activation='relu')(encoder_features)
    
    # Decoder (Text generation)
    decoder_input = Input(shape=(None,), name='decoder_input')
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_input)
    
    # LSTM decoder with initial state from encoder
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    
    # Use encoder output as initial state
    initial_h = Dense(lstm_units, activation='tanh')(encoder_output)
    initial_c = Dense(lstm_units, activation='tanh')(encoder_output)
    
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding, 
        initial_state=[initial_h, initial_c]
    )
    
    # Attention mechanism
    attention = tf.keras.layers.Attention()
    
    # Repeat encoder output to match decoder sequence length
    encoder_repeated = RepeatVector(tf.shape(decoder_outputs)[1])(encoder_output)
    
    # Apply attention
    context_vector = attention([decoder_outputs, encoder_repeated])
    
    # Concatenate context with decoder output
    decoder_concat = Concatenate()([decoder_outputs, context_vector])
    
    # Final dense layer
    decoder_dense = Dense(vocab_size, activation='softmax')(decoder_concat)
    
    # Create model
    model = Model([encoder_input, decoder_input], decoder_dense, name='encoder_decoder_captioning')
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_inference_model(trained_model, vocab_size, max_length):
    """
    Create model for inference (generating captions)
    
    Args:
        trained_model: Trained captioning model
        vocab_size (int): Size of vocabulary
        max_length (int): Maximum length of captions
    
    Returns:
        Model optimized for inference
    """
    
    # Extract feature extractor from trained model
    image_input = Input(shape=(299, 299, 3))
    
    # Feature extraction layers
    inception = InceptionV3(weights='imagenet', include_top=False)
    features = inception(image_input)
    features = GlobalAveragePooling2D()(features)
    image_features = Dense(512, activation='relu')(features)
    
    # Text input for partial sequence
    text_input = Input(shape=(None,))
    text_embedding = Embedding(vocab_size, 512)(text_input)
    
    # LSTM for sequence generation
    lstm_out, lstm_h, lstm_c = LSTM(512, return_state=True)(text_embedding)
    
    # Dense layer for next word prediction
    output = Dense(vocab_size, activation='softmax')(lstm_out)
    
    inference_model = Model([image_input, text_input], [output, lstm_h, lstm_c])
    
    return inference_model

class CaptioningCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for monitoring caption generation during training
    """
    
    def __init__(self, feature_extractor, tokenizer, sample_images, validation_freq=5):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_images = sample_images
        self.validation_freq = validation_freq
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.validation_freq == 0:
            print(f"\n--- Sample Captions at Epoch {epoch + 1} ---")
            
            for i, img_path in enumerate(self.sample_images[:3]):  # Show 3 samples
                try:
                    # Load and preprocess image
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, target_size=(299, 299)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
                    
                    # Extract features
                    features = self.feature_extractor.predict(img_array, verbose=0)
                    
                    # Generate caption (simplified)
                    caption = self._generate_simple_caption(features)
                    print(f"Image {i+1}: {caption}")
                    
                except Exception as e:
                    print(f"Error generating caption for image {i+1}: {e}")
    
    def _generate_simple_caption(self, image_features):
        """
        Simplified caption generation for monitoring
        """
        # This is a placeholder - in practice, you'd use beam search
        # or other advanced decoding techniques
        return "Sample caption generated during training"

if __name__ == "__main__":
    # Test model creation
    vocab_size = 8000
    max_length = 34
    
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    print(f"Feature extractor created: {feature_extractor.output_shape}")
    
    print("Creating captioning model...")
    model = create_captioning_model(vocab_size, max_length)
    print("Model created successfully!")
    print(model.summary())
    
    print("Creating encoder-decoder model...")
    enc_dec_model = create_encoder_decoder_model(vocab_size, max_length)
    print("Encoder-decoder model created successfully!")