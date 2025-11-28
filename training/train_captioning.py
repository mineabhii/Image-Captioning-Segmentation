import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.captioning_model import create_feature_extractor, create_captioning_model, CaptioningCallback
from utils.preprocessing import preprocess_image_for_captioning

class COCOCaptionDataLoader:
    """
    Data loader for MS COCO caption dataset
    """
    
    def __init__(self, data_dir, annotation_file, max_vocab_size=8000, max_length=34):
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.tokenizer = None
        self.captions = []
        self.image_paths = []
        
    def load_annotations(self):
        """Load and process COCO annotations"""
        print("Loading COCO annotations...")
        
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
        
        # Create image id to filename mapping
        image_id_to_path = {}
        for img in data['images']:
            image_id_to_path[img['id']] = os.path.join(
                self.data_dir, img['file_name']
            )
        
        # Extract captions
        captions_dict = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            
            if image_id not in captions_dict:
                captions_dict[image_id] = []
            captions_dict[image_id].append(caption)
        
        # Prepare data lists
        for image_id, caption_list in captions_dict.items():
            if image_id in image_id_to_path:
                image_path = image_id_to_path[image_id]
                
                # Only use images that exist
                if os.path.exists(image_path):
                    for caption in caption_list:
                        self.captions.append(self.preprocess_caption(caption))
                        self.image_paths.append(image_path)
        
        print(f"Loaded {len(self.captions)} caption-image pairs")
        return self.captions, self.image_paths
    
    def preprocess_caption(self, caption):
        """Preprocess caption text"""
        # Convert to lowercase
        caption = caption.lower()
        
        # Remove special characters and extra spaces
        import re
        caption = re.sub(r'[^a-zA-Z0-9\s]', '', caption)
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Add start and end tokens
        caption = f'<start> {caption} <end>'
        
        return caption
    
    def create_tokenizer(self):
        """Create and fit tokenizer on captions"""
        print("Creating tokenizer...")
        
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token='<unk>',
            filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
        )
        
        self.tokenizer.fit_on_texts(self.captions)
        
        # Add special tokens if not present
        word_index = self.tokenizer.word_index
        if '<start>' not in word_index:
            word_index['<start>'] = 1
        if '<end>' not in word_index:
            word_index['<end>'] = 2
        if '<unk>' not in word_index:
            word_index['<unk>'] = 3
            
        self.tokenizer.word_index = word_index
        
        vocab_size = min(len(word_index) + 1, self.max_vocab_size)
        print(f"Vocabulary size: {vocab_size}")
        
        return self.tokenizer
    
    def prepare_sequences(self):
        """Convert captions to sequences and pad them"""
        print("Preparing sequences...")
        
        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(self.captions)
        
        # Find max length
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        print(f"Max caption length: {max_len}")
        
        # Use specified max_length or computed max_len
        self.max_length = min(self.max_length, max_len)
        
        # Pad sequences
        sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        return sequences

class CaptionDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for training caption model
    """
    
    def __init__(self, image_paths, captions, feature_extractor, batch_size=32, shuffle=True):
        self.image_paths = image_paths
        self.captions = captions
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(image_paths))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[i] for i in batch_indices]
        batch_captions = self.captions[batch_indices]
        
        return self._generate_batch(batch_image_paths, batch_captions)
    
    def _generate_batch(self, image_paths, captions):
        """Generate batch of features and targets"""
        batch_size = len(image_paths)
        max_length = captions.shape[1]
        
        # Extract image features
        images = []
        for img_path in image_paths:
            try:
                img = load_img(img_path, target_size=(299, 299))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Use zeros as fallback
                images.append(np.zeros((299, 299, 3)))
        
        images = np.array(images)
        image_features = self.feature_extractor.predict(images, verbose=0)
        
        # Prepare input and target sequences
        input_sequences = captions[:, :-1]  # All tokens except the last
        target_sequences = captions[:, 1:]   # All tokens except the first
        
        return [image_features, input_sequences], target_sequences
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def train_captioning_model(
    data_dir='data/train2017',
    annotation_file='data/annotations/captions_train2017.json',
    val_data_dir='data/val2017',
    val_annotation_file='data/annotations/captions_val2017.json',
    batch_size=32,
    epochs=50,
    max_vocab_size=8000,
    max_length=34,
    use_subset=True,
    subset_size=10000
):
    """
    Train the image captioning model
    
    Args:
        data_dir: Directory containing training images
        annotation_file: Path to training annotations
        val_data_dir: Directory containing validation images
        val_annotation_file: Path to validation annotations
        batch_size: Batch size for training
        epochs: Number of epochs
        max_vocab_size: Maximum vocabulary size
        max_length: Maximum caption length
        use_subset: Whether to use subset of data for quick training
        subset_size: Size of subset if use_subset is True
    """
    
    print("Starting Image Captioning Model Training")
    print("=" * 50)
    
    # Check if data directories exist
    if not os.path.exists(data_dir):
        print(f"Error: Training data directory not found: {data_dir}")
        print("Please download MS COCO 2017 dataset and place it in the data/ directory")
        return
    
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        print("Please download MS COCO 2017 annotations")
        return
    
    # Load training data
    train_loader = COCOCaptionDataLoader(data_dir, annotation_file, max_vocab_size, max_length)
    train_captions, train_image_paths = train_loader.load_annotations()
    
    # Use subset for faster training/testing
    if use_subset:
        print(f"Using subset of {subset_size} samples for training")
        indices = np.random.choice(len(train_captions), subset_size, replace=False)
        train_captions = [train_captions[i] for i in indices]
        train_image_paths = [train_image_paths[i] for i in indices]
    
    # Create tokenizer
    tokenizer = train_loader.create_tokenizer()
    
    # Prepare sequences
    train_sequences = train_loader.prepare_sequences()
    
    # Split data
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        train_image_paths, train_sequences, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train_paths)}")
    print(f"Validation samples: {len(X_val_paths)}")
    
    # Create feature extractor
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    
    # Create captioning model
    print("Creating captioning model...")
    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab_size)
    model = create_captioning_model(vocab_size, max_length)
    
    print(model.summary())
    
    # Create data generators
    train_generator = CaptionDataGenerator(
        X_train_paths, y_train, feature_extractor, batch_size, shuffle=True
    )
    
    val_generator = CaptionDataGenerator(
        X_val_paths, y_val, feature_extractor, batch_size, shuffle=False
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'saved_models/captioning_model.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        CaptioningCallback(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            sample_images=X_val_paths[:5],
            validation_freq=5
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save tokenizer
    print("Saving tokenizer...")
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save model architecture
    model_json = model.to_json()
    with open('saved_models/captioning_model.json', 'w') as f:
        f.write(model_json)
    
    # Plot training history
    if history:
        plot_training_history(history)
    
    print("Training completed!")
    print("Models saved in 'saved_models/' directory")
    
    return model, tokenizer, history

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('saved_models/training_history.png')
    plt.show()

def create_sample_data():
    """
    Create sample data for testing when COCO dataset is not available
    """
    print("Creating sample data for testing...")
    
    # Create more comprehensive sample captions
    sample_captions = [
        "a person sitting on a bench",
        "a cat sleeping on a chair", 
        "a dog running in the park",
        "a car parked on the street",
        "a bird flying in the sky",
        "people walking on the sidewalk",
        "a red flower in the garden",
        "children playing in the yard",
        "a boat floating on water",
        "a mountain covered with snow",
        "a woman standing in a house",
        "a man walking with a dog",
        "a child playing with a cat",
        "a blue car on the street",
        "a green tree in the park",
        "a yellow flower in the garden",
        "a white bird flying",
        "a brown dog sleeping",
        "a black cat sitting",
        "people walking in the park",
        "a beautiful mountain landscape",
        "trees and sky in nature",
        "a scenic view with mountains",
        "green forest and blue sky",
        "a natural outdoor scene",
        "landscape with trees and hills",
        "a view of nature and sky",
        "mountains covered with green trees",
        "a peaceful natural setting",
        "outdoor scene with vegetation"
    ]
    
    # Create simple tokenizer
    tokenizer = Tokenizer(num_words=1000, oov_token='<unk>')
    processed_captions = [f'<start> {cap} <end>' for cap in sample_captions]
    tokenizer.fit_on_texts(processed_captions)
    
    # Save sample tokenizer
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Create and save a simple model
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 10
    model = create_captioning_model(vocab_size, max_length)
    model.save_weights('saved_models/captioning_model.weights.h5')
    
    print("Sample model and tokenizer created!")
    print(f"Vocabulary size: {vocab_size}")
    print("You can now run the Flask app to test the interface.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Image Captioning Model')
    parser.add_argument('--data_dir', default='data/train2017', 
                       help='Directory containing training images')
    parser.add_argument('--annotation_file', default='data/annotations/captions_train2017.json',
                       help='Path to training annotations')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--max_vocab_size', type=int, default=8000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max_length', type=int, default=34,
                       help='Maximum caption length')
    parser.add_argument('--use_subset', action='store_true',
                       help='Use subset of data for quick training')
    parser.add_argument('--subset_size', type=int, default=10000,
                       help='Size of subset if using subset')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample data for testing')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
    else:
        # Check if CUDA is available
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
        
        train_captioning_model(
            data_dir=args.data_dir,
            annotation_file=args.annotation_file,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_vocab_size=args.max_vocab_size,
            max_length=args.max_length,
            use_subset=args.use_subset,
            subset_size=args.subset_size
        )