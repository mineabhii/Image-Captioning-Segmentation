# Integrated Image Captioning and Segmentation Project

This project, based on the Zidio Development Data Science internship, combines two major tasks in computer vision and NLP: **Image Captioning** and **Image Segmentation**. An image is provided as input, and the system generates a descriptive text caption and a semantic segmentation mask.

## 📜 Project Overview

- **Image Captioning**: A CNN (InceptionV3) acts as an encoder to extract features from the image, and an LSTM acts as a decoder to generate a caption word-by-word.
- **Image Segmentation**: A U-Net model is used to perform semantic segmentation, classifying each pixel of the image into a category.
- **Integration**: Both models are integrated into a single Flask web application where users can upload images and view both the generated caption and the segmentation mask.

## 🛠 Tech Stack

- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Development**: Jupyter Notebook (for experimentation), VS Code

## 📁 Project Structure

```
Image_Captioning_and_Segmentation/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/                       # Dataset directory (MS COCO 2017)
│   ├── train2017/             # Training images
│   ├── val2017/               # Validation images
│   └── annotations/           # COCO annotations
├── models/                     # Model architecture files
│   ├── captioning_model.py    # Image captioning model
│   └── segmentation_model.py  # U-Net segmentation model
├── training/                   # Training scripts
│   └── train_captioning.py    # Caption model training
├── utils/                      # Utility functions
│   ├── preprocessing.py       # Data preprocessing
│   └── inference.py          # Model inference
├── templates/                  # HTML templates
│   ├── base.html             # Base template
│   ├── index.html            # Main page
│   └── results.html          # Results page
├── static/                     # Static files
│   └── uploads/              # Uploaded images
├── saved_models/              # Trained models
│   ├── tokenizer.pkl         # Text tokenizer
│   └── captioning_model_weights.h5  # Model weights
└── venv/                      # Virtual environment
```

## ⚙ Setup and Installation

### Prerequisites
- Python 3.8+ (recommended: Python 3.9 or 3.10)
- At least 8GB RAM (16GB recommended for training)
- GPU with CUDA support (optional, but recommended for training)

### Step 1: Clone/Download the Project
```bash
# If you have the project files, navigate to the directory
cd Image_Captioning_and_Segmentation
```

### Step 2: Set up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset (Optional - for training)
The training scripts require the MS COCO 2017 dataset:

1. **Training Images**: [Download 2017 Train images (18 GB)](http://images.cocodataset.org/zips/train2017.zip)
2. **Validation Images**: [Download 2017 Val images (1 GB)](http://images.cocodataset.org/zips/val2017.zip)
3. **Annotations**: [Download 2017 Train/Val annotations (241 MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

Extract them into the `data/` folder:
```
data/
├── annotations/
│   ├── captions_train2017.json
│   └── captions_val2017.json
├── train2017/
│   └── (training images)
└── val2017/
    └── (validation images)
```

## 🚀 How to Run

### Option 1: Quick Start (Without Training)

For testing the interface without training models:

```bash
# Create sample model for testing
python training/train_captioning.py --create_sample

# Run the Flask application
python app.py
```

### Option 2: Full Training Pipeline

If you have the COCO dataset:

```bash
# Train the captioning model (this will take several hours)
python training/train_captioning.py --use_subset --subset_size 10000

# Run the Flask application
python app.py
```

### Training Options

```bash
# Train with full dataset
python training/train_captioning.py

# Train with subset (faster for testing)
python training/train_captioning.py --use_subset --subset_size 5000

# Custom training parameters
python training/train_captioning.py \
    --batch_size 64 \
    --epochs 20 \
    --max_vocab_size 10000 \
    --max_length 40
```

### Web Application

1. Open your browser and go to `http://127.0.0.1:5000`
2. Upload an image using the interface
3. Click "Generate Caption & Segmentation"
4. View the results: original image, generated caption, and segmentation mask

## 📊 Model Architectures

### Image Captioning Model
- **Encoder**: InceptionV3 (pre-trained on ImageNet) for feature extraction
- **Decoder**: LSTM network with attention mechanism
- **Input**: 299×299 RGB images
- **Output**: Text captions (max 34 words)
- **Training Data**: MS COCO 2017 dataset

### Segmentation Model  
- **Architecture**: U-Net with encoder-decoder structure
- **Input**: 256×256 RGB images
- **Output**: Pixel-wise class predictions (21 classes)
- **Features**: Skip connections, batch normalization, dropout

## 🎯 Usage Examples

### Web Interface
1. Navigate to the web application
2. Upload an image (PNG, JPG, JPEG, GIF, BMP)
3. View generated results

### Programmatic Usage
```python
from models.captioning_model import create_captioning_model
from utils.inference import generate_caption
import pickle

# Load trained model and tokenizer
with open('saved_models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = create_captioning_model(len(tokenizer.word_index) + 1, 34)
model.load_weights('saved_models/captioning_model_weights.h5')

# Generate caption for an image
caption = generate_caption('path/to/image.jpg', model, tokenizer)
print(f"Generated caption: {caption}")
```

## 📈 Performance Notes

### Training Time
- **Captioning Model**: 2-6 hours (depending on subset size and hardware)
- **Full Dataset**: 12-24 hours with GPU
- **CPU Training**: Not recommended (very slow)

### Model Sizes
- **InceptionV3 Features**: ~100MB
- **Captioning Model**: ~50MB  
- **Tokenizer**: ~1MB
- **Total**: ~150MB

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA RTX 3080/4080 or better

## 🔧 Development

### Project Structure
The project follows a modular design:
- `models/`: Contains neural network architectures
- `utils/`: Helper functions for preprocessing and inference  
- `training/`: Training scripts and data loaders
- `templates/`: Web interface templates
- `static/`: Static web assets

### Adding New Features
1. **New Model Architecture**: Add to `models/` directory
2. **New Preprocessing**: Add functions to `utils/preprocessing.py`
3. **New Training Script**: Create in `training/` directory
4. **Web Interface Updates**: Modify templates and Flask routes

### Testing
```bash
# Test model loading
python utils/inference.py

# Test individual components
python models/captioning_model.py
python models/segmentation_model.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size in training script
   - Use smaller subset of data
   - Close other applications

2. **CUDA/GPU Issues**:
   - Ensure CUDA is properly installed
   - Check TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"`
   - Fall back to CPU training if needed

3. **Missing Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Model Loading Errors**:
   - Ensure models are trained first
   - Check file paths in `saved_models/` directory
   - Run sample model creation if needed

### Debug Mode
Run Flask in debug mode for development:
```bash
export FLASK_DEBUG=1  # On Windows: set FLASK_DEBUG=1
python app.py
```

## 📸 Screenshots
<img width="1920" height="1080" alt="2025-10-13 (2)" src="https://github.com/user-attachments/assets/53504fa5-4fa6-4849-ac62-96676790066d" />
<img width="1920" height="1080" alt="2025-10-13 (3)" src="https://github.com/user-attachments/assets/82fe7e94-ac9c-4294-89b1-a376cc858482" />
<img width="1920" height="1080" alt="2025-10-13 (4)" src="https://github.com/user-attachments/assets/4a096555-33e6-4551-bc8f-fb0f013d0d27" />
<img width="1920" height="1080" alt="2025-10-13 (1)" src="https://github.com/user-attachments/assets/29c679c4-fb37-46ee-bcfa-d97b7d2cfc92" />
<img width="1920" height="1080" alt="2025-10-13 (5)" src="https://github.com/user-attachments/assets/67480792-360a-4c55-afeb-1c36e0d9a359" />
<img width="1920" height="1080" alt="2025-10-13 (6)" src="https://github.com/user-attachments/assets/22882897-7e77-41b3-a84d-831d62773594" />
<img width="1920" height="1080" alt="2025-10-13 (7)" src="https://github.com/user-attachments/assets/f860f6f0-044d-446c-95fe-7c65745e3a3e" />


## 📄 License

This project is created for educational purposes as part of the Zidio Development Data Science internship.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure you have sufficient hardware resources
4. Try the sample model creation for quick testing
5. If you’d like to connect or provide feedback:  
    - **Author:** Abhishek Kumar  
    - **GitHub:** mineabhbii(https://github.com/mineabhii)  

---

**Note**: Training these models is computationally expensive and requires significant time and resources. For demonstration purposes, you can use the sample model creation feature to test the interface quickly.
