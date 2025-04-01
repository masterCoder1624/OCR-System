# OCR System

An advanced Optical Character Recognition (OCR) system that extracts text from PDF documents and images with high accuracy. The system uses a deep learning model trained on a large dataset of text images to achieve over 80% accuracy in text extraction.

## Features

- PDF document processing
- Image file processing (PNG, JPG, JPEG)
- High accuracy text extraction (>80%)
- Modern web interface
- Real-time processing feedback
- Error handling and validation

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Web browser

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ocr-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
```bash
# Place the best_model.pth file in the project root directory
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a PDF or image file using the web interface

4. Wait for the processing to complete

5. View the extracted text in the results section

## Project Structure

```
ocr-system/
├── app.py              # Flask application
├── data_processor.py   # Data processing utilities
├── model.py           # OCR model architecture
├── best_model.pth     # Pre-trained model weights
├── requirements.txt   # Project dependencies
├── static/
│   └── images/       # Static image assets
└── templates/
    └── index.html    # Web interface template
```

## Technical Details

- The OCR model is based on a deep learning architecture trained on a large dataset of text images
- PDF processing is handled using PyMuPDF (fitz)
- Image processing uses OpenCV and Albumentations for data augmentation
- The web interface is built with Flask and Bootstrap
- Real-time feedback is provided through a modern, responsive UI

## License

This project is licensed under the MIT License - see the LICENSE file for details. 