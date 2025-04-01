from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import tempfile
import cv2
import numpy as np
import zipfile
import io
from PIL import Image
import pytesseract
import re

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

def clean_text(text):
    """Clean and normalize extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()

def process_pdf(file_path):
    """Process PDF with improved text extraction"""
    try:
        doc = fitz.open(file_path)
        text = ""
        
        for i, page in enumerate(doc):
            # Get text from PDF
            pdf_text = page.get_text()
            text += f"Page {i+1}:\n{pdf_text}\n\n"
        
        doc.close()
        return clean_text(text)
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def process_image(file_path):
    """Process image with improved OCR"""
    try:
        # Read image
        image = cv2.imread(file_path)
        if image is None:
            return "Error: Could not read image file"

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        
        # Apply median blur to remove noise
        gray = cv2.medianBlur(gray, 3)
        
        # Apply OCR
        text = pytesseract.image_to_string(gray)
        
        return clean_text(text)
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_zip(file_path):
    """Process ZIP file containing multiple documents"""
    try:
        results = []
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                    # Extract file to temporary location
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file_info.filename)
                    zip_ref.extract(file_info, app.config['UPLOAD_FOLDER'])
                    
                    try:
                        if file_info.filename.lower().endswith('.pdf'):
                            text = process_pdf(temp_path)
                        else:
                            text = process_image(temp_path)
                        
                        results.append({
                            'filename': file_info.filename,
                            'text': text
                        })
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        
        return results
    except Exception as e:
        return f"Error processing ZIP file: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        if filename.lower().endswith('.zip'):
            results = process_zip(file_path)
            if isinstance(results, str):  # Error occurred
                return jsonify({'error': results}), 500
            return jsonify({'results': results})
        elif filename.lower().endswith('.pdf'):
            text = process_pdf(file_path)
            return jsonify({'text': text})
        else:
            text = process_image(file_path)
            return jsonify({'text': text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 