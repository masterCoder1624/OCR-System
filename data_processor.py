import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import zipfile
import fitz  # PyMuPDF
import tempfile
import shutil
import gc

class OCRDataset(Dataset):
    def __init__(self, zip_path, transform=None, is_training=True):
        self.zip_path = zip_path
        self.transform = transform
        self.is_training = is_training
        self.image_paths = []
        self.texts = []
        
        # Create a temporary directory for extracted files
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")
        
        # Extract zip file to temporary directory
        print(f"Extracting zip file: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        
        # Process the extracted files
        self._process_files(self.temp_dir)
        
        # Print dataset statistics
        print(f"Total samples loaded: {len(self.image_paths)}")
    
    def _process_files(self, extract_dir):
        # Walk through the extracted directory
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    print(f"Processing PDF: {file}")
                    self._process_pdf(pdf_path)
                elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Handle regular image files
                    text_file = os.path.splitext(file)[0] + '.txt'
                    text_path = os.path.join(root, text_file)
                    
                    if os.path.exists(text_path):
                        self.image_paths.append(os.path.join(root, file))
                        with open(text_path, 'r', encoding='utf-8') as f:
                            self.texts.append(f.read().strip())
    
    def _process_pdf(self, pdf_path):
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    
                    # Extract text
                    text = page.get_text()
                    
                    # Extract image with lower resolution to save space
                    pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))  # 150 DPI instead of 300
                    
                    # Convert to numpy array directly instead of saving to disk
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    
                    # Store the extracted data
                    if text.strip():  # Only store if there's text content
                        # Convert to RGB if needed
                        if pix.n == 1:  # Grayscale
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                        elif pix.n == 4:  # RGBA
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                        
                        # Store the image array and text
                        self.image_paths.append(img_array)
                        self.texts.append(text.strip())
                    
                    # Clear memory
                    del pix
                    del img_array
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing page {page_num} of {pdf_path}: {str(e)}")
                    continue
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get the image (either path or array)
        img_data = self.image_paths[idx]
        text = self.texts[idx]
        
        # If img_data is already an array, use it directly
        if isinstance(img_data, np.ndarray):
            image = img_data
        else:
            # Otherwise, read from file
            image = cv2.imread(img_data)
            if image is None:
                raise ValueError(f"Failed to load image: {img_data}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, text
    
    def __del__(self):
        # Cleanup temporary directory when the dataset is destroyed
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

def get_transforms(is_training=True):
    if is_training:
        return A.Compose([
            A.Resize(224, 224),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def create_data_loaders(train_zip_path, val_zip_path, batch_size=32):
    print("Creating data loaders...")
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    train_dataset = OCRDataset(train_zip_path, transform=train_transform, is_training=True)
    val_dataset = OCRDataset(val_zip_path, transform=val_transform, is_training=False)
    
    print(f"Found {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader 