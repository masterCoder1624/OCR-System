import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from model import TextRecognitionModel, ContrastiveLoss
from data_processor import create_data_loaders
import numpy as np
from sklearn.metrics import accuracy_score

def train_ssl(model, train_loader, optimizer, contrastive_loss, device, epoch, writer):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'SSL Epoch {epoch}')
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        # Get features for contrastive learning
        features = model(images, ssl_mode=True)
        loss = contrastive_loss(features)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        if batch_idx % 100 == 0:
            writer.add_scalar('SSL/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    
    return total_loss / len(train_loader)

def train_supervised(model, train_loader, optimizer, criterion, device, epoch, writer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Supervised Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
        
        if batch_idx % 100 == 0:
            writer.add_scalar('Supervised/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    
    accuracy = accuracy_score(all_labels, all_preds)
    writer.add_scalar('Supervised/Accuracy', accuracy, epoch)
    
    return total_loss / len(train_loader), accuracy

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch}'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    writer.add_scalar('Validation/Loss', total_loss / len(val_loader), epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    return total_loss / len(val_loader), accuracy

def main():
    # Hyperparameters
    num_classes = 1000  # Adjust based on your dataset
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = TextRecognitionModel(num_classes=num_classes).to(device)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        'OneDrive_2025-04-01.zip',
        'OneDrive_2025-04-01 (1).zip',
        batch_size=batch_size
    )
    
    # Create optimizers and loss functions
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss()
    
    # Create tensorboard writer
    writer = SummaryWriter('runs/ocr_training')
    
    # Self-supervised pre-training
    print("Starting self-supervised pre-training...")
    for epoch in range(num_epochs // 2):
        ssl_loss = train_ssl(model, train_loader, optimizer, contrastive_loss, device, epoch, writer)
        print(f'SSL Epoch {epoch}: Loss = {ssl_loss:.4f}')
    
    # Supervised fine-tuning
    print("Starting supervised fine-tuning...")
    best_accuracy = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_supervised(model, train_loader, optimizer, criterion, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}')
        print(f'Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Training completed. Best validation accuracy: {best_accuracy:.4f}')

if __name__ == '__main__':
    main() 