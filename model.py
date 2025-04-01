import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class TextRecognitionModel(nn.Module):
    def __init__(self, num_classes, hidden_size=768, num_heads=8, num_layers=6):
        super(TextRecognitionModel, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Vision Transformer
        self.vit_config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=512,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            mlp_ratio=4,
            qkv_bias=True,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
        )
        self.vit = ViTModel(self.vit_config)
        
        # Self-supervised head
        self.ssl_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # OCR head
        self.ocr_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x, ssl_mode=False):
        # CNN feature extraction
        cnn_features = self.cnn(x)
        
        # Reshape for ViT
        batch_size = cnn_features.size(0)
        cnn_features = cnn_features.permute(0, 2, 3, 1)
        cnn_features = cnn_features.reshape(batch_size, -1, 512)
        
        # Vision Transformer
        vit_output = self.vit(cnn_features).last_hidden_state
        
        if ssl_mode:
            # Self-supervised learning path
            ssl_features = self.ssl_head(vit_output[:, 0])  # Use [CLS] token
            return ssl_features
        else:
            # OCR path
            ocr_features = self.ocr_head(vit_output[:, 0])  # Use [CLS] token
            return ocr_features

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features):
        n = features.size(0)
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T)
        logits = logits / self.temperature
        
        labels = torch.arange(n, device=features.device)
        loss = F.cross_entropy(logits, labels)
        return loss 