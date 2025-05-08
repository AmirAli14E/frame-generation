"""
Frame Generation with Convolutional LSTM
----------------------------------------
A PyTorch implementation for video frame prediction using convolutional networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
CONFIG = {
    'video_path': 'input_video.mp4',
    'frame_size': (128, 128),
    'sequence_length': 5,
    'batch_size': 4,
    'learning_rate': 0.0002,
    'epochs': 10,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

class VideoDataset(Dataset):
    """Custom dataset for video frame sequences"""
    
    def __init__(self, video_path, seq_length=5, resize=(128, 128)):
        self.cap = cv2.VideoCapture(video_path)
        self.frames = self._load_frames(resize)
        self.seq_length = seq_length
        
    def _load_frames(self, resize):
        """Load and preprocess video frames"""
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
        self.cap.release()
        return frames
    
    def __len__(self):
        return len(self.frames) - self.seq_length
    
    def __getitem__(self, idx):
        seq = self.frames[idx:idx+self.seq_length]
        input_frames = np.array([frame.transpose(2, 0, 1) for frame in seq[:-1]])
        target_frame = seq[-1].transpose(2, 0, 1)
        return torch.FloatTensor(input_frames/255.0), torch.FloatTensor(target_frame/255.0)

class FrameGenerator(nn.Module):
    """Convolutional Autoencoder for frame prediction"""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

def create_test_video(output_path, frame_size=(128, 128), frames=30):
    """Generate a test video with sequential frames"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, frame_size)
    
    for i in range(frames):
        frame = np.zeros((*frame_size, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i+1}", (10, 64), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out.write(frame)
    out.release()
    print(f"✅ Test video created at {output_path}")

def train_model():
    """Main training procedure"""
    
    # Setup device
    device = CONFIG['device']
    print(f"\nUsing device: {device}")
    
    # Create test data if needed
    if not Path(CONFIG['video_path']).exists():
        create_test_video(CONFIG['video_path'], CONFIG['frame_size'])
    
    # Initialize dataset and model
    dataset = VideoDataset(CONFIG['video_path'], 
                          CONFIG['sequence_length'], 
                          CONFIG['frame_size'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = FrameGenerator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.L1Loss()
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs = inputs.mean(dim=1).to(device)  # Average input frames
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {avg_loss:.4f}")
    
    return model

def evaluate_model(model, dataset):
    """Evaluate and visualize model predictions"""
    model.eval()
    inputs, target = dataset[0]
    
    with torch.no_grad():
        inputs = inputs.mean(dim=0).unsqueeze(0).to(CONFIG['device'])
        predicted = model(inputs).cpu()
    
    # Display results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(inputs[0].cpu().numpy().transpose(1, 2, 0))
    ax1.set_title("Input Frame")
    ax1.axis('off')
    
    ax2.imshow(predicted[0].numpy().transpose(1, 2, 0))
    ax2.set_title("Prediction")
    ax2.axis('off')
    
    ax3.imshow(target.numpy().transpose(1, 2, 0))
    ax3.set_title("Target Frame")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the complete pipeline
    trained_model = train_model()
    dataset = VideoDataset(CONFIG['video_path'])
    evaluate_model(trained_model, dataset)