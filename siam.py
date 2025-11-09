import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import Tuple, List
import random

# ===================== MODEL ARCHITECTURE =====================
class SiamFC(nn.Module):
    """SiamFC - Siamese Fully Convolutional Network"""
    def __init__(self):
        super(SiamFC, self).__init__()
        
        # Backbone: AlexNet-like architecture
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3),
            nn.BatchNorm2d(256),
        )
        
    def forward(self, z, x):
        """
        z: template (exemplar) - [B, 3, 127, 127]
        x: search region - [B, 3, 255, 255]
        """
        # Extract features
        z_feat = self.features(z)  # [B, 256, 6, 6]
        x_feat = self.features(x)  # [B, 256, 22, 22]
        
        # Cross-correlation
        response = self.xcorr(z_feat, x_feat)
        return response
    
    def xcorr(self, z, x):
        """Cross-correlation between template and search region"""
        batch_size = x.size(0)
        # Use grouped convolution for batch processing
        out = []
        for i in range(batch_size):
            # Treat z as kernel and x as input
            kernel = z[i:i+1]  # [1, 256, 6, 6]
            feature = x[i:i+1]  # [1, 256, 22, 22]
            
            # Perform correlation
            out_i = torch.nn.functional.conv2d(feature, kernel)
            out.append(out_i)
        
        return torch.cat(out, dim=0)  # [B, 1, 17, 17]


# ===================== DATASET =====================
class SiamFCDataset(Dataset):
    """Dataset for SiamFC training"""
    def __init__(self, root_dir: str, max_frames: int = 100):
        self.root_dir = Path(root_dir)
        self.max_frames = max_frames
        self.videos = []
        
        # Load all videos with template images
        samples_dir = self.root_dir / 'samples'
        for obj_dir in samples_dir.iterdir():
            if obj_dir.is_dir():
                video_file = obj_dir / 'drone_video.mp4'
                object_images_dir = obj_dir / 'object_images'
                
                if video_file.exists() and object_images_dir.exists():
                    # Load all template images (img_1.jpg, img_2.jpg, img_3.jpg)
                    template_images = []
                    for img_file in sorted(object_images_dir.glob('img_*.jpg')):
                        template_images.append(str(img_file))
                    
                    if template_images:
                        self.videos.append({
                            'video': str(video_file),
                            'templates': template_images,
                            'object_name': obj_dir.name
                        })
        
        print(f"Loaded {len(self.videos)} videos for training")
    
    def __len__(self):
        return len(self.videos) * self.max_frames
    
    def __getitem__(self, idx):
        # Select video
        video_idx = idx // self.max_frames
        video_info = self.videos[video_idx]
        
        # Randomly select one of the 3 template images
        template_path = random.choice(video_info['templates'])
        template_img = cv2.imread(template_path)
        
        # Load video
        cap = cv2.VideoCapture(video_info['video'])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Randomly select a search frame
        search_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, search_idx)
        ret, search_frame = cap.read()
        cap.release()
        
        if not ret or search_frame is None:
            # Fallback: return first frame if selected frame fails
            cap = cv2.VideoCapture(video_info['video'])
            ret, search_frame = cap.read()
            cap.release()
        
        # Process template - resize to 127x127
        template = cv2.resize(template_img, (127, 127))
        
        # For search region, we need to simulate object at random position
        # Assume object is somewhere in the frame, add some augmentation
        h, w = search_frame.shape[:2]
        
        # Random crop and resize to 255x255
        scale = random.uniform(0.5, 1.5)
        crop_size = int(255 / scale)
        crop_size = min(crop_size, min(h, w))
        
        x_start = random.randint(0, max(0, w - crop_size))
        y_start = random.randint(0, max(0, h - crop_size))
        
        search_crop = search_frame[y_start:y_start+crop_size, x_start:x_start+crop_size]
        search = cv2.resize(search_crop, (255, 255))
        
        # Convert to tensors
        template = torch.from_numpy(template.transpose(2, 0, 1)).float() / 255.0
        search = torch.from_numpy(search.transpose(2, 0, 1)).float() / 255.0
        
        # Create ground truth response map (17x17) - center position with random offset
        gt_response = self.create_gt_response_center()
        
        return template, search, gt_response
    
    def crop_and_resize(self, image, bbox, output_size):
        """Crop object with context and resize"""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        
        # Add context (2x the object size)
        context = 0.5 * (w + h)
        size = np.sqrt((w + context) * (h + context))
        
        # Crop region
        x1 = int(cx - size/2)
        y1 = int(cy - size/2)
        x2 = int(cx + size/2)
        y2 = int(cy + size/2)
        
        # Handle boundaries
        img_h, img_w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        
        crop = image[y1:y2, x1:x2]
        
        # Resize
        crop_resized = cv2.resize(crop, (output_size, output_size))
        
        # Calculate new bbox position in cropped image
        new_bbox = [
            (x - x1) * output_size / (x2 - x1),
            (y - y1) * output_size / (y2 - y1),
            w * output_size / (x2 - x1),
            h * output_size / (y2 - y1)
        ]
        
        return crop_resized, new_bbox
    
    def create_gt_response_center(self):
        """Create ground truth response map centered with small random offset"""
        response_size = 17
        
        # Center with small random offset
        cx = 8.5 + random.uniform(-2, 2)
        cy = 8.5 + random.uniform(-2, 2)
        
        # Create Gaussian response
        gt = np.zeros((response_size, response_size), dtype=np.float32)
        sigma = 2.0
        
        for i in range(response_size):
            for j in range(response_size):
                dist = np.sqrt((i - cy)**2 + (j - cx)**2)
                gt[i, j] = np.exp(-dist**2 / (2 * sigma**2))
        
        return torch.from_numpy(gt).unsqueeze(0)  # [1, 17, 17]


# ===================== TRAINING =====================
def train_siamfc(root_dir: str, num_epochs: int = 50, batch_size: int = 8, 
                 lr: float = 1e-3, save_dir: str = 'checkpoints', use_cpu: bool = False):
    """Train SiamFC model"""
    
    # Setup device with better error handling
    if use_cpu:
        device = torch.device('cpu')
        print(f"Using device: CPU (forced)")
    else:
        if torch.cuda.is_available():
            try:
                # Test CUDA before using it
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                device = torch.device('cuda')
                print(f"Using device: CUDA - {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"CUDA available but error occurred: {e}")
                print("Falling back to CPU...")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print(f"CUDA not available. Using CPU")
    
    print(f"Final device: {device}\n")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    model = SiamFC().to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Dataset and dataloader - reduce num_workers if using CPU
    dataset = SiamFCDataset(root_dir)
    num_workers = 0 if device.type == 'cpu' else 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (template, search, gt_response) in enumerate(dataloader):
            # Move to device
            template = template.to(device)
            search = search.to(device)
            gt_response = gt_response.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            response = model(template, search)
            
            # Compute loss
            loss = criterion(response, gt_response)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        # Scheduler step
        scheduler.step()
        
        # Save checkpoint
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'siamfc_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    final_path = os.path.join(save_dir, 'siamfc_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved: {final_path}')
    
    return model


# ===================== INFERENCE =====================
class SiamFCTracker:
    """SiamFC Tracker for inference"""
    def __init__(self, model_path: str, template_images: List[str] = None, use_cpu: bool = False):
        # Force CPU if specified or if CUDA has issues
        if use_cpu:
            self.device = torch.device('cpu')
        else:
            try:
                if torch.cuda.is_available():
                    test = torch.zeros(1).cuda()
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
            except:
                self.device = torch.device('cpu')
        
        self.model = SiamFC().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load template images if provided
        if template_images:
            self.templates = []
            for img_path in template_images:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (127, 127))
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                self.templates.append(img_tensor.unsqueeze(0).to(self.device))
        else:
            self.templates = None
        
    def init(self, frame, bbox, use_template_image: bool = True):
        """Initialize tracker with first frame and bbox"""
        self.bbox = bbox
        
        if use_template_image and self.templates:
            # Use one of the pre-loaded template images (average all 3 for better robustness)
            self.template = torch.cat(self.templates, dim=0).mean(dim=0, keepdim=True)
        else:
            # Extract template from frame
            x, y, w, h = bbox
            template = self.crop_and_resize(frame, bbox, 127)
            self.template = torch.from_numpy(template.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            self.template = self.template.to(self.device)
        
    def track(self, frame):
        """Track object in new frame"""
        # Extract search region
        search = self.crop_and_resize(frame, self.bbox, 255)
        search_tensor = torch.from_numpy(search.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        search_tensor = search_tensor.to(self.device)
        
        # Get response map
        with torch.no_grad():
            response = self.model(self.template, search_tensor)
            response = torch.sigmoid(response)
        
        # Find maximum response
        response = response.squeeze().cpu().numpy()
        max_idx = np.unravel_index(response.argmax(), response.shape)
        
        # Update bbox
        dy, dx = max_idx
        scale = 255 / 17
        dx = (dx - 8) * scale
        dy = (dy - 8) * scale
        
        x, y, w, h = self.bbox
        cx, cy = x + w/2, y + h/2
        cx += dx
        cy += dy
        
        self.bbox = [cx - w/2, cy - h/2, w, h]
        return self.bbox
    
    def crop_and_resize(self, image, bbox, output_size):
        """Crop object with context and resize"""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        
        context = 0.5 * (w + h)
        size = np.sqrt((w + context) * (h + context))
        
        x1 = int(cx - size/2)
        y1 = int(cy - size/2)
        x2 = int(cx + size/2)
        y2 = int(cy + size/2)
        
        img_h, img_w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        
        crop = image[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (output_size, output_size))
        
        return crop_resized


# ===================== MAIN =====================
if __name__ == '__main__':
    import sys
    
    # Auto detect root directory
    # If running from Backpack_0 folder, go up to train folder
    current_dir = Path.cwd()
    
    # Find the train directory
    if 'samples' in str(current_dir):
        # Running from inside samples folder, go up
        root_dir = current_dir.parent.parent
    elif 'train' in str(current_dir):
        # Already in train folder
        root_dir = current_dir
    else:
        # Need to specify manually
        root_dir = Path('aeroeyes_training-20251109T065913Z-1-001/aeroeyes_training/observing/train')
    
    print(f"Using root directory: {root_dir}")
    print(f"Looking for samples in: {root_dir / 'samples'}")
    
    # Verify path exists
    if not (root_dir / 'samples').exists():
        print(f"\nERROR: Cannot find samples directory!")
        print(f"Please run this script from the 'train' directory, or")
        print(f"Update root_dir manually in the code.")
        print(f"\nExpected structure:")
        print(f"  train/")
        print(f"    samples/")
        print(f"      Backpack_0/")
        print(f"        object_images/")
        print(f"        drone_video.mp4")
        sys.exit(1)
    
    # Training
    model = train_siamfc(
        root_dir=str(root_dir),
        num_epochs=50,
        batch_size=8,  # Reduce to 4 if using CPU
        lr=1e-3,
        save_dir='checkpoints',
        use_cpu=False  # Set to True to force CPU usage
    )
    
    print("\n=== Training completed! ===")
    print("\nTo use the tracker with template images:")
    print("template_images = ['path/to/img_1.jpg', 'path/to/img_2.jpg', 'path/to/img_3.jpg']")
    print("tracker = SiamFCTracker('checkpoints/siamfc_final.pth', template_images)")
    print("tracker.init(first_frame, initial_bbox, use_template_image=True)")
    print("bbox = tracker.track(new_frame)")