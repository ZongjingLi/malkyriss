import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class MovingSpritesDataset(Dataset):
    """
    Dataset of simple moving sprites for training and testing the Counterfactual World Model.
    """
    def __init__(self, num_samples=1000, img_size=64, num_sprites=3, sprite_size=8, 
                 max_speed=3, sequence_length=2):
        """
        Args:
            num_samples: Number of video sequences to generate
            img_size: Size of the square images (height=width=img_size)
            num_sprites: Number of sprites in each image
            sprite_size: Size of the square sprites
            max_speed: Maximum speed of sprites
            sequence_length: Number of frames in each sequence
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_sprites = num_sprites
        self.sprite_size = sprite_size
        self.max_speed = max_speed
        self.sequence_length = sequence_length
        
        # Generate dataset
        self.sequences = []
        self.positions = []
        self.velocities = []
        self.colors = []
        
        for i in range(num_samples):
            sequence, positions, velocities, colors = self._generate_sequence()
            self.sequences.append(sequence)
            self.positions.append(positions)
            self.velocities.append(velocities)
            self.colors.append(colors)
            
    def _generate_sequence(self):
        """Generate a sequence of frames with moving sprites."""
        # Initialize positions, velocities, and colors for sprites
        positions = []
        velocities = []
        colors = []
        
        # Create a background
        sequence = np.zeros((self.sequence_length, 3, self.img_size, self.img_size), dtype=np.float32)
        
        # Add random background noise
        for t in range(self.sequence_length):
            sequence[t] = np.random.uniform(0, 0.1, (3, self.img_size, self.img_size))
        
        # For each sprite
        for i in range(self.num_sprites):
            # Generate random initial position (ensuring the sprite fits within the image)
            x = np.random.randint(self.sprite_size // 2, self.img_size - self.sprite_size // 2)
            y = np.random.randint(self.sprite_size // 2, self.img_size - self.sprite_size // 2)
            positions.append([(x, y)])
            
            # Generate random velocity
            vx = np.random.randint(-self.max_speed, self.max_speed + 1)
            vy = np.random.randint(-self.max_speed, self.max_speed + 1)
            velocities.append((vx, vy))
            
            # Generate random color (RGB)
            color = np.random.uniform(0.5, 1.0, 3)
            colors.append(color)
            
            # Add sprite to the first frame
            x_start = max(0, x - self.sprite_size // 2)
            x_end = min(self.img_size, x + self.sprite_size // 2)
            y_start = max(0, y - self.sprite_size // 2)
            y_end = min(self.img_size, y + self.sprite_size // 2)
            
            for c in range(3):
                sequence[0, c, y_start:y_end, x_start:x_end] = color[c]
            
            # Update position for remaining frames
            for t in range(1, self.sequence_length):
                x += vx
                y += vy
                
                # Bounce off boundaries
                if x < self.sprite_size // 2 or x >= self.img_size - self.sprite_size // 2:
                    vx = -vx
                    x += 2 * vx  # Correct position
                
                if y < self.sprite_size // 2 or y >= self.img_size - self.sprite_size // 2:
                    vy = -vy
                    y += 2 * vy  # Correct position
                
                positions[i].append((x, y))
                
                # Add sprite to the current frame
                x_start = max(0, x - self.sprite_size // 2)
                x_end = min(self.img_size, x + self.sprite_size // 2)
                y_start = max(0, y - self.sprite_size // 2)
                y_end = min(self.img_size, y + self.sprite_size // 2)
                
                for c in range(3):
                    sequence[t, c, y_start:y_end, x_start:x_end] = color[c]
        
        # Convert to torch tensor
        sequence_tensor = torch.from_numpy(sequence)
        
        return sequence_tensor, positions, velocities, colors
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a sequence of frames."""
        return self.sequences[idx]
    
    def get_flow_ground_truth(self, idx):
        """
        Generate ground truth optical flow for a sequence.
        
        Returns:
            flow: Tensor of shape [sequence_length-1, 2, img_size, img_size]
                  Where flow[t, 0] is the x-displacement and flow[t, 1] is the y-displacement
        """
        flow = np.zeros((self.sequence_length - 1, 2, self.img_size, self.img_size), dtype=np.float32)
        
        # Get positions and velocities for the sequence
        positions = self.positions[idx]
        velocities = self.velocities[idx]
        
        for t in range(self.sequence_length - 1):
            # For each sprite
            for i, ((x1, y1), (x2, y2)) in enumerate(zip([positions[i][t] for i in range(self.num_sprites)],
                                                         [positions[i][t+1] for i in range(self.num_sprites)])):
                # Calculate center position and flow
                vx, vy = x2 - x1, y2 - y1
                
                # Add flow to the region covered by the sprite
                x_start = max(0, x1 - self.sprite_size // 2)
                x_end = min(self.img_size, x1 + self.sprite_size // 2)
                y_start = max(0, y1 - self.sprite_size // 2)
                y_end = min(self.img_size, y1 + self.sprite_size // 2)
                
                flow[t, 0, y_start:y_end, x_start:x_end] = vx
                flow[t, 1, y_start:y_end, x_start:x_end] = vy
        
        return torch.from_numpy(flow)
    
    def get_segmentation_ground_truth(self, idx, frame_idx=0):
        """
        Generate ground truth segmentation mask for a frame.
        
        Args:
            idx: Index of the sequence
            frame_idx: Index of the frame within the sequence
        
        Returns:
            segmentation: Tensor of shape [num_sprites, img_size, img_size]
                         Where segmentation[i] is a binary mask for sprite i
        """
        segmentation = np.zeros((self.num_sprites, self.img_size, self.img_size), dtype=np.float32)
        
        # Get positions for the sequence
        positions = self.positions[idx]
        
        # For each sprite
        for i in range(self.num_sprites):
            x, y = positions[i][frame_idx]
            
            # Add segmentation mask for the sprite
            x_start = max(0, x - self.sprite_size // 2)
            x_end = min(self.img_size, x + self.sprite_size // 2)
            y_start = max(0, y - self.sprite_size // 2)
            y_end = min(self.img_size, y + self.sprite_size // 2)
            
            segmentation[i, y_start:y_end, x_start:x_end] = 1.0
        
        return torch.from_numpy(segmentation)
    
    def visualize_sequence(self, idx):
        """Visualize a sequence of frames."""
        sequence = self.sequences[idx]
        
        fig, axes = plt.subplots(1, self.sequence_length, figsize=(4*self.sequence_length, 4))
        if self.sequence_length == 1:
            axes = [axes]
            
        for t in range(self.sequence_length):
            frame = sequence[t].permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0, 1)
            axes[t].imshow(frame)
            axes[t].set_title(f"Frame {t}")
            axes[t].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_flow(self, idx):
        """Visualize optical flow ground truth."""
        flow = self.get_flow_ground_truth(idx)
        
        fig, axes = plt.subplots(1, self.sequence_length - 1, figsize=(4*(self.sequence_length-1), 4))
        if self.sequence_length - 1 == 1:
            axes = [axes]
            
        for t in range(self.sequence_length - 1):
            # Normalize flow for visualization
            flow_x = flow[t, 0].numpy()
            flow_y = flow[t, 1].numpy()
            
            # Calculate flow magnitude and direction
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            direction = np.arctan2(flow_y, flow_x)
            
            # Create HSV image
            hsv = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            hsv[..., 0] = (direction + np.pi) / (2 * np.pi)  # Hue (direction)
            hsv[..., 1] = np.minimum(1.0, magnitude / self.max_speed)  # Saturation (magnitude)
            hsv[..., 2] = (magnitude > 0).astype(np.float32)  # Value (mask)
            
            # Convert to RGB
            from matplotlib.colors import hsv_to_rgb
            rgb = hsv_to_rgb(hsv)
            
            axes[t].imshow(rgb)
            axes[t].set_title(f"Flow {t} to {t+1}")
            axes[t].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_segmentation(self, idx, frame_idx=0):
        """Visualize segmentation ground truth."""
        segmentation = self.get_segmentation_ground_truth(idx, frame_idx)
        
        fig, axes = plt.subplots(1, self.num_sprites + 1, figsize=(4*(self.num_sprites+1), 4))
        
        # Show the original frame
        frame = self.sequences[idx][frame_idx].permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        axes[0].imshow(frame)
        axes[0].set_title(f"Frame {frame_idx}")
        axes[0].axis('off')
        
        # Show segmentation for each sprite
        for i in range(self.num_sprites):
            axes[i+1].imshow(segmentation[i].numpy(), cmap='gray')
            axes[i+1].set_title(f"Sprite {i}")
            axes[i+1].axis('off')
        
        plt.tight_layout()
        return fig