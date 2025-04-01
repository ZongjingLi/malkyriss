import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from model import CounterfactualWorldModel
from data import MovingSpritesDataset

def train_model(model, dataloader, optimizer, device, num_epochs=50):
    """
    Train the Counterfactual World Model.
    
    Args:
        model: The CounterfactualWorldModel instance
        dataloader: DataLoader with training data
        optimizer: Optimizer for training
        device: Device to run training on
        num_epochs: Number of epochs for training
    """
    model.train()
    
    # Training metrics
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, sequences in enumerate(dataloader):
            sequences = sequences.to(device)
            
            # Get frame pairs for training
            frame1 = sequences[:, 0]
            frame2 = sequences[:, 1]
            
            # Apply temporally-factored masking (keep frame1 fully visible, mask most of frame2)
            mask_ratios = [0.0, 0.99]  # As described in the paper
            
            # Forward pass
            output = model([frame1, frame2], mask_ratios)
            loss = output['loss']
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # End of epoch
        epoch_time = time.time() - start_time
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, Avg Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f"outputs/cwm_checkpoint_epoch{epoch+1}.pt")
    
    return losses

def evaluate_flow_extraction(model, dataset, device, num_samples=5):
    """
    Evaluate the model's ability to extract optical flow through counterfactual prompting.
    
    Args:
        model: Trained CounterfactualWorldModel
        dataset: MovingSpritesDataset
        device: Device to run evaluation on
        num_samples: Number of examples to evaluate
    """
    model.eval()
    
    # Results
    results = []
    
    for i in range(num_samples):
        sequence = dataset[i].to(device)
        
        # Get frame pair
        frame1 = sequence[0].unsqueeze(0)
        frame2 = sequence[1].unsqueeze(0)
        
        # Get ground truth flow
        gt_flow = dataset.get_flow_ground_truth(i)[0].unsqueeze(0).to(device)
        
        # Extract flow using the model
        with torch.no_grad():
            pred_flow, occlusion_map = model.compute_optical_flow(frame1, frame2)
        
        # Compute flow error
        flow_error = torch.norm(pred_flow - gt_flow, dim=1).mean().item()
        
        # Visualize
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Show first frame
        axes[0].imshow(frame1[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Frame 1")
        axes[0].axis('off')
        
        # Show second frame
        axes[1].imshow(frame2[0].permute(1, 2, 0).cpu().numpy())
        axes[1].set_title("Frame 2")
        axes[1].axis('off')
        
        # Visualize ground truth flow
        flow_x = gt_flow[0, 0].cpu().numpy()
        flow_y = gt_flow[0, 1].cpu().numpy()
        
        # Calculate flow magnitude and direction
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        direction = np.arctan2(flow_y, flow_x)
        
        # Create HSV image
        hsv = np.zeros((dataset.img_size, dataset.img_size, 3), dtype=np.float32)
        hsv[..., 0] = (direction + np.pi) / (2 * np.pi)  # Hue (direction)
        hsv[..., 1] = np.minimum(1.0, magnitude / dataset.max_speed)  # Saturation (magnitude)
        hsv[..., 2] = (magnitude > 0).astype(np.float32)  # Value (mask)
        
        # Convert to RGB
        from matplotlib.colors import hsv_to_rgb
        rgb_gt = hsv_to_rgb(hsv)
        
        axes[2].imshow(rgb_gt)
        axes[2].set_title("Ground Truth Flow")
        axes[2].axis('off')
        
        # Visualize predicted flow
        flow_x = pred_flow[0, 0].cpu().numpy()
        flow_y = pred_flow[0, 1].cpu().numpy()
        
        # Calculate flow magnitude and direction
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        direction = np.arctan2(flow_y, flow_x)
        
        # Create HSV image
        hsv = np.zeros((dataset.img_size, dataset.img_size, 3), dtype=np.float32)
        hsv[..., 0] = (direction + np.pi) / (2 * np.pi)  # Hue (direction)
        hsv[..., 1] = np.minimum(1.0, magnitude / dataset.max_speed)  # Saturation (magnitude)
        hsv[..., 2] = (magnitude > 0).astype(np.float32)  # Value (mask)
        
        # Convert to RGB
        rgb_pred = hsv_to_rgb(hsv)
        
        axes[3].imshow(rgb_pred)
        axes[3].set_title(f"Predicted Flow (Error: {flow_error:.4f})")
        axes[3].axis('off')
        
        plt.tight_layout()
        results.append({
            'figure': fig,
            'error': flow_error,
            'sequence_idx': i
        })
    
    return results

def evaluate_segmentation(model, dataset, device, num_samples=5):
    """
    Evaluate the model's ability to extract object segments through counterfactual prompting.
    
    Args:
        model: Trained CounterfactualWorldModel
        dataset: MovingSpritesDataset
        device: Device to run evaluation on
        num_samples: Number of examples to evaluate
    """
    model.eval()
    
    # Results
    results = []
    
    for i in range(num_samples):
        sequence = dataset[i].to(device)
        
        # Get first frame
        frame = sequence[0].unsqueeze(0)
        
        # Get ground truth segmentation
        gt_segments = dataset.get_segmentation_ground_truth(i).to(device)
        
        # Combine all sprite masks for visualization
        gt_combined = torch.zeros_like(gt_segments[0])
        for s in range(dataset.num_sprites):
            gt_combined = torch.maximum(gt_combined, gt_segments[s] * (s + 1))
        
        # Extract segments using the model
        with torch.no_grad():
            pred_segments = model.extract_segments(frame)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Show the frame
        axes[0].imshow(frame[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Input Frame")
        axes[0].axis('off')
        
        # Show ground truth segmentation
        axes[1].imshow(gt_combined.cpu().numpy(), cmap='tab10')
        axes[1].set_title("Ground Truth Segments")
        axes[1].axis('off')
        
        # Show predicted segmentation
        axes[2].imshow(pred_segments[0].cpu().numpy(), cmap='gray')
        axes[2].set_title("Predicted Segments")
        axes[2].axis('off')
        
        plt.tight_layout()
        results.append({
            'figure': fig,
            'sequence_idx': i
        })
    
    return results

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = MovingSpritesDataset(num_samples=1000, img_size=64, num_sprites=3, 
                                  sprite_size=8, max_speed=3, sequence_length=2)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize model
    model = CounterfactualWorldModel(
        img_size=64,
        patch_size=(1, 8, 8),
        in_channels=3,
        embed_dim=256,
        depth=4,
        num_heads=8
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    # Train model
    losses = train_model(model, dataloader, optimizer, device, num_epochs=50)
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('outputs/training_loss.png')
    
    # Evaluate flow extraction
    flow_results = evaluate_flow_extraction(model, dataset, device)
    for i, res in enumerate(flow_results):
        res['figure'].savefig(f'outputs/flow_extraction_{i}.png')
    
    # Evaluate segmentation
    seg_results = evaluate_segmentation(model, dataset, device)
    for i, res in enumerate(seg_results):
        res['figure'].savefig(f'outputs/segmentation_{i}.png')
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()