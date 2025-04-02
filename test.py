import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
import argparse
import os

from model import CounterfactualWorldModel
from data import MovingSpritesDataset

def visualize_counterfactual_predictions(model, dataset, device, idx=0):
    """
    Visualize counterfactual predictions by perturbing different patches.
    
    Args:
        model: Trained CounterfactualWorldModel
        dataset: MovingSpritesDataset
        device: Device to run on
        idx: Index of the sequence to use
    """
    model.eval()
    
    # Get a sequence
    sequence = dataset[idx].to(device)
    frame1 = sequence[0].unsqueeze(0)
    
    # Dimensions
    _, _, height, width = frame1.shape
    patch_size = (1, 4, 4)
    h_patches = height // patch_size[1]
    w_patches = width // patch_size[2]
    
    # Choose patches to perturb (e.g., one for each sprite)
    patch_positions = []
    for i in range(dataset.num_sprites):
        # Get sprite position in the first frame
        x, y = dataset.positions[idx][i][0]
        
        # Convert to patch indices
        patch_h = int(y // patch_size[1])
        patch_w = int(x // patch_size[2])
        
        patch_positions.append((patch_h, patch_w))
    
    # Generate counterfactual predictions by moving each patch in different directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    
    # Create a grid of visualizations
    num_sprites = len(patch_positions)
    num_directions = len(directions)
    
    fig, axes = plt.subplots(num_sprites + 1, num_directions + 1, figsize=(3*(num_directions+1), 3*(num_sprites+1)))
    
    # Show the original frame in the first column
    for i in range(num_sprites + 1):
        axes[i, 0].imshow(frame1[0].permute(1, 2, 0).cpu().numpy())
        if i == 0:
            axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
    
    # Show original sprite positions
    for i, (patch_h, patch_w) in enumerate(patch_positions):
        # Highlight the patch
        h_start = patch_h * patch_size[1]
        h_end = (patch_h + 1) * patch_size[1]
        w_start = patch_w * patch_size[2]
        w_end = (patch_w + 1) * patch_size[2]
        
        img = frame1[0].permute(1, 2, 0).cpu().numpy().copy()
        img[h_start:h_end, w_start:w_end, :] *= 0.5  # Dim the patch
        img[h_start:h_end, w_start:w_end, 1] += 0.5  # Add green highlight
        
        axes[i+1, 0].imshow(img)
        axes[i+1, 0].set_title(f"Sprite {i}")
        axes[i+1, 0].axis('off')
    
    # For each sprite and direction, generate a counterfactual prediction
    for i, (patch_h, patch_w) in enumerate(patch_positions):
        for j, (dh, dw) in enumerate(directions):
            with torch.no_grad():
                # Create a counterfactual second frame with the patch moved
                counterfactual_frame = torch.zeros_like(frame1)
                
                # Get the patch content from the original frame
                h_start = patch_h * patch_size[1]
                h_end = (patch_h + 1) * patch_size[1]
                w_start = patch_w * patch_size[2]
                w_end = (patch_w + 1) * patch_size[2]
                
                patch_content = frame1[0, :, h_start:h_end, w_start:w_end].clone()
                
                # Move the patch in the counterfactual frame
                new_h = max(0, min(h_patches - 1, patch_h + dh))
                new_w = max(0, min(w_patches - 1, patch_w + dw))
                
                new_h_start = new_h * patch_size[1]
                new_h_end = (new_h + 1) * patch_size[1]
                new_w_start = new_w * patch_size[2]
                new_w_end = (new_w + 1) * patch_size[2]
                
                counterfactual_frame[0, :, new_h_start:new_h_end, new_w_start:new_w_end] = patch_content
                
                # Create visible patch information
                visible_patches = torch.zeros(1, 1, np.prod(patch_size) * 3, device=device)
                visible_patches[0, 0] = patch_content.reshape(-1)
                patch_indices = [[new_h * w_patches + new_w]]
                
                # Get prediction
                prediction = model.predict(frame1, visible_patches, patch_indices)
                
                # Display the prediction
                axes[i+1, j+1].imshow(prediction[0].permute(1, 2, 0).cpu().numpy())
                axes[i+1, j+1].set_title(f"Move {['Right', 'Down', 'Left', 'Up'][j]}")
                axes[i+1, j+1].axis('off')
                
                # For the first row, just show the counterfactual input
                if i == 0:
                    axes[0, j+1].imshow(counterfactual_frame[0].permute(1, 2, 0).cpu().numpy())
                    axes[0, j+1].set_title(f"Direction: {['Right', 'Down', 'Left', 'Up'][j]}")
                    axes[0, j+1].axis('off')
    
    plt.tight_layout()
    return fig

def test_optical_flow_accuracy(model, dataset, device, num_samples=20):
    """
    Test the accuracy of optical flow extraction.
    
    Args:
        model: Trained CounterfactualWorldModel
        dataset: MovingSpritesDataset
        device: Device to run on
        num_samples: Number of samples to test
    """
    model.eval()
    
    # Metrics
    flow_errors = []
    
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
        
        # Compute flow error (only on non-zero ground truth flow)
        flow_mask = (torch.norm(gt_flow, dim=1) > 0).float()
        if flow_mask.sum() > 0:
            flow_error = torch.norm(pred_flow - gt_flow, dim=1) * flow_mask
            flow_error = flow_error.sum() / flow_mask.sum()
            flow_errors.append(flow_error.item())
    
    # Calculate statistics
    mean_error = np.mean(flow_errors)
    std_error = np.std(flow_errors)
    
    return mean_error, std_error

def test_segmentation_accuracy(model, dataset, device, num_samples=20):
    """
    Test the accuracy of segmentation extraction.
    
    Args:
        model: Trained CounterfactualWorldModel
        dataset: MovingSpritesDataset
        device: Device to run on
        num_samples: Number of samples to test
    """
    model.eval()
    
    # Metrics
    iou_scores = []
    
    for i in range(num_samples):
        sequence = dataset[i].to(device)
        
        # Get first frame
        frame = sequence[0].unsqueeze(0)
        
        # Get ground truth segmentation (combine all sprites into one mask)
        gt_segments = dataset.get_segmentation_ground_truth(i).to(device)
        gt_combined = torch.zeros_like(gt_segments[0])
        for s in range(dataset.num_sprites):
            gt_combined = torch.maximum(gt_combined, gt_segments[s])
        
        # Extract segments using the model
        with torch.no_grad():
            pred_segments = model.extract_segments(frame)
        
        # Threshold predicted segments
        pred_segments_binary = (pred_segments > 0.5).float()
        
        # Compute IoU
        intersection = (pred_segments_binary * gt_combined).sum()
        union = torch.clamp(pred_segments_binary + gt_combined, 0, 1).sum()
        
        if union > 0:
            iou = intersection / union
            iou_scores.append(iou.item())
    
    # Calculate statistics
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    
    return mean_iou, std_iou


def main():
    parser = argparse.ArgumentParser(description='Test Counterfactual World Model')
    parser.add_argument('--checkpoint', default = "outputs/cwm_checkpoint_epoch50.pt", type=str, help='Path to model checkpoint')
    parser.add_argument('--output_dir',  default='outputs',type=str, help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = MovingSpritesDataset(num_samples=1, img_size=64, num_sprites=3, 
                                  sprite_size=8, max_speed=3, sequence_length=2)
    
    # Initialize model
    model = CounterfactualWorldModel(
        img_size=64,
        patch_size=(1, 8, 8),
        in_channels=3,
        embed_dim=256,
        depth=4,
        num_heads=8
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Test optical flow accuracy
    flow_mean_error, flow_std_error = test_optical_flow_accuracy(model, dataset, device)
    print(f"Optical Flow Mean Error: {flow_mean_error:.4f} ± {flow_std_error:.4f}")
    
    # Test segmentation accuracy
    seg_mean_iou, seg_std_iou = test_segmentation_accuracy(model, dataset, device)
    print(f"Segmentation Mean IoU: {seg_mean_iou:.4f} ± {seg_std_iou:.4f}")
    
    # Visualize counterfactual predictions
    for i in range(5):
        fig = visualize_counterfactual_predictions(model, dataset, device, idx=i)
        fig.savefig(os.path.join(args.output_dir, f'counterfactual_predictions_{i}.png'))
        plt.close(fig)
    
    # Visualize a few examples of flow extraction
    for i in range(5):
        sequence = dataset[i].to(device)
        frame1 = sequence[0].unsqueeze(0)
        frame2 = sequence[1].unsqueeze(0)
        
        # Get ground truth flow
        gt_flow = dataset.get_flow_ground_truth(i)[0].unsqueeze(0).to(device)
        
        # Extract flow using the model
        with torch.no_grad():
            pred_flow, occlusion_map = model.compute_optical_flow(frame1, frame2)
        
        # Visualize
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Show frames
        axes[0].imshow(frame1[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Frame 1")
        axes[0].axis('off')
        
        axes[1].imshow(frame2[0].permute(1, 2, 0).cpu().numpy())
        axes[1].set_title("Frame 2")
        axes[1].axis('off')
        
        # Visualize flows using HSV color coding
        def flow_to_rgb(flow):
            flow_x = flow[0, 0].cpu().numpy()
            flow_y = flow[0, 1].cpu().numpy()
            
            # Calculate magnitude and direction
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            direction = np.arctan2(flow_y, flow_x)
            
            # Create HSV image
            hsv = np.zeros((dataset.img_size, dataset.img_size, 3), dtype=np.float32)
            hsv[..., 0] = (direction + np.pi) / (2 * np.pi)  # Hue (direction)
            hsv[..., 1] = np.minimum(1.0, magnitude / dataset.max_speed)  # Saturation (magnitude)
            hsv[..., 2] = (magnitude > 0).astype(np.float32)  # Value (mask)
            
            return hsv_to_rgb(hsv)
        
        # Show ground truth and predicted flow
        axes[2].imshow(flow_to_rgb(gt_flow))
        axes[2].set_title("Ground Truth Flow")
        axes[2].axis('off')
        
        axes[3].imshow(flow_to_rgb(pred_flow))
        axes[3].set_title("Predicted Flow")
        axes[3].axis('off')
        
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'flow_extraction_{i}.png'))
        plt.close(fig)
    
    # Visualize segmentation examples
    for i in range(5):
        sequence = dataset[i].to(device)
        frame = sequence[0].unsqueeze(0)
        
        # Get ground truth segmentation
        gt_segments = dataset.get_segmentation_ground_truth(i).to(device)
        gt_combined = torch.zeros_like(gt_segments[0])
        for s in range(dataset.num_sprites):
            gt_combined = torch.maximum(gt_combined, gt_segments[s] * (s + 1))
        
        # Extract segments using the model
        with torch.no_grad():
            pred_segments = model.extract_segments(frame)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Show frame
        axes[0].imshow(frame[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Input Frame")
        axes[0].axis('off')
        
        # Show ground truth and predicted segmentation
        axes[1].imshow(gt_combined.cpu().numpy(), cmap='tab10')
        axes[1].set_title("Ground Truth Segments")
        axes[1].axis('off')
        
        axes[2].imshow(pred_segments[0].cpu().numpy(), cmap='gray')
        axes[2].set_title("Predicted Segments")
        axes[2].axis('off')
        
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'segmentation_{i}.png'))
        plt.close(fig)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()