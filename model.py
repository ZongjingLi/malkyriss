import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class Patchify(nn.Module):
    """Convert a set of images or a movie into patch vectors"""
    def __init__(self,
                 patch_size=(16,16),
                 temporal_dim=1,
                 squeeze_channel_dim=True
    ):
        super().__init__()
        self.set_patch_size(patch_size)
        self.temporal_dim = temporal_dim
        assert self.temporal_dim in [1,2], self.temporal_dim
        self._squeeze_channel_dim = squeeze_channel_dim

    @property
    def num_patches(self):
        if (self.T is None) or (self.H is None) or (self.W is None):
            return None
        else:
            return (self.T // self.pt) * (self.H // self.ph) * (self.W // self.pw)
        
    def set_patch_size(self, patch_size):
        self.patch_size = patch_size
        if len(self.patch_size) == 2:
            self.ph, self.pw = self.patch_size
            self.pt = 1
            self._patches_are_3d = False
        elif len(self.patch_size) == 3:
            self.pt, self.ph, self.pw = self.patch_size
            self._patches_are_3d = True
        else:
            raise ValueError("patch_size must be a 2- or 3-tuple, but is %s" % self.patch_size)

        self.shape_inp = self.rank_inp = self.H = self.W = self.T = None
        self.D = self.C = self.E = self.embed_dim = None

    def _check_shape(self, x):
        self.shape_inp = x.shape
        self.rank_inp = len(self.shape_inp)
        self.H, self.W = self.shape_inp[-2:]
        assert (self.H % self.ph) == 0 and (self.W % self.pw) == 0, (self.shape_inp, self.patch_size)
        if (self.rank_inp == 5) and self._patches_are_3d:
            self.T = self.shape_inp[self.temporal_dim]
            assert (self.T % self.pt) == 0, (self.T, self.pt)
        elif self.rank_inp == 5:
            self.T = self.shape_inp[self.temporal_dim]
        else:
            self.T = 1

    def split_by_time(self, x):
        shape = x.shape
        assert shape[1] % self.T == 0, (shape, self.T)
        return x.view(shape[0], self.T, shape[1] // self.T, *shape[2:])

    def merge_by_time(self, x):
        shape = x.shape
        return x.view(shape[0], shape[1]*shape[2], *shape[3:])

    def video_to_patches(self, x):
        if self.rank_inp == 4:
            assert self.pt == 1, (self.pt, x.shape)
            x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw) c', ph=self.ph, pw=self.pw)
        else:
            assert self.rank_inp == 5, (x.shape, self.rank_inp, self.shape_inp)
            dim_order = 'b (t pt) c (h ph) (w pw)' if self.temporal_dim == 1 else 'b c (t pt) (h ph) (w pw)'
            x = rearrange(x, dim_order + ' -> b (t h w) (pt ph pw) c', pt=self.pt, ph=self.ph, pw=self.pw)

        self.N, self.D, self.C = x.shape[-3:]
        self.embed_dim = self.E = self.D * self.C
        return x

    def patches_to_video(self, x, mask_mode='zeros'):
        shape = x.shape
        rank = len(shape)
        if rank == 4:
            B,_N,_D,_C = shape
        else:
            assert rank == 3, rank
            B,_N,_E = shape
            assert (_E % self.D == 0), (_E, self.D)
            x = x.view(B,_N,self.D,-1)

        if _N < self.num_patches:
            masked_patches = self.get_masked_patches(
                x,
                num_patches=(self.num_patches - _N),
                mask_mode=mask_mode)
            x = torch.cat([x, masked_patches], 1)

        x = rearrange(
            x,
            'b (t h w) (pt ph pw) c -> b c (t pt) (h ph) (w pw)',
            pt=self.pt, ph=self.ph, pw=self.pw,
            t=(self.T // self.pt), h=(self.H // self.ph), w=(self.W // self.pw))

        if self.rank_inp == 5 and (self.temporal_dim == 1):
            x = x.transpose(1, 2)
        elif self.rank_inp == 4:
            assert x.shape[2] == 1, x.shape
            x = x[:,:,0]
        return x

    @staticmethod
    def get_masked_patches(x, num_patches, mask_mode='zeros'):
        shape = x.shape
        patches_shape = (shape[0], num_patches, *shape[2:])
        if mask_mode == 'zeros':
            return torch.zeros(patches_shape).to(x.device).to(x.dtype).detach()
        elif mask_mode == 'gray':
            return 0.5 * torch.ones(patches_shape).to(x.device).to(x.dtype).detach()
        else:
            raise NotImplementedError("Haven't implemented mask_mode == %s" % mask_mode)

    def forward(self, x, to_video=False, mask_mode='zeros'):
        if not to_video:
            self._check_shape(x)
            x = self.video_to_patches(x)
            return x if not self._squeeze_channel_dim else x.view(x.size(0), self.N, -1)

        else: # x are patches
            assert (self.shape_inp is not None) and (self.num_patches is not None)
            x = self.patches_to_video(x, mask_mode=mask_mode)
            return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CounterfactualWorldModel(nn.Module):
    def __init__(self, 
                 img_size=64, 
                 patch_size=(1, 8, 8), 
                 in_channels=3,
                 embed_dim=512, 
                 depth=6, 
                 num_heads=8,
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Patch embedding
        self.patchify = Patchify(patch_size=patch_size)
        self.patch_embedding = nn.Linear(np.prod(patch_size) * in_channels, embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = PositionalEncoding(embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder (simple MLP to reconstruct patches)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, np.prod(patch_size) * in_channels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.mask_token, std=0.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_encoder(self, frames, mask_ratios):
        """
        Forward pass of encoder with masking
        frames: list of frame tensors [batch_size, in_channels, img_size, img_size]
        mask_ratios: list of ratios per frame, e.g. [0.0, 0.99] for temporally-factored masking
        """
        batch_size = frames[0].shape[0]
        device = frames[0].device
        
        all_embeddings = []
        all_masks = []
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Get patches
            patches = self.patchify(frame)  # [batch_size, num_patches, patch_dim]
            num_patches = patches.shape[1]
                
            # Embed patches
            embeddings = self.patch_embedding(patches)  # [batch_size, num_patches, embed_dim]
            
            # Create mask - temporally factored: fully visible in frame 0, almost all masked in frame 1
            mask_ratio = mask_ratios[i]
            num_masked = int(num_patches * mask_ratio)
            
            # Create mask tensor
            mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
            
            # Only create mask indices if there are tokens to mask
            if num_masked > 0:
                # Create random mask
                noise = torch.rand(batch_size, num_patches, device=device)
                mask_indices = torch.argsort(noise, dim=1)[:, :num_masked]
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_masked)
                mask[batch_indices, mask_indices] = True
            
            all_masks.append(mask)
            
            # Apply mask - replace masked tokens with mask token
            masked_embeddings = embeddings.clone()
            
            # Only apply masking if there are tokens to mask

            if mask.sum() > 0:
                # The mask_token is [1, 1, embed_dim], we need to reshape it to [num_masked, embed_dim]
                mask_tokens = self.mask_token.squeeze(0).repeat(mask.sum(), 1)
                masked_embeddings[mask] = mask_tokens
            all_embeddings.append(masked_embeddings)
        
        # Concatenate frame embeddings
        combined_embedding = torch.cat(all_embeddings, dim=1)
        
        # Add positional encoding
        combined_embedding = self.pos_embed(combined_embedding)
        
        # Pass through transformer
        features = self.transformer(combined_embedding)
        
        return features, all_masks
    
    def forward_decoder(self, features, all_masks):
        """
        Decode features and return reconstructed patches
        """
        # Get number of patches per frame
        num_frames = len(all_masks)
        num_patches = all_masks[0].shape[1]
        
        reconstructed_frames = []
        
        # Process each frame's features
        for i in range(num_frames):
            start_idx = i * num_patches
            end_idx = start_idx + num_patches
            frame_features = features[:, start_idx:end_idx]
            
            # Apply decoder to get reconstructed patches
            reconstructed_patches = self.decoder(frame_features)
            reconstructed_frames.append(reconstructed_patches)
        
        return reconstructed_frames
    
    def forward(self, frames, mask_ratios):
        """
        Full forward pass
        frames: list of frame tensors [batch_size, in_channels, img_size, img_size]
        mask_ratios: list of ratios per frame, e.g. [0.0, 0.99] for temporally-factored masking
        """
        features, all_masks = self.forward_encoder(frames, mask_ratios)
        reconstructed_patches = self.forward_decoder(features, all_masks)
        
        # Compute loss only on the second frame and only on masked patches
        loss = 0
        for i in range(1, len(frames)):  # Start from the second frame
            frame_idx = i
            orig_patches = self.patchify(frames[frame_idx])
            recon_patches = reconstructed_patches[frame_idx]
            mask = all_masks[frame_idx]
            
            # Calculate loss on masked patches
            loss_i = F.mse_loss(recon_patches[mask], orig_patches[mask])
            loss += loss_i
            
        return {
            'loss': loss,
            'features': features,
            'reconstructed_patches': reconstructed_patches,
            'masks': all_masks
        }
    
    def predict(self, first_frame, visible_patches_of_second_frame=None, patch_indices=None):
        """
        Make a prediction of the second frame given the first frame and optionally some patches of the second frame.
        
        Args:
            first_frame: Tensor [batch_size, channels, height, width]
            visible_patches_of_second_frame: Tensor [batch_size, num_patches, patch_dim] or None
            patch_indices: List of indices for the visible patches or None
            
        Returns:
            Predicted second frame
        """
        batch_size = first_frame.shape[0]
        device = first_frame.device
        
        # Generate patches for first frame
        first_frame_patches = self.patchify(first_frame)
        num_patches = first_frame_patches.shape[1]
        
        # Embed first frame patches (fully visible)
        first_frame_embeddings = self.patch_embedding(first_frame_patches)
        
        # Create second frame with all masked patches
        second_frame_embeddings = self.mask_token.expand(batch_size, num_patches, -1)
        
        # If we have visible patches for the second frame, add them
        if visible_patches_of_second_frame is not None and patch_indices is not None:
            visible_embeddings = self.patch_embedding(visible_patches_of_second_frame)
            for b in range(batch_size):
                for i, idx in enumerate(patch_indices[b]):
                    second_frame_embeddings[b, idx] = visible_embeddings[b, i]
        
        # Combine both frame embeddings
        combined_embedding = torch.cat([first_frame_embeddings, second_frame_embeddings], dim=1)
        
        # Add positional encoding
        combined_embedding = self.pos_embed(combined_embedding)
        
        # Pass through transformer
        features = self.transformer(combined_embedding)
        
        # Get the features for the second frame
        second_frame_features = features[:, num_patches:2*num_patches]
        
        # Decode to get reconstructed patches
        reconstructed_patches = self.decoder(second_frame_features)
        
        # Convert patches back to an image
        # Store original shape to reconstruct later
        self.patchify._check_shape(first_frame)
        reconstructed_image = self.patchify(reconstructed_patches, to_video=True)
        
        return reconstructed_image
    
    def compute_optical_flow(self, frame1, frame2, patch_size=(1,8,8), delta_magnitude=0.1):
        """
        Compute optical flow through counterfactual prompting as described in the paper.
        
        Args:
            frame1: First frame [batch_size, channels, height, width]
            frame2: Second frame [batch_size, channels, height, width]
            patch_size: Size of patches
            delta_magnitude: Magnitude of perturbation for finite differences
            
        Returns:
            Optical flow field and occlusion map
        """
        batch_size, channels, height, width = frame1.shape
        device = frame1.device
        
        # Patchify frames
        patchify = Patchify(patch_size=patch_size)
        patchify._check_shape(frame1)
        
        # Number of patches in each dimension
        h_patches = height // patch_size[1]
        w_patches = width // patch_size[2]
        
        # Get a few visible patches from the second frame (about 1% as per the paper)
        frame2_patches = patchify(frame2)
        num_patches = frame2_patches.shape[1]
        num_visible_patches = max(1, int(0.01 * num_patches))
        
        # Randomly select patches
        visible_indices = []
        for b in range(batch_size):
            indices = torch.randperm(num_patches, device=device)[:num_visible_patches]
            visible_indices.append(indices)
            
        visible_patches = []
        for b in range(batch_size):
            visible_patches.append(frame2_patches[b, visible_indices[b]])
        visible_patches = torch.stack(visible_patches)
        
        # Base prediction without perturbation
        base_prediction = self.predict(
            frame1, 
            visible_patches_of_second_frame=visible_patches, 
            patch_indices=visible_indices
        )
        
        # Initialize flow and occlusion maps
        flow_map = torch.zeros(batch_size, 2, height, width, device=device)
        occlusion_map = torch.zeros(batch_size, 1, height, width, device=device)
        
        # For each patch in frame1, add a small perturbation and see where it moves in the prediction
        for h in range(h_patches):
            for w in range(w_patches):
                # Create perturbation
                perturbed_frame1 = frame1.clone()
                
                # Add a small delta to the patch
                patch_h_start = h * patch_size[1]
                patch_h_end = (h + 1) * patch_size[1]
                patch_w_start = w * patch_size[2]
                patch_w_end = (w + 1) * patch_size[2]
                
                # Add perturbation for each color channel to make it distinctive
                perturbed_frame1[:, 0, patch_h_start:patch_h_end, patch_w_start:patch_w_end] += delta_magnitude
                perturbed_frame1[:, 1, patch_h_start:patch_h_end, patch_w_start:patch_w_end] -= delta_magnitude
                perturbed_frame1[:, 2, patch_h_start:patch_h_end, patch_w_start:patch_w_end] += delta_magnitude
                
                # Get prediction with perturbed input
                perturbed_prediction = self.predict(
                    perturbed_frame1, 
                    visible_patches_of_second_frame=visible_patches, 
                    patch_indices=visible_indices
                )
                
                # Calculate perturbation response 
                perturbation_response = perturbed_prediction - base_prediction
                
                # Calculate the overall magnitude of the perturbation response 
                response_magnitude = torch.sum(torch.abs(perturbation_response), dim=1, keepdim=True)
                
                # Find the location with maximum response (where the perturbation moved to)
                # For each batch element
                for b in range(batch_size):
                    # Flatten the response to find the max index
                    flat_response = response_magnitude[b, 0].view(-1)
                    max_response_idx = torch.argmax(flat_response)
                    
                    # Check if the response is significant
                    if flat_response[max_response_idx] < 0.1 * delta_magnitude:
                        # This patch is likely occluded in the second frame
                        occlusion_map[b, 0, patch_h_start:patch_h_end, patch_w_start:patch_w_end] = 1.0
                        continue
                    
                    # Convert flat index to 2D coordinates
                    max_h_idx = max_response_idx // width
                    max_w_idx = max_response_idx % width
                    
                    # Calculate displacement
                    center_h = patch_h_start + patch_size[1] // 2
                    center_w = patch_w_start + patch_size[2] // 2
                    
                    # Flow is the displacement from the center of the patch to where it moved
                    flow_h = max_h_idx.item() - center_h
                    flow_w = max_w_idx.item() - center_w
                    
                    # Set the flow values for the entire patch
                    flow_map[b, 0, patch_h_start:patch_h_end, patch_w_start:patch_w_end] = flow_h
                    flow_map[b, 1, patch_h_start:patch_h_end, patch_w_start:patch_w_end] = flow_w
        
        return flow_map, occlusion_map
    def extract_segments(self, frame, num_motion_samples=5, flow_threshold=2.0):
        """
        Extract movable object segments from a single frame using counterfactual motion as per the paper.
        
        Args:
            frame: Single frame [batch_size, channels, height, width]
            num_motion_samples: Number of different motion samples to try
            flow_threshold: Threshold for determining connected components
            
        Returns:
            Segmentation masks for movable objects
        """
        batch_size, channels, height, width = frame.shape
        device = frame.device
        
        # Patchify frame
        patchify = Patchify(patch_size=(1,8,8))
        patchify._check_shape(frame)
        
        # Number of patches in each dimension
        h_patches = height // patchify.ph
        w_patches = width // patchify.pw
        num_patches = h_patches * w_patches
        
        # Initialize segmentation masks
        segment_masks = torch.zeros(batch_size, height, width, device=device)
        flow_correlation = torch.zeros(batch_size, num_patches, num_patches, device=device)
        
        # For each motion sample, choose a random patch and apply motion
        for sample in range(num_motion_samples):
            # Choose a random patch to move
            patch_idx = torch.randint(0, num_patches, (batch_size,), device=device)
            
            # Calculate patch coordinates
            patch_h = patch_idx // w_patches
            patch_w = patch_idx % w_patches
            
            # Create a counterfactual second frame where the patch moves
            counterfactual_frame = torch.zeros_like(frame)
            
            # Direction of motion (randomly chosen)
            direction_h = torch.randint(-2, 3, (batch_size,), device=device)
            direction_w = torch.randint(-2, 3, (batch_size,), device=device)
            
            # Apply motion to the chosen patch
            for b in range(batch_size):
                h, w = patch_h[b], patch_w[b]
                h_start = h * patchify.ph
                h_end = (h + 1) * patchify.ph
                w_start = w * patchify.pw
                w_end = (w + 1) * patchify.pw
                
                # Move patch to new location (with boundary checking)
                new_h = torch.clamp(h + direction_h[b], 0, h_patches - 1)
                new_w = torch.clamp(w + direction_w[b], 0, w_patches - 1)
                
                new_h_start = new_h * patchify.ph
                new_h_end = (new_h + 1) * patchify.ph
                new_w_start = new_w * patchify.pw
                new_w_end = (new_w + 1) * patchify.pw
                
                # Copy patch to new location in counterfactual frame
                counterfactual_frame[b, :, new_h_start:new_h_end, new_w_start:new_w_end] = \
                    frame[b, :, h_start:h_end, w_start:w_end]
            
            # Use a simpler approach for visible patches - just use the moved patch
            visible_indices = []
            batch_visible_patches = []
            
            for b in range(batch_size):
                # Use the new patch location as the visible patch
                new_h = torch.clamp(patch_h[b] + direction_h[b], 0, h_patches - 1)
                new_w = torch.clamp(patch_w[b] + direction_w[b], 0, w_patches - 1)
                patch_index = new_h * w_patches + new_w
                
                # Get the content of the patch
                h_start = patch_h[b] * patchify.ph
                h_end = (patch_h[b] + 1) * patchify.ph
                w_start = patch_w[b] * patchify.pw
                w_end = (patch_w[b] + 1) * patchify.pw
                
                patch_content = frame[b, :, h_start:h_end, w_start:w_end].reshape(-1)
                
                visible_indices.append([patch_index.item()])
                batch_visible_patches.append(patch_content.unsqueeze(0))
                
            # Stack patches for batch processing
            batch_visible_patches = torch.stack(batch_visible_patches)
            
            # Predict what would happen if we moved just that patch
            try:
                counterfactual_prediction = self.predict(
                    frame,
                    batch_visible_patches,
                    visible_indices
                )
                
                # Compute counterfactual flow
                flow, _ = self.compute_optical_flow(frame, counterfactual_prediction)
                
                # Compute flow magnitude
                flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
                
                # Update correlation matrix - which patches move together
                for b in range(batch_size):
                    # Convert flow magnitude to patches
                    # Create an empty tensor to hold the patchified flow magnitude
                    flow_magnitude_patches = torch.zeros(num_patches, device=device)
                    
                    # Loop through each patch and calculate the average flow magnitude
                    for h in range(h_patches):
                        for w in range(w_patches):
                            h_start = h * patchify.ph
                            h_end = (h + 1) * patchify.ph
                            w_start = w * patchify.pw
                            w_end = (w + 1) * patchify.pw
                            
                            # Calculate average flow magnitude in this patch
                            patch_flow = flow_magnitude[b, h_start:h_end, w_start:w_end]
                            avg_flow = torch.mean(patch_flow)
                            
                            # Store in the patch tensor
                            patch_idx_curr = h * w_patches + w
                            flow_magnitude_patches[patch_idx_curr] = avg_flow
                    
                    # Identify moving patches (flow > threshold)
                    moving_patches = (flow_magnitude_patches > flow_threshold)
                    
                    # Update correlation matrix
                    for i in range(num_patches):
                        if moving_patches[i].item():  # Convert tensor to scalar boolean
                            flow_correlation[b, patch_idx[b], i] += 1
                            flow_correlation[b, i, patch_idx[b]] += 1
                
            except Exception as e:
                print(f"Error in sample {sample}: {str(e)}")
                continue
        
        # Normalize correlation
        max_counts = torch.max(flow_correlation, dim=2, keepdim=True)[0]
        max_counts = torch.clamp(max_counts, min=1)  # Avoid division by zero
        flow_correlation = flow_correlation / max_counts
        
        # Threshold correlation to get segments
        threshold = 0.7  # Correlation threshold for considering patches part of the same object
        
        # For each batch element
        for b in range(batch_size):
            # Start with largest segment
            while True:
                # Find patch with most correlations
                corr_sum = torch.sum(flow_correlation[b] > threshold, dim=1)
                if torch.max(corr_sum) <= 1:  # No more segments
                    break
                    
                seed_idx = torch.argmax(corr_sum)
                
                # Get all patches correlated with this patch
                correlated_patches = flow_correlation[b, seed_idx] > threshold
                
                # Convert to patch coordinates
                for patch_idx in torch.nonzero(correlated_patches):
                    p_idx = patch_idx.item()
                    p_h = p_idx // w_patches
                    p_w = p_idx % w_patches
                    
                    h_start = p_h * patchify.ph
                    h_end = (p_h + 1) * patchify.ph
                    w_start = p_w * patchify.pw
                    w_end = (p_w + 1) * patchify.pw
                    
                    # Add to segment mask
                    segment_masks[b, h_start:h_end, w_start:w_end] = 1.0
                
                # Remove these patches from correlation matrix to find next segment
                flow_correlation[b, correlated_patches, :] = 0
                flow_correlation[b, :, correlated_patches] = 0
        
        return segment_masks