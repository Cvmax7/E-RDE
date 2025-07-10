import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplementaryContrastiveLearning(nn.Module):
    """
    Complementary Contrastive Learning inspired by RCL
    Implements "push-first, pull-later" progressive training strategy
    """
    def __init__(self, temperature=0.07, push_only_epochs=10):
        super().__init__()
        self.temperature = temperature
        self.push_only_epochs = push_only_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Set current training epoch for progressive strategy"""
        self.current_epoch = epoch
        
    def compute_negative_loss(self, similarities, negative_mask):
        """
        Compute loss using only negative samples (complementary labels)
        
        Args:
            similarities: [B, B] - similarity matrix
            negative_mask: [B, B] - mask where 1 indicates negative pairs
            
        Returns:
            neg_loss: scalar - complementary negative loss
        """
        batch_size = similarities.size(0)
        
        # Apply temperature scaling
        logits = similarities / self.temperature
        
        # Create negative sample distribution
        neg_exp = torch.exp(logits) * negative_mask
        neg_sum = torch.sum(neg_exp, dim=1, keepdim=True) + 1e-8
        
        # Complementary loss: encourage dissimilarity to negatives
        # Use log of negative probabilities
        neg_loss = torch.log(neg_sum)
        
        return neg_loss.mean()
        
    def compute_brownian_negative_loss(self, visual_feats, textual_feats, negative_mask):
        """
        Brownian motion inspired negative loss using collective force of negatives
        
        Args:
            visual_feats: [B, D] - visual features  
            textual_feats: [B, D] - textual features
            negative_mask: [B, B] - negative pair mask
            
        Returns:
            brownian_loss: scalar - brownian motion inspired loss
        """
        batch_size = visual_feats.size(0)
        
        # Normalize features
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        textual_norm = F.normalize(textual_feats, p=2, dim=-1)
        
        # Compute all similarities
        similarities = visual_norm @ textual_norm.t()
        
        # Apply temperature
        scaled_sims = similarities / self.temperature
        
        # Create negative forces (push away from all negatives)
        negative_forces = torch.exp(scaled_sims) * negative_mask
        
        # Aggregate negative forces (Brownian motion collective effect)
        total_negative_force = torch.sum(negative_forces, dim=1)
        
        # Loss: minimize total negative attraction
        brownian_loss = torch.log(total_negative_force + 1e-8).mean()
        
        return brownian_loss
        
    def forward(self, visual_feats, textual_feats, confident_clean_mask, confident_noisy_mask):
        """
        Complementary contrastive learning forward pass
        
        Args:
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features  
            confident_clean_mask: [B] - mask for confident clean samples
            confident_noisy_mask: [B] - mask for confident noisy samples
            
        Returns:
            result: dict containing CCL losses and info
        """
        batch_size = visual_feats.size(0)
        device = visual_feats.device
        
        # Create negative pair mask (confident noisy samples)
        negative_mask = torch.zeros(batch_size, batch_size, device=device)
        
        # Mark confident noisy pairs as negatives
        for i in range(batch_size):
            if confident_noisy_mask[i]:
                negative_mask[i, i] = 1.0  # Self as negative
                
        # Cross-negative mask for different samples
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and (confident_noisy_mask[i] or confident_noisy_mask[j]):
                    negative_mask[i, j] = 1.0
                    
        # Compute similarities
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        textual_norm = F.normalize(textual_feats, p=2, dim=-1)
        similarities = visual_norm @ textual_norm.t()
        
        # Complementary negative loss
        comp_neg_loss = self.compute_negative_loss(similarities, negative_mask)
        
        # Brownian motion inspired loss
        brownian_loss = self.compute_brownian_negative_loss(visual_feats, textual_feats, negative_mask)
        
        result = {
            'complementary_loss': comp_neg_loss,
            'brownian_loss': brownian_loss,
            'negative_mask': negative_mask,
            'similarities': similarities
        }
        
        return result


class ProgressiveContrastiveLearning(nn.Module):
    """
    Progressive "push-first, pull-later" contrastive learning strategy
    """
    def __init__(self, temperature=0.07, push_phase_epochs=15, transition_epochs=5):
        super().__init__()
        self.temperature = temperature
        self.push_phase_epochs = push_phase_epochs
        self.transition_epochs = transition_epochs
        self.current_epoch = 0
        
        # Sub-modules
        self.ccl = ComplementaryContrastiveLearning(temperature, push_phase_epochs)
        
    def set_epoch(self, epoch):
        """Set current epoch for both modules"""
        self.current_epoch = epoch
        self.ccl.set_epoch(epoch)
        
    def get_training_phase(self):
        """
        Determine current training phase
        
        Returns:
            phase: str - 'push_only', 'transition', or 'push_pull'
        """
        if self.current_epoch < self.push_phase_epochs:
            return 'push_only'
        elif self.current_epoch < self.push_phase_epochs + self.transition_epochs:
            return 'transition'
        else:
            return 'push_pull'
            
    def compute_positive_pull_loss(self, visual_feats, textual_feats, transport_matrix, 
                                   confident_clean_mask):
        """
        Compute positive pulling loss for clean samples and rematch results
        
        Args:
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features
            transport_matrix: [B, B] - rematch transport matrix
            confident_clean_mask: [B] - mask for confident clean samples
            
        Returns:
            pull_loss: scalar - positive pulling loss
        """
        # Normalize features
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        textual_norm = F.normalize(textual_feats, p=2, dim=-1)
        
        # Compute similarities
        similarities = visual_norm @ textual_norm.t()
        scaled_sims = similarities / self.temperature
        
        # Positive mask: confident clean + rematch positives
        positive_mask = torch.eye(similarities.size(0), device=similarities.device)
        
        # Add confident clean diagonal elements
        for i in range(similarities.size(0)):
            if confident_clean_mask[i]:
                positive_mask[i, i] = 1.0
                
        # Add rematch positives (where transport > threshold)
        rematch_threshold = 0.1
        rematch_positives = (transport_matrix > rematch_threshold).float()
        positive_mask = positive_mask + rematch_positives
        positive_mask = torch.clamp(positive_mask, 0, 1)
        
        # Positive pulling: maximize similarity for positive pairs
        positive_sims = scaled_sims * positive_mask
        pull_loss = -torch.sum(positive_sims) / (positive_mask.sum() + 1e-8)
        
        return pull_loss
        
    def compute_transition_loss(self, visual_feats, textual_feats, transport_matrix,
                               confident_clean_mask, confident_noisy_mask):
        """
        Compute gradual transition loss that balances push and pull
        
        Args:
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features  
            transport_matrix: [B, B] - rematch results
            confident_clean_mask: [B] - confident clean mask
            confident_noisy_mask: [B] - confident noisy mask
            
        Returns:
            transition_loss: scalar - balanced transition loss
        """
        # Get CCL loss (push)
        ccl_result = self.ccl(visual_feats, textual_feats, confident_clean_mask, confident_noisy_mask)
        push_loss = ccl_result['complementary_loss']
        
        # Get positive pull loss 
        pull_loss = self.compute_positive_pull_loss(visual_feats, textual_feats, 
                                                   transport_matrix, confident_clean_mask)
        
        # Gradual transition weight
        transition_progress = (self.current_epoch - self.push_phase_epochs) / self.transition_epochs
        transition_progress = max(0, min(1, transition_progress))
        
        # Weighted combination
        transition_loss = (1 - transition_progress) * push_loss + transition_progress * pull_loss
        
        return transition_loss, {'push_loss': push_loss, 'pull_loss': pull_loss, 'transition_weight': transition_progress}
        
    def forward(self, visual_feats, textual_feats, transport_matrix, 
                confident_clean_mask, confident_noisy_mask):
        """
        Progressive contrastive learning forward pass
        
        Args:
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features
            transport_matrix: [B, B] - rematch transport matrix  
            confident_clean_mask: [B] - confident clean samples
            confident_noisy_mask: [B] - confident noisy samples
            
        Returns:
            result: dict containing progressive learning results
        """
        phase = self.get_training_phase()
        
        if phase == 'push_only':
            # Only push negatives away
            ccl_result = self.ccl(visual_feats, textual_feats, confident_clean_mask, confident_noisy_mask)
            
            result = {
                'progressive_loss': ccl_result['complementary_loss'],
                'phase': phase,
                'ccl_result': ccl_result
            }
            
        elif phase == 'transition':
            # Gradual transition from push to pull
            transition_loss, transition_info = self.compute_transition_loss(
                visual_feats, textual_feats, transport_matrix, confident_clean_mask, confident_noisy_mask)
            
            result = {
                'progressive_loss': transition_loss,
                'phase': phase,
                'transition_info': transition_info
            }
            
        else:  # push_pull
            # Full push and pull
            ccl_result = self.ccl(visual_feats, textual_feats, confident_clean_mask, confident_noisy_mask)
            pull_loss = self.compute_positive_pull_loss(visual_feats, textual_feats, 
                                                       transport_matrix, confident_clean_mask)
            
            # Balanced combination
            push_weight = 0.3
            pull_weight = 0.7
            combined_loss = push_weight * ccl_result['complementary_loss'] + pull_weight * pull_loss
            
            result = {
                'progressive_loss': combined_loss,
                'phase': phase,
                'ccl_result': ccl_result,
                'pull_loss': pull_loss,
                'push_weight': push_weight,
                'pull_weight': pull_weight
            }
            
        return result


class AdaptiveContrastiveWeighting(nn.Module):
    """
    Adaptive weighting for contrastive losses based on sample uncertainty
    """
    def __init__(self, min_weight=0.1, max_weight=1.0):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def compute_uncertainty_weights(self, uncertainty_scores, sample_types):
        """
        Compute adaptive weights based on uncertainty scores
        
        Args:
            uncertainty_scores: [B] - uncertainty scores
            sample_types: [B] - sample types (0: clean, 1: noisy, 2: uncertain)
            
        Returns:
            weights: [B] - adaptive weights for loss computation
        """
        weights = torch.ones_like(uncertainty_scores)
        
        # For uncertain samples: higher uncertainty -> lower weight initially
        uncertain_mask = (sample_types == 2)
        if uncertain_mask.any():
            uncertain_weights = self.max_weight - (self.max_weight - self.min_weight) * uncertainty_scores[uncertain_mask]
            weights[uncertain_mask] = uncertain_weights
            
        # For confident samples: full weight
        confident_mask = (sample_types != 2)
        weights[confident_mask] = self.max_weight
        
        return weights
        
    def forward(self, losses, uncertainty_scores, sample_types):
        """
        Apply adaptive weighting to losses
        
        Args:
            losses: [B] - per-sample losses
            uncertainty_scores: [B] - uncertainty scores
            sample_types: [B] - sample types
            
        Returns:
            weighted_loss: scalar - weighted average loss
        """
        weights = self.compute_uncertainty_weights(uncertainty_scores, sample_types)
        weighted_losses = losses * weights
        
        # Normalize by total weight
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_loss = weighted_losses.sum() / total_weight
        else:
            weighted_loss = losses.mean()
            
        return weighted_loss


class ComplementaryContrastiveModule(nn.Module):
    """
    Complete complementary contrastive learning module
    """
    def __init__(self, temperature=0.07, push_phase_epochs=15, transition_epochs=5):
        super().__init__()
        
        self.progressive_cl = ProgressiveContrastiveLearning(temperature, push_phase_epochs, transition_epochs)
        self.adaptive_weighting = AdaptiveContrastiveWeighting()
        
    def set_epoch(self, epoch):
        """Set current epoch"""
        self.progressive_cl.set_epoch(epoch)
        
    def forward(self, visual_feats, textual_feats, transport_matrix,
                confident_clean_mask, confident_noisy_mask, uncertain_mask,
                uncertainty_scores, sample_types):
        """
        Complete complementary contrastive learning forward pass
        
        Args:
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features
            transport_matrix: [B, B] - rematch transport matrix
            confident_clean_mask: [B] - confident clean samples
            confident_noisy_mask: [B] - confident noisy samples  
            uncertain_mask: [B] - uncertain samples
            uncertainty_scores: [B] - uncertainty scores
            sample_types: [B] - sample types
            
        Returns:
            result: dict containing all complementary contrastive results
        """
        # Progressive contrastive learning
        prog_result = self.progressive_cl(
            visual_feats, textual_feats, transport_matrix,
            confident_clean_mask, confident_noisy_mask
        )
        
        # Get progressive loss
        progressive_loss = prog_result['progressive_loss']
        
        # Apply adaptive weighting if we have per-sample losses
        # For now, use the aggregate loss but store weighting capability
        adaptive_weights = self.adaptive_weighting.compute_uncertainty_weights(uncertainty_scores, sample_types)
        
        result = {
            'complementary_loss': progressive_loss,
            'progressive_result': prog_result,
            'adaptive_weights': adaptive_weights,
            'training_phase': prog_result['phase']
        }
        
        return result 