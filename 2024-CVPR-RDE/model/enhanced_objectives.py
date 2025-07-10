import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UncertaintyWeightedTAL(nn.Module):
    """
    Uncertainty-weighted Triplet Alignment Loss with adaptive margins
    """
    def __init__(self, base_tau=0.02, base_margin=0.2):
        super().__init__()
        self.base_tau = base_tau
        self.base_margin = base_margin
        
    def compute_tal_per_sample(self, scores, pid, tau, margin):
        """
        Compute TAL loss per sample
        
        Args:
            scores: [B, B] - similarity matrix
            pid: [B] - person IDs
            tau: float - temperature parameter
            margin: float - margin parameter
            
        Returns:
            per_sample_loss: [B] - TAL loss per sample
        """
        batch_size = scores.shape[0]
        pid = pid.reshape((batch_size, 1))
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float().to(scores.device)
        mask = 1 - labels

        alpha_i2t = ((scores/tau).exp() * labels / ((scores/tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
        alpha_t2i = ((scores.t()/tau).exp() * labels / ((scores.t()/tau).exp() * labels).sum(dim=1, keepdim=True)).detach()

        loss_i2t = (- (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
        loss_t2i = (- (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
        
        return loss_i2t + loss_t2i
        
    def forward(self, visual_feats, textual_feats, pid, adaptive_margins, 
                uncertainty_scores, sample_types):
        """
        Uncertainty-weighted TAL forward pass
        
        Args:
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features
            pid: [B] - person IDs
            adaptive_margins: [B] - adaptive margin values
            uncertainty_scores: [B] - uncertainty scores
            sample_types: [B] - sample types (0: clean, 1: noisy, 2: uncertain)
            
        Returns:
            result: dict containing weighted TAL losses
        """
        # Normalize features
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        textual_norm = F.normalize(textual_feats, p=2, dim=-1)
        scores = textual_norm @ visual_norm.t()
        
        batch_size = scores.size(0)
        total_loss = 0.0
        
        # Compute per-sample losses with adaptive margins
        per_sample_losses = torch.zeros(batch_size, device=scores.device)
        
        for i in range(batch_size):
            margin_i = adaptive_margins[i].item()
            # Extract single sample for TAL computation
            scores_i = scores[i:i+1, :]  # [1, B]
            pid_i = pid[i:i+1]  # [1]
            
            # Compute TAL loss for this sample
            loss_i = self.compute_tal_per_sample(scores_i.expand(batch_size, -1), 
                                               pid.expand(batch_size), 
                                               self.base_tau, margin_i)
            per_sample_losses[i] = loss_i[0]  # Take the first (and only) loss
            
        # Apply uncertainty-based weighting
        uncertainty_weights = torch.ones_like(uncertainty_scores)
        
        # Confident samples get full weight
        confident_mask = (sample_types != 2)
        uncertainty_weights[confident_mask] = 1.0
        
        # Uncertain samples get reduced weight based on uncertainty
        uncertain_mask = (sample_types == 2)
        if uncertain_mask.any():
            # Higher uncertainty -> lower weight
            uncertainty_weights[uncertain_mask] = 1.0 - 0.5 * uncertainty_scores[uncertain_mask]
            uncertainty_weights[uncertain_mask] = torch.clamp(uncertainty_weights[uncertain_mask], 0.1, 1.0)
        
        # Weighted loss
        weighted_losses = per_sample_losses * uncertainty_weights
        final_loss = weighted_losses.sum() / uncertainty_weights.sum()
        
        result = {
            'weighted_tal_loss': final_loss,
            'per_sample_losses': per_sample_losses,
            'uncertainty_weights': uncertainty_weights,
            'similarity_scores': scores
        }
        
        return result


class RematchSupervisionLoss(nn.Module):
    """
    Loss for supervising the model with rematch results
    """
    def __init__(self, temperature=0.07, kl_weight=1.0, alignment_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.alignment_weight = alignment_weight
        
    def symmetric_kl_divergence(self, transport_matrix, similarity_matrix):
        """
        Symmetric KL divergence between transport plan and predictions
        """
        # Convert similarities to probabilities
        logits = similarity_matrix / self.temperature
        pred_probs = F.softmax(logits, dim=1)
        log_pred_probs = F.log_softmax(logits, dim=1)
        
        # Normalize transport matrix as target distribution
        target_probs = transport_matrix / (transport_matrix.sum(dim=1, keepdim=True) + 1e-8)
        
        # Symmetric KL divergence
        kl_1 = F.kl_div(log_pred_probs, target_probs, reduction='none').sum(dim=1)
        kl_2 = F.kl_div(torch.log(target_probs + 1e-8), pred_probs, reduction='none').sum(dim=1)
        
        return (kl_1 + kl_2).mean() / 2.0
        
    def alignment_loss(self, transport_matrix, visual_feats, textual_feats):
        """
        Direct alignment loss based on transport weights
        """
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        textual_norm = F.normalize(textual_feats, p=2, dim=-1)
        similarities = visual_norm @ textual_norm.t()
        
        # Weighted alignment
        weighted_sim = transport_matrix * similarities
        return -weighted_sim.sum() / (transport_matrix.sum() + 1e-8)
        
    def forward(self, transport_matrix, visual_feats, textual_feats, uncertain_mask):
        """
        Compute rematch supervision loss
        
        Args:
            transport_matrix: [B, B] - optimal transport plan
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features
            uncertain_mask: [B] - mask for uncertain samples
            
        Returns:
            result: dict containing rematch supervision losses
        """
        # Only apply to uncertain samples
        if not uncertain_mask.any():
            return {
                'rematch_kl_loss': torch.tensor(0.0, device=visual_feats.device),
                'rematch_alignment_loss': torch.tensor(0.0, device=visual_feats.device),
                'total_rematch_loss': torch.tensor(0.0, device=visual_feats.device)
            }
            
        # Extract uncertain samples
        uncertain_indices = torch.where(uncertain_mask)[0]
        
        if len(uncertain_indices) > 1:
            # Extract submatrices for uncertain samples
            transport_sub = transport_matrix[uncertain_indices][:, uncertain_indices]
            visual_sub = visual_feats[uncertain_indices]
            textual_sub = textual_feats[uncertain_indices]
            
            # Compute similarities for uncertain subset
            visual_norm = F.normalize(visual_sub, p=2, dim=-1)
            textual_norm = F.normalize(textual_sub, p=2, dim=-1)
            sim_sub = visual_norm @ textual_norm.t()
            
            # KL divergence loss
            kl_loss = self.symmetric_kl_divergence(transport_sub, sim_sub)
            
            # Alignment loss
            align_loss = self.alignment_loss(transport_sub, visual_sub, textual_sub)
            
            # Combined loss
            total_loss = self.kl_weight * kl_loss + self.alignment_weight * align_loss
        else:
            kl_loss = torch.tensor(0.0, device=visual_feats.device)
            align_loss = torch.tensor(0.0, device=visual_feats.device)
            total_loss = torch.tensor(0.0, device=visual_feats.device)
            
        result = {
            'rematch_kl_loss': kl_loss,
            'rematch_alignment_loss': align_loss,
            'total_rematch_loss': total_loss
        }
        
        return result


class EvidenceLearningLoss(nn.Module):
    """
    Loss for training the evidence network using Dirichlet distribution
    """
    def __init__(self, concentration_reg=0.1):
        super().__init__()
        self.concentration_reg = concentration_reg
        
    def dirichlet_kl_loss(self, evidence, target_labels):
        """
        KL divergence loss for Dirichlet distribution
        
        Args:
            evidence: [B, 2] - evidence values
            target_labels: [B] - binary target labels (0 or 1)
            
        Returns:
            kl_loss: scalar - KL divergence loss
        """
        # Convert evidence to Dirichlet parameters
        alpha = evidence + 1.0
        alpha_sum = alpha.sum(dim=1, keepdim=True)
        
        # Expected probabilities under Dirichlet
        expected_probs = alpha / alpha_sum
        
        # Target one-hot vectors
        target_one_hot = F.one_hot(target_labels.long(), num_classes=2).float()
        
        # KL divergence between predicted and target distributions
        kl_loss = F.kl_div(torch.log(expected_probs + 1e-8), target_one_hot, reduction='batchmean')
        
        return kl_loss
        
    def concentration_regularization(self, evidence):
        """
        Regularization to encourage appropriate concentration
        """
        alpha = evidence + 1.0
        concentration = alpha.sum(dim=1)
        
        # Penalize extreme concentrations
        reg_loss = F.mse_loss(concentration, torch.ones_like(concentration) * 3.0)
        
        return reg_loss
        
    def forward(self, evidence_bge, evidence_tse, confident_clean_mask, confident_noisy_mask):
        """
        Evidence learning loss computation
        
        Args:
            evidence_bge: [B, 2] - BGE evidence values
            evidence_tse: [B, 2] - TSE evidence values
            confident_clean_mask: [B] - confident clean sample mask
            confident_noisy_mask: [B] - confident noisy sample mask
            
        Returns:
            result: dict containing evidence learning losses
        """
        device = evidence_bge.device
        batch_size = evidence_bge.size(0)
        
        # Create target labels for confident samples
        target_labels = torch.zeros(batch_size, device=device)
        target_labels[confident_clean_mask] = 1  # clean = 1, noisy = 0
        
        # Only compute loss on confident samples
        confident_mask = confident_clean_mask | confident_noisy_mask
        
        if confident_mask.any():
            # Extract confident samples
            evidence_bge_conf = evidence_bge[confident_mask]
            evidence_tse_conf = evidence_tse[confident_mask]
            target_conf = target_labels[confident_mask]
            
            # KL divergence losses
            kl_loss_bge = self.dirichlet_kl_loss(evidence_bge_conf, target_conf)
            kl_loss_tse = self.dirichlet_kl_loss(evidence_tse_conf, target_conf)
            
            # Concentration regularization
            reg_loss_bge = self.concentration_regularization(evidence_bge)
            reg_loss_tse = self.concentration_regularization(evidence_tse)
            
            total_evidence_loss = kl_loss_bge + kl_loss_tse + \
                                self.concentration_reg * (reg_loss_bge + reg_loss_tse)
        else:
            kl_loss_bge = torch.tensor(0.0, device=device)
            kl_loss_tse = torch.tensor(0.0, device=device)
            reg_loss_bge = torch.tensor(0.0, device=device)
            reg_loss_tse = torch.tensor(0.0, device=device)
            total_evidence_loss = torch.tensor(0.0, device=device)
            
        result = {
            'evidence_kl_bge': kl_loss_bge,
            'evidence_kl_tse': kl_loss_tse,
            'evidence_reg_bge': reg_loss_bge,
            'evidence_reg_tse': reg_loss_tse,
            'total_evidence_loss': total_evidence_loss
        }
        
        return result


class EnhancedLossObjectives(nn.Module):
    """
    Enhanced loss objectives integrating all components
    """
    def __init__(self, base_tau=0.02, base_margin=0.2, 
                 rematch_temp=0.07, evidence_reg=0.1):
        super().__init__()
        
        # Component losses
        self.uncertainty_tal = UncertaintyWeightedTAL(base_tau, base_margin)
        self.rematch_supervision = RematchSupervisionLoss(rematch_temp)
        self.evidence_learning = EvidenceLearningLoss(evidence_reg)
        
        # Loss weights
        self.tal_weight = 1.0
        self.rematch_weight = 0.5
        self.evidence_weight = 0.3
        self.complementary_weight = 0.8
        self.cost_weight = 0.2
        
    def forward(self, visual_bge, textual_bge, visual_tse, textual_tse, 
                pid, uncertainty_result, rematch_result, 
                complementary_result, cost_loss=None):
        """
        Complete enhanced loss computation
        
        Args:
            visual_bge: [B, D] - BGE visual features
            textual_bge: [B, D] - BGE textual features
            visual_tse: [B, D] - TSE visual features  
            textual_tse: [B, D] - TSE textual features
            pid: [B] - person IDs
            uncertainty_result: dict - uncertainty analysis results
            rematch_result: dict - optimal transport results
            complementary_result: dict - complementary contrastive results
            cost_loss: scalar - cost network training loss (optional)
            
        Returns:
            result: dict containing all losses and final combined loss
        """
        device = visual_bge.device
        
        # Extract components from results
        division_result = uncertainty_result['division_result']
        adaptive_margins = uncertainty_result['adaptive_margins']
        sample_types = uncertainty_result['sample_types']
        uncertainty_scores = division_result['uncertainty_scores']
        
        confident_clean = division_result['confident_clean']
        confident_noisy = division_result['confident_noisy']
        uncertain = division_result['uncertain']
        
        transport_matrix = rematch_result['transport_matrix']
        evidence_bge = uncertainty_result['evidence_bge']
        evidence_tse = uncertainty_result['evidence_tse']
        
        # 1. Uncertainty-weighted TAL loss for BGE
        tal_bge_result = self.uncertainty_tal(
            visual_bge, textual_bge, pid, adaptive_margins,
            uncertainty_scores, sample_types
        )
        
        # 2. Uncertainty-weighted TAL loss for TSE
        tal_tse_result = self.uncertainty_tal(
            visual_tse, textual_tse, pid, adaptive_margins,
            uncertainty_scores, sample_types
        )
        
        # 3. Rematch supervision loss
        rematch_loss_result = self.rematch_supervision(
            transport_matrix, visual_bge, textual_bge, uncertain
        )
        
        # 4. Evidence learning loss
        evidence_loss_result = self.evidence_learning(
            evidence_bge, evidence_tse, confident_clean, confident_noisy
        )
        
        # 5. Complementary contrastive loss
        complementary_loss = complementary_result['complementary_loss']
        
        # 6. Cost network loss (if provided)
        if cost_loss is None:
            cost_loss = torch.tensor(0.0, device=device)
            
        # Combine all losses
        total_tal_loss = tal_bge_result['weighted_tal_loss'] + tal_tse_result['weighted_tal_loss']
        total_rematch_loss = rematch_loss_result['total_rematch_loss']
        total_evidence_loss = evidence_loss_result['total_evidence_loss']
        
        # Final combined loss
        combined_loss = (
            self.tal_weight * total_tal_loss +
            self.rematch_weight * total_rematch_loss +
            self.evidence_weight * total_evidence_loss +
            self.complementary_weight * complementary_loss +
            self.cost_weight * cost_loss
        )
        
        # Collect all results
        result = {
            # Individual component losses
            'tal_bge_loss': tal_bge_result['weighted_tal_loss'],
            'tal_tse_loss': tal_tse_result['weighted_tal_loss'],
            'total_tal_loss': total_tal_loss,
            
            'rematch_kl_loss': rematch_loss_result['rematch_kl_loss'],
            'rematch_alignment_loss': rematch_loss_result['rematch_alignment_loss'],
            'rematch_loss': total_rematch_loss,  # For compatibility with tests
            'total_rematch_loss': total_rematch_loss,
            
            'evidence_loss': total_evidence_loss,
            'complementary_loss': complementary_loss,
            'cost_loss': cost_loss,
            
            # Final loss
            'combined_loss': combined_loss,
            
            # Additional info
            'tal_bge_result': tal_bge_result,
            'tal_tse_result': tal_tse_result,
            'rematch_loss_result': rematch_loss_result,
            'evidence_loss_result': evidence_loss_result,
            
            # Loss weights for monitoring
            'loss_weights': {
                'tal_weight': self.tal_weight,
                'rematch_weight': self.rematch_weight,
                'evidence_weight': self.evidence_weight,
                'complementary_weight': self.complementary_weight,
                'cost_weight': self.cost_weight
            }
        }
        
        return result
    
    def update_loss_weights(self, epoch, total_epochs):
        """
        Dynamic loss weight scheduling
        
        Args:
            epoch: int - current epoch
            total_epochs: int - total number of epochs
        """
        progress = epoch / total_epochs
        
        # Gradually increase rematch and complementary weights
        self.rematch_weight = 0.2 + 0.6 * progress
        self.complementary_weight = 0.5 + 0.5 * progress
        
        # Gradually decrease evidence weight after initial learning
        if epoch > total_epochs * 0.3:
            self.evidence_weight = 0.3 * (1 - (progress - 0.3) / 0.7)
        
        # Cost weight remains constant
        self.cost_weight = 0.2 