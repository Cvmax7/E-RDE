import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Dirichlet


class MultiViewEvidenceExtractor(nn.Module):
    """
    Extract multiple views from visual and textual features for evidence learning
    """
    def __init__(self, bge_dim, tse_dim, num_views=4, hidden_dim=256):
        super().__init__()
        self.num_views = num_views
        self.bge_dim = bge_dim
        self.tse_dim = tse_dim
        
        # Projection heads for different views - separate for BGE and TSE
        self.visual_bge_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bge_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(num_views)
        ])
        
        self.textual_bge_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bge_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(num_views)
        ])
        
        self.visual_tse_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tse_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(num_views)
        ])
        
        self.textual_tse_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tse_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(num_views)
        ])
        
    def forward_bge(self, visual_feats, textual_feats):
        """
        Forward for BGE features
        Args:
            visual_feats: [B, bge_dim] - BGE visual features
            textual_feats: [B, bge_dim] - BGE textual features
        Returns:
            visual_views: [num_views, B, hidden_dim]
            textual_views: [num_views, B, hidden_dim]
        """
        visual_views = []
        textual_views = []
        
        for i in range(self.num_views):
            v_view = self.visual_bge_projectors[i](visual_feats)
            t_view = self.textual_bge_projectors[i](textual_feats)
            
            # L2 normalize
            v_view = F.normalize(v_view, p=2, dim=-1)
            t_view = F.normalize(t_view, p=2, dim=-1)
            
            visual_views.append(v_view)
            textual_views.append(t_view)
            
        return torch.stack(visual_views), torch.stack(textual_views)
    
    def forward_tse(self, visual_feats, textual_feats):
        """
        Forward for TSE features
        Args:
            visual_feats: [B, tse_dim] - TSE visual features
            textual_feats: [B, tse_dim] - TSE textual features
        Returns:
            visual_views: [num_views, B, hidden_dim]
            textual_views: [num_views, B, hidden_dim]
        """
        visual_views = []
        textual_views = []
        
        for i in range(self.num_views):
            v_view = self.visual_tse_projectors[i](visual_feats)
            t_view = self.textual_tse_projectors[i](textual_feats)
            
            # L2 normalize
            v_view = F.normalize(v_view, p=2, dim=-1)
            t_view = F.normalize(t_view, p=2, dim=-1)
            
            visual_views.append(v_view)
            textual_views.append(t_view)
            
        return torch.stack(visual_views), torch.stack(textual_views)


class EvidenceNetwork(nn.Module):
    """
    Evidence network that computes evidence values from similarity scores
    """
    def __init__(self, num_views=4, hidden_dim=128):
        super().__init__()
        self.num_views = num_views
        
        # Evidence computation network
        self.evidence_net = nn.Sequential(
            nn.Linear(num_views, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # evidence for match/non-match
        )
        
    def forward(self, similarities):
        """
        Args:
            similarities: [B, num_views] - similarity scores from different views
        Returns:
            evidence: [B, 2] - evidence values for [match, non-match]
        """
        # Apply softplus + exponential transformation to ensure non-negative evidence
        evidence = self.evidence_net(similarities)
        evidence = F.softplus(evidence) + 1e-8  # Ensure positive
        return evidence


class SubjectiveLogicUncertainty(nn.Module):
    """
    Subjective Logic based uncertainty calculation using Dirichlet distribution
    """
    def __init__(self, num_views=4):
        super().__init__()
        self.num_views = num_views
        
    def dempster_shafer_fusion(self, evidences):
        """
        Fuse evidence from multiple views using Dempster-Shafer rule
        Args:
            evidences: [num_views, B, 2] - evidence from each view
        Returns:
            fused_evidence: [B, 2] - fused evidence
        """
        # Start with first view
        fused = evidences[0]
        
        for i in range(1, self.num_views):
            current = evidences[i]
            
            # Dempster-Shafer combination rule
            # m1(A) * m2(A) / (1 - conflict)
            conflict = torch.sum(fused[:, :1] * current[:, 1:] + fused[:, 1:] * current[:, :1], dim=1, keepdim=True)
            conflict = torch.clamp(conflict, max=0.99)  # Avoid division by zero
            
            combined = torch.zeros_like(fused)
            combined[:, 0] = (fused[:, 0] * current[:, 0]) / (1 - conflict[:, 0])
            combined[:, 1] = (fused[:, 1] * current[:, 1]) / (1 - conflict[:, 0])
            
            fused = combined
            
        return fused
    
    def compute_uncertainty(self, evidence):
        """
        Compute uncertainty using Subjective Logic
        Args:
            evidence: [B, 2] - evidence values [match, non-match]
        Returns:
            uncertainty: [B] - uncertainty mass
            belief: [B, 2] - belief masses for [match, non-match]
        """
        # Dirichlet parameters (evidence + 1)
        alpha = evidence + 1.0
        alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        
        # Belief masses (expected probabilities)
        belief = evidence / alpha_sum
        
        # Uncertainty mass
        uncertainty = 2.0 / alpha_sum  # For binary classification
        uncertainty = uncertainty.squeeze(-1)
        
        return uncertainty, belief


class UncertaintyGuidedDivision(nn.Module):
    """
    Enhanced version of CCD using continuous uncertainty scores
    """
    def __init__(self, uncertainty_threshold=0.5, confidence_threshold=0.7):
        super().__init__()
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        
    def forward(self, uncertainty_a, belief_a, uncertainty_b, belief_b):
        """
        Args:
            uncertainty_a: [B] - uncertainty from BGE embedding
            belief_a: [B, 2] - belief from BGE embedding
            uncertainty_b: [B] - uncertainty from TSE embedding  
            belief_b: [B, 2] - belief from TSE embedding
        Returns:
            division_result: dict with keys:
                - confident_clean: [B] - boolean mask
                - confident_noisy: [B] - boolean mask
                - uncertain: [B] - boolean mask
                - uncertainty_scores: [B] - continuous uncertainty scores
                - soft_labels: [B] - soft labels for uncertain samples
        """
        batch_size = uncertainty_a.size(0)
        
        # Average uncertainty and belief from both embeddings
        avg_uncertainty = (uncertainty_a + uncertainty_b) / 2.0
        avg_belief = (belief_a + belief_b) / 2.0
        
        # Consensus between two embeddings
        belief_diff = torch.norm(belief_a - belief_b, dim=1)
        consensus_score = 1.0 - belief_diff  # Higher score means more consensus
        
        # Combined uncertainty score (higher means more uncertain)
        combined_uncertainty = avg_uncertainty + 0.5 * (1.0 - consensus_score)
        combined_uncertainty = torch.clamp(combined_uncertainty, 0.0, 1.0)
        
        # Division based on uncertainty and confidence
        max_belief_a = torch.max(belief_a, dim=1)[0]
        max_belief_b = torch.max(belief_b, dim=1)[0]
        avg_confidence = (max_belief_a + max_belief_b) / 2.0
        
        # Confident clean: low uncertainty + high confidence + belief in match
        confident_clean = (combined_uncertainty < self.uncertainty_threshold) & \
                         (avg_confidence > self.confidence_threshold) & \
                         (avg_belief[:, 0] > avg_belief[:, 1])
        
        # Confident noisy: low uncertainty + high confidence + belief in non-match
        confident_noisy = (combined_uncertainty < self.uncertainty_threshold) & \
                         (avg_confidence > self.confidence_threshold) & \
                         (avg_belief[:, 0] <= avg_belief[:, 1])
        
        # Uncertain: high uncertainty or low confidence
        uncertain = ~(confident_clean | confident_noisy)
        
        # Soft labels for uncertain samples (use belief values)
        soft_labels = avg_belief[:, 0]  # Probability of being a match
        
        return {
            'confident_clean': confident_clean,
            'confident_noisy': confident_noisy, 
            'uncertain': uncertain,
            'uncertainty_scores': combined_uncertainty,
            'soft_labels': soft_labels
        }


class AdaptiveMarginCalculator(nn.Module):
    """
    Calculate adaptive margins based on uncertainty scores
    """
    def __init__(self, base_margin=0.2, margin_range=(0.1, 0.4)):
        super().__init__()
        self.base_margin = base_margin
        self.min_margin, self.max_margin = margin_range
        
    def forward(self, uncertainty_scores, sample_type):
        """
        Args:
            uncertainty_scores: [B] - uncertainty scores
            sample_type: [B] - sample types (0: clean, 1: noisy, 2: uncertain)
        Returns:
            adaptive_margins: [B] - adaptive margin values
        """
        margins = torch.full_like(uncertainty_scores, self.base_margin)
        
        # For uncertain samples: higher uncertainty -> larger margin for negatives
        uncertain_mask = (sample_type == 2)
        if uncertain_mask.any():
            # Scale margin based on uncertainty
            uncertain_margins = self.min_margin + \
                              (self.max_margin - self.min_margin) * uncertainty_scores[uncertain_mask]
            margins[uncertain_mask] = uncertain_margins
            
        # For confident clean samples: lower margin to encourage tighter clusters
        clean_mask = (sample_type == 0)
        if clean_mask.any():
            margins[clean_mask] = self.min_margin
            
        # For confident noisy samples: higher margin to push them away
        noisy_mask = (sample_type == 1)
        if noisy_mask.any():
            margins[noisy_mask] = self.max_margin
            
        return margins


class UncertaintyDrivenSoftLabeling(nn.Module):
    """
    Main module for uncertainty-driven soft labeling
    """
    def __init__(self, bge_dim, tse_dim, num_views=4, uncertainty_threshold=0.5, confidence_threshold=0.7):
        super().__init__()
        self.num_views = num_views
        
        # Evidence extraction with separate dimensions
        self.evidence_extractor = MultiViewEvidenceExtractor(
            bge_dim=bge_dim, 
            tse_dim=tse_dim, 
            num_views=num_views
        )
        
        # Evidence networks for both BGE and TSE
        self.evidence_net_bge = EvidenceNetwork(num_views)
        self.evidence_net_tse = EvidenceNetwork(num_views)
        
        # Subjective logic uncertainty calculator
        self.uncertainty_calc = SubjectiveLogicUncertainty(num_views)
        
        # Division module
        self.division_module = UncertaintyGuidedDivision(
            uncertainty_threshold, confidence_threshold
        )
        
        # Adaptive margin calculator
        self.margin_calc = AdaptiveMarginCalculator()
        
    def forward(self, visual_bge, textual_bge, visual_tse, textual_tse):
        """
        Main forward pass
        Args:
            visual_bge: [B, bge_dim]
            textual_bge: [B, bge_dim]
            visual_tse: [B, tse_dim]
            textual_tse: [B, tse_dim]
        """
        batch_size = visual_bge.size(0)
        
        # Extract multi-view features
        visual_views_bge, textual_views_bge = self.evidence_extractor.forward_bge(visual_bge, textual_bge)
        visual_views_tse, textual_views_tse = self.evidence_extractor.forward_tse(visual_tse, textual_tse)
        
        # Compute similarities for each view
        similarities_bge = []
        similarities_tse = []
        
        for i in range(self.evidence_extractor.num_views):
            # BGE similarities
            sim_bge = torch.sum(visual_views_bge[i] * textual_views_bge[i], dim=1)
            similarities_bge.append(sim_bge)
            
            # TSE similarities
            sim_tse = torch.sum(visual_views_tse[i] * textual_views_tse[i], dim=1)
            similarities_tse.append(sim_tse)
            
        similarities_bge = torch.stack(similarities_bge, dim=1)  # [B, num_views]
        similarities_tse = torch.stack(similarities_tse, dim=1)  # [B, num_views]
        
        # Generate evidence values
        evidence_bge = self.evidence_net_bge(similarities_bge)  # [B, 2]
        evidence_tse = self.evidence_net_tse(similarities_tse)  # [B, 2]
        
        # Fuse evidence across views (if multiple views per embedding)
        # Here we treat BGE and TSE as single "views" each, so no fusion needed
        
        # Compute uncertainty using Subjective Logic
        uncertainty_bge, belief_bge = self.uncertainty_calc.compute_uncertainty(evidence_bge)
        uncertainty_tse, belief_tse = self.uncertainty_calc.compute_uncertainty(evidence_tse)
        
        # Uncertainty-guided division
        division_result = self.division_module(uncertainty_bge, belief_bge, 
                                                   uncertainty_tse, belief_tse)
        
        # Determine sample types for margin calculation
        sample_types = torch.zeros(batch_size, dtype=torch.long, device=visual_bge.device)
        sample_types[division_result['confident_clean']] = 0
        sample_types[division_result['confident_noisy']] = 1  
        sample_types[division_result['uncertain']] = 2
        
        # Calculate adaptive margins
        adaptive_margins = self.margin_calc(division_result['uncertainty_scores'], sample_types)
        
        result = {
            'division_result': division_result,
            'adaptive_margins': adaptive_margins,
            'sample_types': sample_types,
            'uncertainty_bge': uncertainty_bge,
            'uncertainty_tse': uncertainty_tse,
            'belief_bge': belief_bge,
            'belief_tse': belief_tse,
            'evidence_bge': evidence_bge,
            'evidence_tse': evidence_tse
        }
        
        return result 