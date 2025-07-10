import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnableCostNetwork(nn.Module):
    """
    Learnable cost function inspired by L2RM
    Maps similarity matrices to reliable semantic costs
    """
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # Multi-layer cost prediction network
        self.cost_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Ensure cost is in [0, 1]
        )
        
        # Initialize to identity mapping initially
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to approximate identity mapping"""
        for module in self.cost_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, similarity_matrix):
        """
        Args:
            similarity_matrix: [B, B] - pairwise similarity matrix
        Returns:
            cost_matrix: [B, B] - transformed cost matrix
        """
        # Flatten similarity matrix for processing
        B = similarity_matrix.size(0)
        sim_flat = similarity_matrix.view(-1, 1)  # [B*B, 1]
        
        # Apply cost transformation
        cost_flat = self.cost_net(sim_flat)  # [B*B, 1]
        
        # Reshape back to matrix form
        cost_matrix = cost_flat.view(B, B)
        
        # Convert similarity to cost (higher similarity -> lower cost)
        cost_matrix = 1.0 - cost_matrix
        
        return cost_matrix


class SinkhornSolver(nn.Module):
    """
    Sinkhorn algorithm for solving optimal transport with entropy regularization
    """
    def __init__(self, reg=0.1, max_iter=100, threshold=1e-3):
        super().__init__()
        self.reg = reg
        self.max_iter = max_iter
        self.threshold = threshold
        
    def forward(self, cost_matrix, a=None, b=None, mask=None):
        """
        Solve optimal transport using Sinkhorn algorithm
        
        Args:
            cost_matrix: [B, B] - cost matrix
            a: [B] - source distribution (if None, uniform)
            b: [B] - target distribution (if None, uniform)
            mask: [B, B] - mask for partial OT (1 for allowed transport, 0 for forbidden)
        
        Returns:
            transport_matrix: [B, B] - optimal transport plan
        """
        B = cost_matrix.size(0)
        device = cost_matrix.device
        
        # Default to uniform distributions
        if a is None:
            a = torch.ones(B, device=device) / B
        if b is None:
            b = torch.ones(B, device=device) / B
            
        # Initialize transport kernel
        K = torch.exp(-cost_matrix / self.reg)
        
        # Apply mask for partial OT
        if mask is not None:
            K = K * mask
            
        # Sinkhorn iterations
        u = torch.ones(B, device=device) / B
        v = torch.ones(B, device=device) / B
        
        for i in range(self.max_iter):
            u_prev = u.clone()
            
            # Update v
            v = b / (K.t() @ u + 1e-8)
            
            # Update u  
            u = a / (K @ v + 1e-8)
            
            # Check convergence
            error = torch.norm(u - u_prev)
            if error < self.threshold:
                break
                
        # Compute transport matrix
        transport_matrix = torch.diag(u) @ K @ torch.diag(v)
        
        return transport_matrix


class PartialOptimalTransport(nn.Module):
    """
    Partial Optimal Transport with masking for original positive pairs
    """
    def __init__(self, reg=0.1, max_iter=100):
        super().__init__()
        self.sinkhorn = SinkhornSolver(reg, max_iter)
        
    def create_mask(self, uncertain_indices, confident_indices, batch_size):
        """
        Create mask that prevents transport between confident samples
        
        Args:
            uncertain_indices: [K] - indices of uncertain samples
            confident_indices: [M] - indices of confident samples  
            batch_size: int - total batch size
            
        Returns:
            mask: [B, B] - transport mask
        """
        mask = torch.zeros(batch_size, batch_size)
        
        # Allow transport only between uncertain samples
        for i in uncertain_indices:
            for j in uncertain_indices:
                mask[i, j] = 1.0
                
        return mask
        
    def forward(self, cost_matrix, uncertain_mask):
        """
        Perform partial optimal transport on uncertain samples only
        
        Args:
            cost_matrix: [B, B] - semantic cost matrix
            uncertain_mask: [B] - boolean mask for uncertain samples
            
        Returns:
            transport_matrix: [B, B] - partial transport plan
            rematch_info: dict - information about rematchings
        """
        B = cost_matrix.size(0)
        device = cost_matrix.device
        
        # Get uncertain sample indices
        uncertain_indices = torch.where(uncertain_mask)[0]
        confident_indices = torch.where(~uncertain_mask)[0]
        
        # Create transport mask
        transport_mask = torch.zeros(B, B, device=device)
        if len(uncertain_indices) > 1:
            for i in uncertain_indices:
                for j in uncertain_indices:
                    transport_mask[i, j] = 1.0
        
        # Extract cost submatrix for uncertain samples
        if len(uncertain_indices) > 1:
            uncertain_cost = cost_matrix[uncertain_indices][:, uncertain_indices]
            
            # Solve OT on uncertain subset
            uncertain_transport = self.sinkhorn(uncertain_cost)
            
            # Embed back into full matrix
            transport_matrix = torch.zeros(B, B, device=device)
            for idx_i, i in enumerate(uncertain_indices):
                for idx_j, j in enumerate(uncertain_indices):
                    transport_matrix[i, j] = uncertain_transport[idx_i, idx_j]
        else:
            # No transport needed if less than 2 uncertain samples
            transport_matrix = torch.eye(B, device=device)
            
        # Preserve identity for confident samples
        for i in confident_indices:
            transport_matrix[i, i] = 1.0
            
        return transport_matrix, {
            'num_uncertain': len(uncertain_indices),
            'num_confident': len(confident_indices),
            'uncertain_indices': uncertain_indices
        }


class SemanticRematcher(nn.Module):
    """
    Complete semantic rematcher combining cost learning and optimal transport
    """
    def __init__(self, cost_hidden_dim=128, ot_reg=0.1, ot_max_iter=100):
        super().__init__()
        
        self.cost_network = LearnableCostNetwork(cost_hidden_dim)
        self.partial_ot = PartialOptimalTransport(ot_reg, ot_max_iter)
        
        # Training parameters for cost network
        self.cost_learning_ratio = 0.1  # Ratio of batch to use for cost learning
        
    def generate_training_pairs(self, batch_size, device):
        """
        Generate training pairs for cost network by reconstructing some clean pairs
        
        Args:
            batch_size: int
            device: torch device
            
        Returns:
            supervision_matrix: [B, B] - supervision matrix for cost learning
        """
        # Create supervision matrix with some known positive pairs
        supervision = torch.zeros(batch_size, batch_size, device=device)
        
        # Randomly select some indices to form positive pairs
        num_pos_pairs = max(1, int(batch_size * self.cost_learning_ratio))
        selected_indices = torch.randperm(batch_size, device=device)[:num_pos_pairs]
        
        for i in range(0, len(selected_indices), 2):
            if i + 1 < len(selected_indices):
                idx1, idx2 = selected_indices[i], selected_indices[i + 1]
                supervision[idx1, idx2] = 1.0
                supervision[idx2, idx1] = 1.0
                
        return supervision
        
    def compute_cost_learning_loss(self, similarity_matrix, supervision_matrix):
        """
        Compute loss for training the cost network
        
        Args:
            similarity_matrix: [B, B] - similarity matrix
            supervision_matrix: [B, B] - ground truth supervision
            
        Returns:
            cost_loss: scalar - cost learning loss
        """
        # Get cost predictions
        predicted_costs = self.cost_network(similarity_matrix)
        
        # Convert supervision to costs (positive pairs should have low cost)
        target_costs = 1.0 - supervision_matrix
        
        # Only supervise on known pairs
        mask = (supervision_matrix > 0).float()
        
        if mask.sum() > 0:
            cost_loss = F.mse_loss(predicted_costs * mask, target_costs * mask, reduction='sum') / mask.sum()
        else:
            cost_loss = torch.tensor(0.0, device=similarity_matrix.device, requires_grad=True)
            
        return cost_loss
        
    def forward(self, visual_feats, textual_feats, uncertain_mask, training=True):
        """
        Complete rematcher forward pass
        
        Args:
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features
            uncertain_mask: [B] - mask for uncertain samples
            training: bool - whether in training mode
            
        Returns:
            result: dict containing rematch results and losses
        """
        B = visual_feats.size(0)
        device = visual_feats.device
        
        # Compute similarity matrix
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        textual_norm = F.normalize(textual_feats, p=2, dim=-1) 
        similarity_matrix = visual_norm @ textual_norm.t()
        
        # Generate cost matrix using learnable network
        cost_matrix = self.cost_network(similarity_matrix)
        
        # Perform partial optimal transport
        transport_matrix, rematch_info = self.partial_ot(cost_matrix, uncertain_mask)
        
        result = {
            'transport_matrix': transport_matrix,
            'cost_matrix': cost_matrix,
            'similarity_matrix': similarity_matrix,
            'rematch_info': rematch_info
        }
        
        # Add cost learning loss if training
        if training:
            supervision_matrix = self.generate_training_pairs(B, device)
            cost_loss = self.compute_cost_learning_loss(similarity_matrix, supervision_matrix)
            result['cost_loss'] = cost_loss
            result['supervision_matrix'] = supervision_matrix
            
        return result


class RematchLossCalculator(nn.Module):
    """
    Calculate losses based on rematch results
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def symmetric_kl_loss(self, transport_matrix, similarity_matrix):
        """
        Symmetric KL divergence loss inspired by L2RM
        
        Args:
            transport_matrix: [B, B] - optimal transport plan (soft labels)
            similarity_matrix: [B, B] - current model predictions
            
        Returns:
            kl_loss: scalar - symmetric KL divergence loss
        """
        # Convert similarities to probabilities
        logits = similarity_matrix / self.temperature
        pred_probs = F.softmax(logits, dim=1)
        log_pred_probs = F.log_softmax(logits, dim=1)
        
        # Transport matrix as target distribution
        target_probs = transport_matrix
        target_probs = target_probs / (target_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        # Symmetric KL divergence
        kl_1 = F.kl_div(log_pred_probs, target_probs, reduction='none').sum(dim=1)
        kl_2 = F.kl_div(torch.log(target_probs + 1e-8), pred_probs, reduction='none').sum(dim=1)
        
        symmetric_kl = (kl_1 + kl_2) / 2.0
        
        return symmetric_kl.mean()
        
    def rematch_alignment_loss(self, transport_matrix, visual_feats, textual_feats):
        """
        Direct alignment loss based on transport matrix
        
        Args:
            transport_matrix: [B, B] - transport plan
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features
            
        Returns:
            alignment_loss: scalar
        """
        # Normalize features
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        textual_norm = F.normalize(textual_feats, p=2, dim=-1)
        
        # Compute similarities
        similarities = visual_norm @ textual_norm.t()
        
        # Weighted alignment based on transport weights
        alignment_loss = -torch.sum(transport_matrix * similarities) / transport_matrix.sum()
        
        return alignment_loss
        
    def forward(self, rematch_result, visual_feats, textual_feats, loss_type='symmetric_kl'):
        """
        Calculate rematch-based losses
        
        Args:
            rematch_result: dict - output from SemanticRematcher
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features  
            loss_type: str - type of loss to compute
            
        Returns:
            loss: scalar - rematch loss
        """
        transport_matrix = rematch_result['transport_matrix']
        similarity_matrix = rematch_result['similarity_matrix']
        
        if loss_type == 'symmetric_kl':
            return self.symmetric_kl_loss(transport_matrix, similarity_matrix)
        elif loss_type == 'alignment':
            return self.rematch_alignment_loss(transport_matrix, visual_feats, textual_feats)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class OptimalTransportModule(nn.Module):
    """
    Complete optimal transport module integrating all components
    """
    def __init__(self, cost_hidden_dim=128, ot_reg=0.1, ot_max_iter=100, 
                 rematch_temperature=0.07):
        super().__init__()
        
        self.rematcher = SemanticRematcher(cost_hidden_dim, ot_reg, ot_max_iter)
        self.loss_calculator = RematchLossCalculator(rematch_temperature)
        
    def forward(self, visual_feats, textual_feats, uncertain_mask, training=True):
        """
        Complete optimal transport forward pass
        
        Args:
            visual_feats: [B, D] - visual features
            textual_feats: [B, D] - textual features
            uncertain_mask: [B] - boolean mask for uncertain samples
            training: bool - whether in training mode
            
        Returns:
            result: dict containing all OT results and losses
        """
        # Perform rematch
        rematch_result = self.rematcher(visual_feats, textual_feats, uncertain_mask, training)
        
        # Calculate rematch losses
        rematch_loss = self.loss_calculator(rematch_result, visual_feats, textual_feats, 'symmetric_kl')
        alignment_loss = self.loss_calculator(rematch_result, visual_feats, textual_feats, 'alignment')
        
        result = rematch_result.copy()
        result.update({
            'rematch_loss': rematch_loss,
            'alignment_loss': alignment_loss
        })
        
        return result 