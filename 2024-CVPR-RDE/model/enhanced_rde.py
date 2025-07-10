import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import RDE
from .uncertainty_evidence import UncertaintyDrivenSoftLabeling
from .optimal_transport import OptimalTransportModule
from .complementary_contrastive import ComplementaryContrastiveModule
from .enhanced_objectives import EnhancedLossObjectives


class EnhancedRDE(nn.Module):
    """
    Enhanced RDE model with uncertainty-driven soft labeling,
    optimal transport rematch, and complementary contrastive learning
    """
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.current_epoch = 0
        
        # Original RDE components
        self.base_rde = RDE(args, num_classes)
        self.embed_dim = self.base_rde.embed_dim
        
        # Enhanced components
        self.uncertainty_system = UncertaintyDrivenSoftLabeling(
            bge_dim=self.embed_dim,  # BGE features are 512-dim
            tse_dim=1024,  # TSE features are 1024-dim
            num_views=getattr(args, 'num_evidence_views', 4),
            uncertainty_threshold=getattr(args, 'uncertainty_threshold', 0.5),
            confidence_threshold=getattr(args, 'confidence_threshold', 0.7)
        )
        
        self.optimal_transport = OptimalTransportModule(
            cost_hidden_dim=getattr(args, 'cost_hidden_dim', 128),
            ot_reg=getattr(args, 'ot_reg', 0.1),
            ot_max_iter=getattr(args, 'ot_max_iter', 100),
            rematch_temperature=getattr(args, 'rematch_temperature', 0.07)
        )
        
        self.complementary_contrastive = ComplementaryContrastiveModule(
            temperature=getattr(args, 'ccl_temperature', 0.07),
            push_phase_epochs=getattr(args, 'push_phase_epochs', 15),
            transition_epochs=getattr(args, 'transition_epochs', 5)
        )
        
        self.enhanced_objectives = EnhancedLossObjectives(
            base_tau=args.tau,
            base_margin=args.margin,
            rematch_temp=getattr(args, 'rematch_temperature', 0.07),
            evidence_reg=getattr(args, 'evidence_reg', 0.1)
        )
        
        # Training phase control
        self.training_phases = {
            'warmup': getattr(args, 'warmup_epochs', 5),
            'uncertainty_learning': getattr(args, 'uncertainty_epochs', 10),
            'progressive_training': getattr(args, 'progressive_epochs', 50)
        }
        
    def set_epoch(self, epoch):
        """Set current epoch for all modules"""
        self.current_epoch = epoch
        self.complementary_contrastive.set_epoch(epoch)
        self.enhanced_objectives.update_loss_weights(epoch, self.args.num_epoch)
        
    def get_training_phase(self):
        """Determine current training phase"""
        if self.current_epoch < self.training_phases['warmup']:
            return 'warmup'
        elif self.current_epoch < self.training_phases['warmup'] + self.training_phases['uncertainty_learning']:
            return 'uncertainty_learning'
        else:
            return 'progressive_training'
            
    def encode_image(self, image):
        """BGE image encoding"""
        return self.base_rde.encode_image(image)
        
    def encode_text(self, text):
        """BGE text encoding"""
        return self.base_rde.encode_text(text)
        
    def encode_image_tse(self, image):
        """TSE image encoding"""
        return self.base_rde.encode_image_tse(image)
        
    def encode_text_tse(self, text):
        """TSE text encoding"""
        return self.base_rde.encode_text_tse(text)
        
    def extract_features(self, batch):
        """
        Extract both BGE and TSE features
        
        Args:
            batch: dict containing 'images' and 'caption_ids'
            
        Returns:
            features: dict containing all extracted features
        """
        images = batch['images']
        caption_ids = batch['caption_ids']
        
        # Get base model features (includes attention)
        image_feats, atten_i, text_feats, atten_t = self.base_rde.base_model(images, caption_ids)
        
        # BGE features
        visual_bge = image_feats[:, 0, :].float()
        textual_bge = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        # TSE features
        visual_tse = self.base_rde.visul_emb_layer(image_feats, atten_i).float()
        textual_tse = self.base_rde.texual_emb_layer(text_feats, caption_ids, atten_t).float()
        
        return {
            'visual_bge': visual_bge,
            'textual_bge': textual_bge,
            'visual_tse': visual_tse,
            'textual_tse': textual_tse,
            'image_feats': image_feats,
            'text_feats': text_feats,
            'atten_i': atten_i,
            'atten_t': atten_t
        }
        
    def warmup_forward(self, batch):
        """
        Warmup phase: standard RDE training
        """
        # Extract features
        features = self.extract_features(batch)
        
        # Use original RDE loss computation
        ret = {}
        ret.update({'temperature': 1 / self.base_rde.logit_scale})
        
        # Simple consensus division (binary)
        batch_size = features['visual_bge'].size(0)
        label_hat = torch.ones(batch_size, device=features['visual_bge'].device)
        batch['label_hat'] = label_hat
        
        # Original RDE losses
        from . import objectives
        loss1, loss2 = objectives.compute_rbs(
            features['visual_bge'], features['textual_bge'],
            features['visual_tse'], features['textual_tse'],
            batch['pids'], label_hat=label_hat,
            margin=self.args.margin, tau=self.args.tau,
            loss_type=self.base_rde.loss_type,
            logit_scale=self.base_rde.logit_scale
        )
        
        ret.update({'bge_loss': loss1, 'tse_loss': loss2})
        return ret
        
    def uncertainty_learning_forward(self, batch):
        """
        Uncertainty learning phase: learn evidence networks and uncertainty estimation
        """
        # Extract features
        features = self.extract_features(batch)
        
        # Uncertainty analysis
        uncertainty_result = self.uncertainty_system(
            features['visual_bge'], features['textual_bge'],
            features['visual_tse'], features['textual_tse']
        )
        
        division_result = uncertainty_result['division_result']
        
        # Use original TAL for confident samples and weighted TAL for uncertain ones
        confident_mask = division_result['confident_clean'] | division_result['confident_noisy']
        uncertain_mask = division_result['uncertain']
        
        # Create label_hat based on uncertainty analysis
        label_hat = torch.zeros_like(division_result['uncertainty_scores'])
        label_hat[division_result['confident_clean']] = 1.0
        label_hat[division_result['confident_noisy']] = 0.0
        label_hat[uncertain_mask] = division_result['soft_labels'][uncertain_mask]
        
        batch['label_hat'] = label_hat
        
        # Compute losses with uncertainty weighting
        from . import objectives
        loss1, loss2 = objectives.compute_rbs(
            features['visual_bge'], features['textual_bge'],
            features['visual_tse'], features['textual_tse'],
            batch['pids'], label_hat=label_hat,
            margin=self.args.margin, tau=self.args.tau,
            loss_type=self.base_rde.loss_type,
            logit_scale=self.base_rde.logit_scale
        )
        
        # Add evidence learning loss
        evidence_loss_result = self.enhanced_objectives.evidence_learning(
            uncertainty_result['evidence_bge'], uncertainty_result['evidence_tse'],
            division_result['confident_clean'], division_result['confident_noisy']
        )
        
        total_loss = loss1 + loss2 + 0.3 * evidence_loss_result['total_evidence_loss']
        
        ret = {
            'bge_loss': loss1,
            'tse_loss': loss2,
            'evidence_loss': evidence_loss_result['total_evidence_loss'],
            'total_loss': total_loss,
            'temperature': 1 / self.base_rde.logit_scale,
            'uncertainty_result': uncertainty_result,
            'num_confident_clean': division_result['confident_clean'].sum().item(),
            'num_confident_noisy': division_result['confident_noisy'].sum().item(),
            'num_uncertain': uncertain_mask.sum().item()
        }
        
        return ret
        
    def progressive_training_forward(self, batch):
        """
        Progressive training phase: full enhanced RDE with all components
        """
        # Extract features
        features = self.extract_features(batch)
        
        # 1. Uncertainty analysis
        uncertainty_result = self.uncertainty_system(
            features['visual_bge'], features['textual_bge'],
            features['visual_tse'], features['textual_tse']
        )
        
        division_result = uncertainty_result['division_result']
        
        # 2. Optimal transport rematch for uncertain samples
        rematch_result = self.optimal_transport(
            features['visual_bge'], features['textual_bge'],
            division_result['uncertain'], training=self.training
        )
        
        # 3. Complementary contrastive learning
        complementary_result = self.complementary_contrastive(
            features['visual_bge'], features['textual_bge'],
            rematch_result['transport_matrix'],
            division_result['confident_clean'],
            division_result['confident_noisy'],
            division_result['uncertain'],
            division_result['uncertainty_scores'],
            uncertainty_result['sample_types']
        )
        
        # 4. Enhanced loss computation
        enhanced_loss_result = self.enhanced_objectives(
            features['visual_bge'], features['textual_bge'],
            features['visual_tse'], features['textual_tse'],
            batch['pids'], uncertainty_result, rematch_result,
            complementary_result, rematch_result.get('cost_loss', None)
        )
        
        # Prepare return values
        ret = {
            'combined_loss': enhanced_loss_result['combined_loss'],
            'tal_bge_loss': enhanced_loss_result['tal_bge_loss'],
            'tal_tse_loss': enhanced_loss_result['tal_tse_loss'],
            'rematch_loss': enhanced_loss_result['total_rematch_loss'],
            'evidence_loss': enhanced_loss_result['evidence_loss'],
            'complementary_loss': enhanced_loss_result['complementary_loss'],
            'cost_loss': enhanced_loss_result['cost_loss'],
            'temperature': 1 / self.base_rde.logit_scale,
            
            # Additional info for monitoring
            'uncertainty_result': uncertainty_result,
            'rematch_result': rematch_result,
            'complementary_result': complementary_result,
            'training_phase': complementary_result['training_phase'],
            'loss_weights': enhanced_loss_result['loss_weights'],
            
            # Statistics
            'num_confident_clean': division_result['confident_clean'].sum().item(),
            'num_confident_noisy': division_result['confident_noisy'].sum().item(),
            'num_uncertain': division_result['uncertain'].sum().item(),
            'num_rematch_pairs': rematch_result['rematch_info']['num_uncertain']
        }
        
        return ret
        
    def forward(self, batch):
        """
        Forward pass with phase-dependent processing
        
        Args:
            batch: dict containing input data
            
        Returns:
            result: dict containing losses and information
        """
        phase = self.get_training_phase()
        
        if phase == 'warmup':
            return self.warmup_forward(batch)
        elif phase == 'uncertainty_learning':
            return self.uncertainty_learning_forward(batch)
        else:  # progressive_training
            return self.progressive_training_forward(batch)
            
    def compute_similarities(self, visual_feats, textual_feats):
        """
        Compute similarities for inference
        
        Args:
            visual_feats: [N, D] - visual features
            textual_feats: [M, D] - textual features
            
        Returns:
            similarities: [N, M] - similarity matrix
        """
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        textual_norm = F.normalize(textual_feats, p=2, dim=-1)
        return visual_norm @ textual_norm.t()
        
    def inference_mode(self):
        """Set model to inference mode"""
        self.eval()
        
    def get_statistics(self):
        """Get training statistics"""
        return {
            'current_epoch': self.current_epoch,
            'training_phase': self.get_training_phase(),
            'uncertainty_threshold': self.uncertainty_system.division_module.uncertainty_threshold,
            'confidence_threshold': self.uncertainty_system.division_module.confidence_threshold,
            'ot_regularization': self.optimal_transport.rematcher.partial_ot.sinkhorn.reg,
            'ccl_phase': self.complementary_contrastive.progressive_cl.get_training_phase()
        }


def build_enhanced_model(args, num_classes=11003):
    """
    Build enhanced RDE model
    
    Args:
        args: configuration arguments
        num_classes: number of identity classes
        
    Returns:
        model: EnhancedRDE model
    """
    model = EnhancedRDE(args, num_classes)
    
    # Note: Do not convert to fp16 to maintain compatibility with original RDE
    # The original RDE also has convert_weights commented out in clip_model.py
    # from .clip_model import convert_weights
    # convert_weights(model)
    
    return model 