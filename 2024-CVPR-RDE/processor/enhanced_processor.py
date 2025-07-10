import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np


def enhanced_do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
                     scheduler, checkpointer):
    """
    Enhanced training function for the improved RDE model with three-phase training
    """
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("Enhanced_RDE.train")
    logger.info('Starting enhanced RDE training with three-phase strategy')

    # Enhanced meters for all loss components
    meters = {
        "total_loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "tal_bge_loss": AverageMeter(),
        "tal_tse_loss": AverageMeter(),
        "rematch_loss": AverageMeter(),
        "evidence_loss": AverageMeter(),
        "complementary_loss": AverageMeter(),
        "cost_loss": AverageMeter(),
        "combined_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0
    
    # Enhanced statistics tracking
    phase_statistics = {
        'warmup': {'epochs': 0, 'avg_loss': 0.0},
        'uncertainty_learning': {'epochs': 0, 'avg_loss': 0.0, 'avg_evidence_loss': 0.0},
        'progressive_training': {'epochs': 0, 'avg_combined_loss': 0.0}
    }

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        
        # Set epoch for all modules
        if hasattr(model, 'module'):
            model.module.set_epoch(epoch)
        else:
            model.set_epoch(epoch)
            
        # Get current training phase
        current_phase = model.module.get_training_phase() if hasattr(model, 'module') else model.get_training_phase()
        
        logger.info(f'Epoch {epoch}/{num_epoch} - Training Phase: {current_phase}')
        
        # Reset meters
        for meter in meters.values():
            meter.reset()

        model.train()
        
        epoch_losses = []
        epoch_stats = {
            'num_confident_clean': [],
            'num_confident_noisy': [],
            'num_uncertain': [],
            'num_rematch_pairs': []
        }

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            ret = model(batch)
            
            # Determine total loss based on training phase
            if current_phase == 'warmup':
                total_loss = ret['bge_loss'] + ret['tse_loss']
                meters['bge_loss'].update(ret['bge_loss'].item(), batch['images'].shape[0])
                meters['tse_loss'].update(ret['tse_loss'].item(), batch['images'].shape[0])
                
            elif current_phase == 'uncertainty_learning':
                total_loss = ret['total_loss']
                meters['bge_loss'].update(ret['bge_loss'].item(), batch['images'].shape[0])
                meters['tse_loss'].update(ret['tse_loss'].item(), batch['images'].shape[0])
                meters['evidence_loss'].update(ret['evidence_loss'].item(), batch['images'].shape[0])
                
                # Track uncertainty statistics
                epoch_stats['num_confident_clean'].append(ret['num_confident_clean'])
                epoch_stats['num_confident_noisy'].append(ret['num_confident_noisy'])
                epoch_stats['num_uncertain'].append(ret['num_uncertain'])
                
            else:  # progressive_training
                total_loss = ret['combined_loss']
                meters['combined_loss'].update(ret['combined_loss'].item(), batch['images'].shape[0])
                meters['tal_bge_loss'].update(ret['tal_bge_loss'].item(), batch['images'].shape[0])
                meters['tal_tse_loss'].update(ret['tal_tse_loss'].item(), batch['images'].shape[0])
                meters['rematch_loss'].update(ret['rematch_loss'].item(), batch['images'].shape[0])
                meters['evidence_loss'].update(ret['evidence_loss'].item(), batch['images'].shape[0])
                meters['complementary_loss'].update(ret['complementary_loss'].item(), batch['images'].shape[0])
                meters['cost_loss'].update(ret['cost_loss'].item(), batch['images'].shape[0])
                
                # Track all statistics
                epoch_stats['num_confident_clean'].append(ret['num_confident_clean'])
                epoch_stats['num_confident_noisy'].append(ret['num_confident_noisy'])
                epoch_stats['num_uncertain'].append(ret['num_uncertain'])
                epoch_stats['num_rematch_pairs'].append(ret['num_rematch_pairs'])

            # Update total loss meter
            batch_size = batch['images'].shape[0]
            meters['total_loss'].update(total_loss.item(), batch_size)
            epoch_losses.append(total_loss.item())

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            # Logging
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Phase[{current_phase}]"
                
                # Phase-specific logging
                if current_phase == 'warmup':
                    info_str += f", BGE Loss: {meters['bge_loss'].avg:.4f}"
                    info_str += f", TSE Loss: {meters['tse_loss'].avg:.4f}"
                elif current_phase == 'uncertainty_learning':
                    info_str += f", BGE Loss: {meters['bge_loss'].avg:.4f}"
                    info_str += f", TSE Loss: {meters['tse_loss'].avg:.4f}"
                    info_str += f", Evidence Loss: {meters['evidence_loss'].avg:.4f}"
                else:
                    info_str += f", Combined Loss: {meters['combined_loss'].avg:.4f}"
                    info_str += f", TAL BGE: {meters['tal_bge_loss'].avg:.4f}"
                    info_str += f", TAL TSE: {meters['tal_tse_loss'].avg:.4f}"
                    info_str += f", Rematch: {meters['rematch_loss'].avg:.4f}"
                    info_str += f", Complementary: {meters['complementary_loss'].avg:.4f}"
                
                info_str += f", Total Loss: {meters['total_loss'].avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        # End of epoch processing
        avg_epoch_loss = np.mean(epoch_losses)
        
        # Update phase statistics
        phase_statistics[current_phase]['epochs'] += 1
        if current_phase == 'warmup':
            phase_statistics[current_phase]['avg_loss'] = \
                (phase_statistics[current_phase]['avg_loss'] * (phase_statistics[current_phase]['epochs'] - 1) + avg_epoch_loss) / phase_statistics[current_phase]['epochs']
        elif current_phase == 'uncertainty_learning':
            phase_statistics[current_phase]['avg_loss'] = \
                (phase_statistics[current_phase]['avg_loss'] * (phase_statistics[current_phase]['epochs'] - 1) + avg_epoch_loss) / phase_statistics[current_phase]['epochs']
            if 'evidence_loss' in ret:
                phase_statistics[current_phase]['avg_evidence_loss'] = \
                    (phase_statistics[current_phase].get('avg_evidence_loss', 0) * (phase_statistics[current_phase]['epochs'] - 1) + ret['evidence_loss'].item()) / phase_statistics[current_phase]['epochs']
        else:
            phase_statistics[current_phase]['avg_combined_loss'] = \
                (phase_statistics[current_phase].get('avg_combined_loss', 0) * (phase_statistics[current_phase]['epochs'] - 1) + avg_epoch_loss) / phase_statistics[current_phase]['epochs']

        # TensorBoard logging
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('total_loss', meters['total_loss'].avg, epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        
        # Phase-specific TensorBoard logs
        if current_phase == 'warmup':
            tb_writer.add_scalar('bge_loss', meters['bge_loss'].avg, epoch)
            tb_writer.add_scalar('tse_loss', meters['tse_loss'].avg, epoch)
        elif current_phase == 'uncertainty_learning':
            tb_writer.add_scalar('bge_loss', meters['bge_loss'].avg, epoch)
            tb_writer.add_scalar('tse_loss', meters['tse_loss'].avg, epoch)
            tb_writer.add_scalar('evidence_loss', meters['evidence_loss'].avg, epoch)
            
            # Uncertainty statistics
            if epoch_stats['num_confident_clean']:
                tb_writer.add_scalar('avg_confident_clean', np.mean(epoch_stats['num_confident_clean']), epoch)
                tb_writer.add_scalar('avg_confident_noisy', np.mean(epoch_stats['num_confident_noisy']), epoch)
                tb_writer.add_scalar('avg_uncertain', np.mean(epoch_stats['num_uncertain']), epoch)
        else:
            tb_writer.add_scalar('combined_loss', meters['combined_loss'].avg, epoch)
            tb_writer.add_scalar('tal_bge_loss', meters['tal_bge_loss'].avg, epoch)
            tb_writer.add_scalar('tal_tse_loss', meters['tal_tse_loss'].avg, epoch)
            tb_writer.add_scalar('rematch_loss', meters['rematch_loss'].avg, epoch)
            tb_writer.add_scalar('evidence_loss', meters['evidence_loss'].avg, epoch)
            tb_writer.add_scalar('complementary_loss', meters['complementary_loss'].avg, epoch)
            tb_writer.add_scalar('cost_loss', meters['cost_loss'].avg, epoch)
            
            # Progressive training statistics
            if epoch_stats['num_confident_clean']:
                tb_writer.add_scalar('avg_confident_clean', np.mean(epoch_stats['num_confident_clean']), epoch)
                tb_writer.add_scalar('avg_confident_noisy', np.mean(epoch_stats['num_confident_noisy']), epoch)
                tb_writer.add_scalar('avg_uncertain', np.mean(epoch_stats['num_uncertain']), epoch)
                tb_writer.add_scalar('avg_rematch_pairs', np.mean(epoch_stats['num_rematch_pairs']), epoch)
            
            # Training phase info
            if 'training_phase' in ret:
                tb_writer.add_text('ccl_training_phase', ret['training_phase'], epoch)

        scheduler.step()
        
        # End of epoch timing
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
            
            # Phase statistics summary
            logger.info(f"Phase Statistics - {current_phase}:")
            if current_phase == 'warmup':
                logger.info(f"  Average Loss: {phase_statistics[current_phase]['avg_loss']:.4f}")
            elif current_phase == 'uncertainty_learning':
                logger.info(f"  Average Loss: {phase_statistics[current_phase]['avg_loss']:.4f}")
                logger.info(f"  Average Evidence Loss: {phase_statistics[current_phase].get('avg_evidence_loss', 0):.4f}")
                if epoch_stats['num_confident_clean']:
                    logger.info(f"  Avg Confident Clean: {np.mean(epoch_stats['num_confident_clean']):.1f}")
                    logger.info(f"  Avg Confident Noisy: {np.mean(epoch_stats['num_confident_noisy']):.1f}")
                    logger.info(f"  Avg Uncertain: {np.mean(epoch_stats['num_uncertain']):.1f}")
            else:
                logger.info(f"  Average Combined Loss: {phase_statistics[current_phase].get('avg_combined_loss', 0):.4f}")
                if epoch_stats['num_confident_clean']:
                    logger.info(f"  Avg Confident Clean: {np.mean(epoch_stats['num_confident_clean']):.1f}")
                    logger.info(f"  Avg Confident Noisy: {np.mean(epoch_stats['num_confident_noisy']):.1f}")
                    logger.info(f"  Avg Uncertain: {np.mean(epoch_stats['num_uncertain']):.1f}")
                    logger.info(f"  Avg Rematch Pairs: {np.mean(epoch_stats['num_rematch_pairs']):.1f}")

        # Evaluation
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                    logger.info(f"New best R1: {best_top1:.4f} at epoch {epoch}")

    # Final statistics summary
    if get_rank() == 0:
        logger.info(f"Training completed. Best R1: {best_top1:.4f} at epoch {arguments.get('epoch', 'N/A')}")
        
        # Print final phase statistics
        logger.info("Final Phase Statistics:")
        for phase, stats in phase_statistics.items():
            if stats['epochs'] > 0:
                logger.info(f"  {phase}: {stats['epochs']} epochs")
                if 'avg_loss' in stats:
                    logger.info(f"    Average Loss: {stats['avg_loss']:.4f}")
                if 'avg_evidence_loss' in stats:
                    logger.info(f"    Average Evidence Loss: {stats['avg_evidence_loss']:.4f}")
                if 'avg_combined_loss' in stats:
                    logger.info(f"    Average Combined Loss: {stats['avg_combined_loss']:.4f}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)


def enhanced_do_inference(model, test_img_loader, test_txt_loader):
    """
    Enhanced inference function compatible with enhanced RDE model
    """
    logger = logging.getLogger("Enhanced_RDE.test")
    logger.info("Enter enhanced inferencing")

    # Set model to inference mode
    if hasattr(model, 'inference_mode'):
        model.inference_mode()
    else:
        model.eval()

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model)
    
    # Get and log model statistics if available
    if hasattr(model, 'get_statistics'):
        stats = model.get_statistics()
        logger.info("Model Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    return top1


class EnhancedTrainingMonitor:
    """
    Enhanced training monitor for detailed tracking of training progress
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.phase_transitions = []
        self.loss_history = {
            'warmup': [],
            'uncertainty_learning': [],
            'progressive_training': []
        }
        
    def log_phase_transition(self, epoch, old_phase, new_phase):
        """Log phase transition"""
        transition = {
            'epoch': epoch,
            'from_phase': old_phase,
            'to_phase': new_phase,
            'timestamp': time.time()
        }
        self.phase_transitions.append(transition)
        
    def log_loss(self, epoch, phase, losses):
        """Log loss values for a phase"""
        self.loss_history[phase].append({
            'epoch': epoch,
            'losses': losses
        })
        
    def save_training_log(self):
        """Save comprehensive training log"""
        log_data = {
            'phase_transitions': self.phase_transitions,
            'loss_history': self.loss_history
        }
        
        import json
        log_file = os.path.join(self.log_dir, 'enhanced_training_log.json')
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
    def generate_training_report(self):
        """Generate a comprehensive training report"""
        report = []
        report.append("Enhanced RDE Training Report")
        report.append("=" * 50)
        
        # Phase transitions
        report.append("\nPhase Transitions:")
        for transition in self.phase_transitions:
            report.append(f"  Epoch {transition['epoch']}: {transition['from_phase']} -> {transition['to_phase']}")
            
        # Loss statistics
        report.append("\nLoss Statistics by Phase:")
        for phase, history in self.loss_history.items():
            if history:
                avg_losses = {}
                for entry in history:
                    for loss_name, loss_value in entry['losses'].items():
                        if loss_name not in avg_losses:
                            avg_losses[loss_name] = []
                        avg_losses[loss_name].append(loss_value)
                
                report.append(f"\n  {phase.title()}:")
                for loss_name, values in avg_losses.items():
                    avg_val = np.mean(values)
                    std_val = np.std(values)
                    report.append(f"    {loss_name}: {avg_val:.4f} ± {std_val:.4f}")
        
        return "\n".join(report) 