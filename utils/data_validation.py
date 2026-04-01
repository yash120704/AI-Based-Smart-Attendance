"""
Data validation and preprocessing utilities for behavior model training
Ensures data quality and consistency across training pipeline
"""
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_sequence(sequence):
    """
    Normalize each sequence: (sequence - mean) / (std + 1e-6)
    Applied per-sequence to ensure stable training
    
    Args:
        sequence: Array of shape (30, 99) or (30, 198)
    
    Returns:
        Normalized sequence of same shape
    """
    mean = np.mean(sequence)
    std = np.std(sequence)
    normalized = (sequence - mean) / (std + 1e-6)
    return normalized.astype(np.float32)


def validate_sequence(sequence, min_std=0.003):
    """
    Validate sequence quality: reject static/noisy data
    
    Args:
        sequence: Array of shape (30, 99) or (30, 198)
        min_std: Minimum standard deviation threshold (reject if below)
    
    Returns:
        True if sequence is valid, False otherwise
    """
    std = np.std(sequence)
    if std < min_std:
        logger.debug(f"Sequence rejected: std={std:.6f} < min_std={min_std:.6f}")
        return False
    return True


def validate_and_filter_sequences(sequences, min_std=0.003):
    """
    Filter sequences by quality threshold
    
    Args:
        sequences: Array of shape (n, 30, 99) or (n, 30, 198)
        min_std: Minimum standard deviation threshold
    
    Returns:
        valid_sequences: Filtered sequences
        rejection_count: Number of sequences rejected
    """
    valid_sequences = []
    rejection_count = 0
    
    for seq in sequences:
        if validate_sequence(seq, min_std):
            valid_sequences.append(seq)
        else:
            rejection_count += 1
    
    return np.array(valid_sequences) if valid_sequences else np.array([]), rejection_count


def print_data_distribution(sequences_dict, labels_dict):
    """
    Print dataset statistics and distribution
    
    Args:
        sequences_dict: Dictionary mapping person_name -> sequence array
        labels_dict: Dictionary mapping person_name -> label array
    """
    logger.info("\n" + "=" * 70)
    logger.info("DATA DISTRIBUTION ANALYSIS")
    logger.info("=" * 70)
    
    total_sequences = 0
    total_valid_sequences = 0
    total_rejected = 0
    
    distribution = {}
    
    for person_name in sorted(sequences_dict.keys()):
        sequences = sequences_dict[person_name]
        valid_seqs, rejected = validate_and_filter_sequences(sequences, min_std=0.003)
        
        distribution[person_name] = {
            'total': len(sequences),
            'valid': len(valid_seqs),
            'rejected': rejected,
            'percentage_valid': (len(valid_seqs) / len(sequences) * 100) if sequences.size > 0 else 0
        }
        
        total_sequences += len(sequences)
        total_valid_sequences += len(valid_seqs)
        total_rejected += rejected
    
    # Print per-person statistics
    for person_name, stats in distribution.items():
        logger.info(
            f"  {person_name:15} | "
            f"Total: {stats['total']:3d} | "
            f"Valid: {stats['valid']:3d} | "
            f"Rejected: {stats['rejected']:3d} | "
            f"Quality: {stats['percentage_valid']:6.1f}%"
        )
    
    # Print summary
    logger.info("-" * 70)
    logger.info(f"TOTAL: {total_sequences} sequences | Valid: {total_valid_sequences} | Rejected: {total_rejected}")
    
    if total_sequences > 0:
        avg_quality = (total_valid_sequences / total_sequences) * 100
        logger.info(f"AVERAGE DATA QUALITY: {avg_quality:.1f}%")
    
    # Check for class imbalance
    logger.info("\n" + "=" * 70)
    logger.info("CLASS BALANCE CHECK")
    logger.info("=" * 70)
    
    valid_counts = {name: stats['valid'] for name, stats in distribution.items()}
    min_count = min(valid_counts.values()) if valid_counts else 0
    max_count = max(valid_counts.values()) if valid_counts else 0
    
    if max_count > 0 and min_count > 0:
        imbalance_ratio = max_count / min_count
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 2.0:
            logger.warning(f"⚠️  HIGH IMBALANCE (ratio > 2.0). Consider re-collecting data to balance classes.")
        elif imbalance_ratio > 1.5:
            logger.warning(f"⚠️  MODERATE IMBALANCE (ratio > 1.5). Class weighting will be applied.")
        else:
            logger.info(f"✓ BALANCED dataset (ratio < 1.5)")
    
    logger.info("=" * 70 + "\n")
    
    return distribution


def check_dataset_consistency(sequences_dict, labels_dict):
    """
    Verify dataset consistency (sequences match labels)
    
    Args:
        sequences_dict: Dictionary mapping person_name -> sequence array
        labels_dict: Dictionary mapping person_name -> label array
    
    Returns:
        True if consistent, False otherwise
    """
    for person_name in sequences_dict.keys():
        if person_name not in labels_dict:
            logger.error(f"Missing labels for {person_name}")
            return False
        
        seq_count = len(sequences_dict[person_name])
        label_count = len(labels_dict[person_name])
        
        if seq_count != label_count:
            logger.error(f"{person_name}: sequences ({seq_count}) != labels ({label_count})")
            return False
    
    logger.info("✓ Dataset consistency verified")
    return True
