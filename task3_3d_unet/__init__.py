"""Conditional 3D U-Net experiment for MRIxFields2026 Task 3.

The package initializer stays dependency-light so data audits can run before
the training environment is installed.
"""

from .common import FIELDS, MODALITIES, directed_pairs, joint_domain

__all__ = ["FIELDS", "MODALITIES", "directed_pairs", "joint_domain"]
