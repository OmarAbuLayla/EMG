"""Dataset helpers for the transformer-based 15-channel EMG model.

This module simply re-exports the improved data pipeline introduced for the
CNN-GRU trainer so both back-ends share the exact same preprocessing,
normalisation, and caching behaviour.
"""

from emg15_dataset import EMGDataset15, MFSCConfig, build_dataloaders

__all__ = ["EMGDataset15", "MFSCConfig", "build_dataloaders"]
