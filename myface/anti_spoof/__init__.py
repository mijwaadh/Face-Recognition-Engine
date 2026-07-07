"""
Anti-spoofing module.
Computes depth consistency, Moiré frequencies, and chrominance diffusion check scores.
"""

from myface.anti_spoof.liveness import LivenessDetector

__all__ = ["LivenessDetector"]
