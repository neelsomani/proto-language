"""
Compute and GPU utilities for proto-language.

This module provides utilities for managing compute resources,
including local GPU and cloud GPU selection.
"""

import os


def use_cloud_gpu() -> bool:
    """
    Smart GPU selection: try local GPU first, fall back to cloud.
    
    Returns:
        bool: True if should use cloud, False if should use local GPU.
        
    Environment Variables:
        USE_CLOUD: Set to "true" to force cloud, "false" to force local
                   If not set, automatically chooses based on GPU availability
    """
    # Check if user explicitly set preference
    use_cloud_env = os.getenv("USE_CLOUD")
    if use_cloud_env is not None:
        return use_cloud_env.lower() == "true"
    
    # Auto-detect: try local GPU first, fall back to cloud
    if _is_local_gpu_available():
        return False
    elif _is_cloud_available():
        print("Local GPU not available, falling back to cloud")
        return True
    else:
        raise RuntimeError(
            "Neither local GPU nor cloud is available. "
            "Please either:\n"
            "1. Ensure you have CUDA available locally\n"
            "2. Set up cloud (cloud token new)\n"
            "3. Set USE_CLOUD=true to force cloud execution"
        )


def _is_local_gpu_available() -> bool:
    """Check if local GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_cloud_available() -> bool:
    """Check if cloud is available and configured."""
    try:
        import cloud
        # Try creating a simple app to test authentication
        cloud.App('test-auth')
        return True
    except (ImportError, Exception) as e:
        print(f"cloud not available: {e}")
        return False


def is_gpu_available() -> bool:
    """Check if any GPU is available (local CUDA or cloud)."""
    return _is_local_gpu_available() or _is_cloud_available()

