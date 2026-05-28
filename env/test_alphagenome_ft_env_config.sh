#!/bin/bash
# Script to check JAX and CUDA config on partition

echo "Checking JAX and CUDA setup..."

python -c '
import os
import ctypes.util

print("--- Environment Variables ---")
for var in ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"]:
    val = os.environ.get(var, "Not set")
    print(f"{var}: {val}")

print("\n--- CUDA Libraries ---")
for lib in ["cudart", "cublas", "cufft", "cusparse", "cudnn"]:
    print(f"{lib}: {ctypes.util.find_library(lib)}")

print("\n--- JAX Configuration ---")
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    
    # Check default backend
    backend = jax.default_backend()
    print(f"JAX default backend: {backend}")
    
    # List devices
    devices = jax.devices()
    print(f"Total JAX devices found: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"Device {i}: {device.device_kind} (ID: {device.id})")
        
    if backend == "gpu":
        import jax.numpy as jnp
        x = jnp.ones((10, 10))
        y = jnp.dot(x, x)
        print("\nSUCCESS: JAX tensor operation completed on GPU.")
    else:
        print("\nWARNING: JAX is not using the GPU backend.")
        
except ImportError as e:
    print(f"\nERROR: Could not import JAX. {e}")
except Exception as e:
    print(f"\nERROR during JAX device check: {e}")
'