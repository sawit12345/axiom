# Axiom CUDA Backend Python Bindings

This directory contains pybind11-based Python bindings for the axiomcuda C++/CUDA backend.

## Files

### Core Binding Files

1. **bindings.cpp** - Main pybind11 module entry point
   - Creates `axiomcuda_backend` module
   - Registers submodules: device, tensor, math, distributions, transforms, models
   - Provides module-level functions: `cuda_available()`, `get_cuda_version()`

2. **bindings.h** - Common header for bindings
   - Type casters for numpy interop
   - Helper macros for binding classes
   - Docstring utilities

3. **tensor_bindings.cpp** - Tensor and array operations
   - Exposes `Tensor` class with CPU/GPU support
   - numpy array conversion utilities
   - Linear algebra operations
   - Memory management (MemoryPool, CudaStream)

4. **distribution_bindings.cpp** - Probability distributions
   - **Base classes**: Distribution, ExponentialFamily, Conjugate
   - **Exponential family**: MultivariateNormal, Multinomial, Delta
   - **Conjugate priors**: MultivariateNormalConjugate (NIW), MultinomialConjugate (Dirichlet)
   - Helper class: ArrayDict

5. **transform_bindings.cpp** - Variational inference transforms
   - Base Transform class
   - LinearMatrixNormalGamma (linear transformation with MNG prior)
   - Message passing methods (forward, backward, variational)

6. **model_bindings.cpp** - Probabilistic models
   - **SMM**: Slot Mixture Model with EM algorithm
   - **RMM**: Relational Mixture Model for action prediction
   - **TMM**: Transition Mixture Model for dynamics
   - **IMM**: Identity Mixture Model for object classification
   - **Mixture**: Generic mixture models
   - **HybridMixture**: Mixtures with continuous and discrete components

### Build Configuration

7. **CMakeLists.txt** - CMake build configuration
   - Finds CUDA and pybind11
   - Sets compilation flags (-O3, -arch=sm_70, etc.)
   - Creates static and shared libraries
   - Builds Python bindings
   - Supports sm_70, sm_75, sm_80, sm_86, sm_90 architectures

8. **setup.py** - Python setup script
   - Uses setuptools with CMakeExtension
   - Auto-detects CUDA installation
   - Handles different CUDA versions
   - Configures parallel builds

## Building

### Prerequisites

- CMake >= 3.18
- CUDA Toolkit >= 11.0
- Python >= 3.8
- pybind11 (pip install pybind11)
- numpy, scipy

### Build Steps

```bash
# From the axiom repository root
pip install src/

# Or manually with CMake
cd src
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j4

# Install the built module
cd ..
pip install . --no-build-isolation
```

### Environment Variables

- `CUDA_HOME`: Path to CUDA installation
- `TORCH_CUDA_ARCH_LIST`: CUDA architectures to compile for (e.g., "7.0;7.5;8.0")
- `CMAKE_BUILD_PARALLEL_LEVEL`: Parallel build jobs

## Usage

```python
import axiomcuda_backend as ac

# Device management
if ac.cuda_available():
    print(f"GPUs: {ac.device.get_device_count()}")
    ac.device.set_device(0)

# Tensor operations
from axiomcuda_backend import tensor
t = tensor.randn((100, 50))
t_gpu = t.cuda()
result = tensor.matmul(t_gpu, t_gpu.T)

# Distributions
from axiomcuda_backend import distributions
mvn = distributions.MultivariateNormal(
    event_shape=(10, 1),
    batch_shape=(5,)
)
samples = mvn.sample(key=None, shape=(100,))

# Models
from axiomcuda_backend import models
config = models.SMMConfig(num_slots=32, slot_dim=2)
smm = models.SMM.create(config)

# Process image
qx, qz, used = smm.infer_and_update(
    inputs, qx_prev, num_slots=32
)
```

## Features

- **Complete API Coverage**: All core C++/CUDA functionality exposed
- **Numpy Interop**: Seamless conversion between numpy arrays and Tensors
- **GPU/CPU Flexibility**: Automatic device management with explicit control
- **Type Safety**: Full type information for IDE support
- **Documentation**: Docstrings for all classes and methods
- **Error Handling**: C++ exceptions translated to Python exceptions

## Architecture

The bindings follow a hierarchical structure:

```
axiomcuda_backend (module)
├── device (submodule)
│   ├── get_device_count()
│   ├── set_device()
│   └── DeviceProperties
├── tensor (submodule)
│   ├── Tensor (class)
│   ├── zeros(), ones(), randn()
│   ├── matmul(), inverse(), svd()
│   └── MemoryPool, CudaStream
├── math (submodule)
│   ├── gammaln(), digamma()
│   ├── logsumexp(), softmax()
│   └── gaussian_loglik()
├── distributions (submodule)
│   ├── Distribution (base)
│   ├── ExponentialFamily (base)
│   ├── MultivariateNormal, Multinomial, Delta
│   ├── Conjugate (base)
│   ├── MultivariateNormalConjugate (NIW)
│   └── MultinomialConjugate (Dirichlet)
├── transforms (submodule)
│   ├── Transform (base)
│   └── LinearMatrixNormalGamma
└── models (submodule)
    ├── SMM, RMM, TMM, IMM
    └── Mixture, HybridMixture
```

## Notes

- The bindings assume corresponding C++ headers exist in `src/` subdirectories
- All numpy arrays are automatically converted to the appropriate device
- Memory is managed automatically with RAII
- CUDA streams are used for async operations where beneficial
