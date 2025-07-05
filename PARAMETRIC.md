# Parametric Models in NAM

This document describes the history and implementation of parametric models in Neural Amp Modeler (NAM), which existed from July 2022 to January 2024.

## Timeline

- **Added**: July 20, 2022 (commit `ced3ec4355568faa1b89ff722afdf2987f219be2`, tag: v0.3.0)
  - Initial implementation of parametric models
  - Added CatLSTM and HyperConvNet implementations
  - PR #39: "Version 0.3.0"

- **Removed**: January 14, 2024 (commit `090fd22bf60e89c34740ca356dae61eed19db873`)
  - Complete removal of parametric modeling code
  - Breaking change
  - PR #367: "[BREAKING] Remove parametric modeling code"

## Implementation Details

The parametric models came in two flavors:

### 1. CatNets (Concatenation Networks)

These models concatenated parameter inputs with the audio input at each time point. Two implementations existed:

#### CatLSTM
- Extended the base LSTM model
- Used RNN-style shapes `(batch, sequence_length, features)`
- Parameters were tiled across time and concatenated in feature dimension
- Required special handling for initial states and nominal settings

#### CatWaveNet
- Extended the base WaveNet model
- Used CNN-style shapes `(batch, channels, sequence_length)`
- Parameters were tiled across time and concatenated in channel dimension
- Simpler implementation due to WaveNet's existing convolutional architecture

Both used a shared `_CatMixin` class that handled:
- Parameter concatenation
- Model export
- Configuration management
- Weight serialization

### 2. HyperNets (Hypernetworks)

A more sophisticated approach where a hypernetwork generated the weights for the main network:

- `HyperConvNet`: Used a hypernetwork to generate parameters for a convolutional network
- Supported batch normalization and various activation functions
- Included comprehensive export functionality for C++ inference

## Code Structure

The parametric models lived in `nam/models/parametric/` with the following structure:

```
nam/models/parametric/
├── __init__.py
├── catnets.py      # CatLSTM and CatWaveNet implementations
├── hyper_net.py    # HyperConvNet implementation
└── params.py       # Parameter type definitions
```

## How to Replicate

If you want to experiment with the parametric model implementations:

1. Clone NAM and check out the last version before removal:
```bash
git clone https://github.com/sdatkinson/neural-amp-modeler.git
cd neural-amp-modeler
git checkout ae86979  # Last commit before removal
```

2. The parametric models are in `nam/models/parametric/`:
- `catnets.py` contains both CatLSTM and CatWaveNet
- `hyper_net.py` contains HyperConvNet

3. Key implementation details:
- CatNets concatenate parameters with input: `[audio, params]`
- Parameters are tiled across time to match audio length
- Different shapes for RNN vs CNN architectures
- Comprehensive export system for C++ inference

## Example Usage

```python
# CatLSTM example
model = CatLSTM(
    input_size=1,  # Mono audio
    hidden_size=32,
    num_layers=2,
    bidirectional=False
)

# Process audio with parameters
audio = torch.randn(1, 48000)  # (batch, samples)
params = torch.tensor([0.5, 0.3])  # (param_dimensions,)
output = model(params, audio)
```

## Why Were They Removed?

The parametric models were removed in January 2024 as part of a breaking change (PR #367). While the exact reasoning isn't documented in the commit message, this likely aligned with NAM's focus on single-snapshot modeling and the introduction of the multi-knob extension as a more flexible alternative for parameter-aware modeling.

## Alternative: Multi-Knob Extension

The current recommended approach for parameter-aware modeling is the multi-knob extension, which:
- Supports multiple control parameters
- Uses embedding layers for parameter conditioning
- Integrates with NAM's training pipeline
- Maintains backward compatibility with existing models

See the multi-knob extension documentation for more details on the current approach to parameter-aware modeling. 