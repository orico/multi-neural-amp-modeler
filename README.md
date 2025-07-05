# NAM Multi-Knob Extension

This extension adds support for training Neural Amp Models with multiple knob parameters. It enables modeling of amps and effects with multiple controls (like gain, tone, volume, etc.) in a single model.

## Installation

1. First, install NAM in development mode:
```bash
git clone https://github.com/sdatkinson/neural-amp-modeler.git
cd neural-amp-modeler
pip install -e .
```

2. Create the extensions directory in your home folder if it doesn't exist:
```bash
mkdir -p ~/.neural-amp-modeler/extensions
```

3. Copy the extension files to your NAM extensions directory:
```bash
cp extensions/multi_knob.py ~/.neural-amp-modeler/extensions/
```

## Delay Compensation

The multi-knob extension includes a delay compensation mechanism to handle timing differences between input and output audio files. This is particularly important when recording through real amplifiers where there might be latency in the signal chain.

### How Delay Works

1. **Delay Parameter**: The `delay` parameter in the dataset configuration represents the length difference between input and output signals:
   - If delay > 0: Input is longer than output by 'delay' samples
     - Zeros are added at the beginning of the output signal to align with input
   - If delay < 0: Output is longer than input by 'delay' samples
     - Zeros are added at the beginning of the input signal to align with output

2. **Calculating Delay**: Use the provided `analyze_wav.py` script to determine the correct delay values:
   ```bash
   python extensions/multi_knob/analyze_wav.py
   ```
   The script will analyze your audio files and output the length differences, which you can use as delay values.

3. **Example**:
   If your input.wav is 210,477 samples and output.wav is 208,943 samples:
   - Difference = 210,477 - 208,943 = 1,534 samples
   - Use delay = 1,534 in your configuration
   - This will add 1,534 zeros at the start of the output to align it with the input
   - The alignment ensures that the actual audio content of both signals lines up correctly

### Sample Rate Considerations

The delay is specified in samples, not milliseconds. To convert between them:
- Samples = Milliseconds * (Sample Rate / 1000)
- Milliseconds = Samples * (1000 / Sample Rate)

For example, at 44.1kHz:
- 1,534 samples ≈ 34.78ms
- 1ms ≈ 44.1 samples

### Alignment Logic

The delay compensation works by padding the shorter signal with zeros at its beginning to match the length of the longer signal. This ensures that:
1. Both signals end up with the same length
2. The actual audio content is properly aligned
3. No audio data is lost in the process

## Configuration Files

You'll need three JSON configuration files to train a multi-knob model:

### 1. Model Configuration (multi_knob_config.json)
```json
{
  "net": {
    "name": "MultiKnob",
    "config": {
      "knob_config": {
        "volume": {
          "name": "Volume",
          "min_value": 0.0,
          "max_value": 1.0,
          "default_value": 0.5,
          "embedding_dim": 8
        },
        "gain": {
          "name": "Gain",
          "min_value": 0.0,
          "max_value": 1.0,
          "default_value": 0.5,
          "embedding_dim": 8
        }
      },
      "base_model": "WaveNet",
      "sample_rate": 48000
    }
  }
}
```

### 2. Dataset Configuration (multi_knob_dataset.json)
```json
{
  "type": "multi_knob",
  "train": {
    "x_path": "path/to/input.wav",
    "y_path": "path/to/output.wav",
    "knob_settings": {
      "volume": 0.5,
      "gain": 0.7
    },
    "ny": 8192,
    "delay": 1534  // Length difference in samples (positive = input longer, negative = output longer)
  },
  "validation": {
    "x_path": "path/to/val_input.wav",
    "y_path": "path/to/val_output.wav",
    "knob_settings": {
      "volume": 0.3,
      "gain": 0.8
    }
  },
  "common": {
    "sample_rate": 48000
  }
}
```

### 3. Learning Configuration (multi_knob_learning.json)
```json
{
  "train_dataloader": {
    "batch_size": 16,
    "shuffle": true,
    "pin_memory": true,
    "drop_last": true,
    "num_workers": 0
  },
  "val_dataloader": {},
  "trainer": {
    "accelerator": "cpu",
    "devices": 1,
    "max_epochs": 30
  },
  "trainer_fit_kwargs": {}
}
```

## Training

To train a multi-knob model, use the `nam-full` command with your configuration files:

```bash
nam-full \
  path/to/multi_knob_dataset.json \
  path/to/multi_knob_config.json \
  path/to/multi_knob_learning.json \
  path/to/output_directory
```

## Features

- Support for multiple continuous knob parameters with customizable ranges
- Knob parameter embedding for better generalization
- Compatible with NAM's WaveNet architecture
- Preserves NAM's export format for plugin compatibility
- Full metadata support for knob ranges and defaults
- Comprehensive test suite ensuring reliability

## Testing

The extension includes a comprehensive test suite in `tests/test_nam/test_models/test_multi_knob.py` that verifies:

- Knob metadata handling
- Dataset creation and manipulation
- Model creation and forward pass
- Weight export/import functionality
- Input validation and error handling
- Default knob value handling
- Model configuration export

To run the tests:
```bash
pytest tests/test_nam/test_models/test_multi_knob.py -v
```

## Requirements

- NAM version 0.9 or later
- PyTorch 2.2.2
- Numpy 1.24.0
- Python 3.10