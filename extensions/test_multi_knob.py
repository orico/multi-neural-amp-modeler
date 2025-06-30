"""Tests for the multi_knob extension"""

import pytest
import torch
import numpy as np
import os
from nam.models.metadata import UserMetadata
from extensions.multi_knob import (
    KnobMetadata,
    MultiKnobUserMetadata,
    MultiKnobDataset,
    MultiKnobModel
)

# Test KnobMetadata
def test_knob_metadata_creation():
    """Test creating KnobMetadata with valid parameters"""
    knob = KnobMetadata(
        name="gain",
        min_value=0.0,
        max_value=10.0,
        default_value=5.0,
        units="dB"
    )
    assert knob.name == "gain"
    assert knob.min_value == 0.0
    assert knob.max_value == 10.0
    assert knob.default_value == 5.0
    assert knob.units == "dB"

def test_knob_metadata_optional_units():
    """Test creating KnobMetadata without units"""
    knob = KnobMetadata(
        name="gain",
        min_value=0.0,
        max_value=10.0,
        default_value=5.0
    )
    assert knob.units is None

# Test MultiKnobUserMetadata
def test_multi_knob_user_metadata():
    """Test creating MultiKnobUserMetadata with knobs"""
    knobs = {
        "gain": KnobMetadata(
            name="gain",
            min_value=0.0,
            max_value=10.0,
            default_value=5.0,
            units="dB"
        ),
        "tone": KnobMetadata(
            name="tone",
            min_value=-5.0,
            max_value=5.0,
            default_value=0.0
        )
    }
    metadata = MultiKnobUserMetadata(knobs=knobs)
    assert len(metadata.knobs) == 2
    assert "gain" in metadata.knobs
    assert "tone" in metadata.knobs

# Test MultiKnobDataset
class TestMultiKnobDataset:
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing"""
        x = torch.randn(1000)
        y = torch.randn(1000)
        knob_settings = {
            "gain": torch.full((1000,), 5.0),
            "tone": torch.full((1000,), 0.0)
        }
        return MultiKnobDataset(
            x=x,
            y=y,
            knob_settings=knob_settings,
            nx=100,
            ny=50,
            sample_rate=44100
        )

    def test_dataset_creation(self, sample_dataset):
        """Test basic dataset creation"""
        assert isinstance(sample_dataset, MultiKnobDataset)
        assert sample_dataset.sample_rate == 44100
        assert sample_dataset.nx == 100
        assert sample_dataset.ny == 50

    def test_dataset_length(self, sample_dataset):
        """Test dataset length calculation"""
        expected_length = (1000 - 100 + 1) // 50
        assert len(sample_dataset) == expected_length

    def test_dataset_getitem(self, sample_dataset):
        """Test getting items from dataset"""
        x_segment, knob_segment, y_segment = sample_dataset[0]
        assert isinstance(x_segment, torch.Tensor)
        assert isinstance(knob_segment, dict)
        assert isinstance(y_segment, torch.Tensor)
        assert x_segment.shape[0] == 149  # nx + ny - 1
        assert y_segment.shape[0] == 50  # ny
        assert all(k in knob_segment for k in ["gain", "tone"])

    def test_dataset_out_of_bounds(self, sample_dataset):
        """Test accessing out of bounds index"""
        with pytest.raises(IndexError):
            _ = sample_dataset[len(sample_dataset)]

# Test MultiKnobModel
class TestMultiKnobModel:
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        knob_config = {
            "gain": {
                "embedding_dim": 8,
                "default_value": 5.0
            },
            "tone": {
                "embedding_dim": 8,
                "default_value": 0.0
            }
        }
        return MultiKnobModel(
            knob_config=knob_config,
            base_model="WaveNet",
            sample_rate=44100
        )

    def test_model_creation(self, sample_model):
        """Test basic model creation"""
        assert isinstance(sample_model, MultiKnobModel)
        assert len(sample_model.knob_embeddings) == 2
        assert sample_model.sample_rate == 44100

    def test_model_forward(self, sample_model):
        """Test model forward pass"""
        batch_size = 2
        input_length = sample_model.receptive_field + 100
        x = torch.randn(batch_size, 1, input_length)  # Add channel dimension
        # Knob values should be [batch_size, length]
        gain = torch.full((batch_size, input_length), 5.0)
        tone = torch.full((batch_size, input_length), 0.0)

        # Test with explicit knob values
        output = sample_model(x, gain=gain, tone=tone)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, input_length)

        # Test with scalar knob values
        output = sample_model(x, gain=5.0, tone=0.0)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, input_length)

    def test_model_forward_default_knobs(self, sample_model):
        """Test model forward pass with default knob values"""
        batch_size = 2
        input_length = sample_model.receptive_field + 100
        x = torch.randn(batch_size, input_length)

        output = sample_model(x)  # No knob values provided
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, input_length)

    def test_model_export_config(self, sample_model):
        """Test exporting model configuration"""
        config = sample_model._export_config()
        assert "knob_config" in config
        assert "base_model" in config
        assert "sample_rate" in config

    def test_model_export_weights(self, sample_model):
        """Test exporting model weights"""
        weights = sample_model._export_weights()
        assert isinstance(weights, np.ndarray)
        assert weights.ndim == 1  # Should be flattened

    @pytest.mark.xfail(reason="Head importing not implemented yet")
    def test_model_import_weights(self, sample_model):
        """Test importing model weights"""
        original_weights = sample_model._export_weights()
        sample_model.import_weights(original_weights)
        new_weights = sample_model._export_weights()
        np.testing.assert_array_almost_equal(original_weights, new_weights)

    def test_model_receptive_field(self, sample_model):
        """Test model receptive field property"""
        assert isinstance(sample_model.receptive_field, int)
        assert sample_model.receptive_field > 0

    def test_model_pad_start_default(self, sample_model):
        """Test pad_start_default property"""
        assert isinstance(sample_model.pad_start_default, bool)

    def test_insufficient_samples(self, sample_model):
        """Test handling of input with insufficient samples"""
        # Create input with insufficient samples (before padding)
        x = torch.randn(2, 1, sample_model.receptive_field - 1)  # Too few samples, with channel dim
        # Create knob values with matching length
        gain = torch.full((2, sample_model.receptive_field - 1), 5.0)
        tone = torch.full((2, sample_model.receptive_field - 1), 0.0)
        
        with pytest.raises(ValueError, match=r".*too few.*"):
            _ = sample_model(x, gain=gain, tone=tone, pad_start=False)  # Disable padding to force error

# Test dataset initialization from config
def test_dataset_init_from_config(tmp_path):
    """Test initializing dataset from configuration"""
    # Create temporary wav files
    x_path = os.path.join(tmp_path, "input.wav")
    y_path = os.path.join(tmp_path, "output.wav")
    
    # Save random audio as wav files
    import soundfile as sf
    sample_rate = 44100
    duration = 1.0  # seconds
    samples = int(sample_rate * duration)
    
    x_data = np.random.randn(samples)
    y_data = np.random.randn(samples)
    
    sf.write(x_path, x_data, sample_rate)
    sf.write(y_path, y_data, sample_rate)
    
    # Create config
    config = {
        "x_path": x_path,
        "y_path": y_path,
        "sample_rate": sample_rate,
        "nx": 100,
        "ny": 50,
        "knob_settings": {
            "gain": 5.0,
            "tone": 0.0
        }
    }
    
    # Initialize dataset
    dataset = MultiKnobDataset.init_from_config(config)
    
    # Verify dataset
    assert isinstance(dataset, MultiKnobDataset)
    assert dataset.sample_rate == sample_rate
    assert dataset.nx == 100
    assert dataset.ny == 50
    assert len(dataset._knob_settings) == 2
    assert "gain" in dataset._knob_settings
    assert "tone" in dataset._knob_settings 