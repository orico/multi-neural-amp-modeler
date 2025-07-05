"""
Multi-knob extension for Neural Amp Modeler
Enables training models with multiple knob parameters
"""

from typing import Dict, Optional, Sequence, Tuple, Any, Union
import torch
import torch.nn as nn
import numpy as np
from nam.data import AbstractDataset, register_dataset_initializer
from nam.models.base import BaseNet
from nam.models._abc import ImportsWeights
from nam.models.metadata import UserMetadata
from nam.models.wavenet import WaveNet
from nam.train.lightning_module import LightningModule
from pydantic import BaseModel


# Metadata Extensions
class KnobMetadata(BaseModel):
    """Metadata for a single knob parameter"""
    name: str
    min_value: float
    max_value: float
    default_value: float
    units: Optional[str] = None


class MultiKnobUserMetadata(UserMetadata):
    """Extended metadata to include knob information"""
    knobs: Dict[str, KnobMetadata]


# Dataset Implementation
class MultiKnobDataset(AbstractDataset):
    """Dataset that handles both audio and knob parameter data"""
    
    def __init__(
        self,
        x: torch.Tensor,  # Input audio
        y: torch.Tensor,  # Output audio
        knob_settings: Dict[str, torch.Tensor],  # Knob values per sample
        nx: int,  # Receptive field
        ny: Optional[int] = None,  # Output samples per datum
        sample_rate: Optional[float] = None,
        delay: Optional[int] = 0,  # Common NAM parameter
        **kwargs  # Handle any other NAM parameters
    ):
        super().__init__()
        self._x = x
        self._y = y
        self._knob_settings = knob_settings
        self._nx = nx
        self._ny = ny if ny is not None else len(x) - nx + 1
        self._sample_rate = sample_rate
        
        # Apply delay if needed
        if delay:
            if delay > 0:
                self._x = torch.cat([torch.zeros(delay), self._x])
            else:
                self._y = torch.cat([torch.zeros(-delay), self._y])
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Get input audio, output audio, and knob settings for a segment"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            
        i = idx * self._ny
        j = i + self._nx - 1
        x_segment = self._x[i:i + self._nx + self._ny - 1]
        y_segment = self._y[j:j + self._ny]
        knob_segment = {k: v[i:i + self._nx + self._ny - 1] for k, v in self._knob_settings.items()}
        return x_segment, knob_segment, y_segment  # Return y_segment last

    def __len__(self) -> int:
        """Calculate number of segments in dataset"""
        n = len(self._x)
        single_pairs = n - self._nx + 1
        return single_pairs // self._ny

    @property
    def sample_rate(self) -> Optional[float]:
        """Get the sample rate of the audio data"""
        return self._sample_rate

    @property
    def nx(self) -> int:
        """Get the receptive field size"""
        return self._nx

    @property
    def ny(self) -> int:
        """Get the output length"""
        return self._ny

    @property
    def x(self) -> torch.Tensor:
        """Get the input audio data"""
        return self._x

    @property
    def y(self) -> torch.Tensor:
        """Get the output audio data"""
        return self._y

    @property
    def knob_settings(self) -> Dict[str, torch.Tensor]:
        """Get the knob settings"""
        return self._knob_settings

    @classmethod
    def init_from_config(cls, config):
        """Initialize dataset from configuration dictionary"""
        parsed_config = cls.parse_config(config)
        return cls(**parsed_config)

    @classmethod
    def parse_config(cls, config):
        """Parse configuration and convert file paths to tensors"""
        from nam.data import wav_to_tensor
        
        # Get sample rate from config
        sample_rate = config.pop("sample_rate", None)
        
        # Load audio files
        x = wav_to_tensor(config.pop("x_path"), rate=sample_rate)
        y = wav_to_tensor(config.pop("y_path"), rate=sample_rate)
        
        # Convert knob settings to tensors
        raw_knob_settings = config.pop("knob_settings", {})
        knob_settings = {}
        for knob_name, value in raw_knob_settings.items():
            if value is not None:
                # Create a tensor of the same length as x filled with the knob value
                knob_settings[knob_name] = torch.full((len(x),), value, dtype=torch.float32)
        
        # Get other parameters
        nx = config.pop("nx", None)
        if nx is None:
            # If nx not provided, try to get it from the model's receptive field
            from nam.train.lightning_module import _model_net_init_registry
            model_config = config.get("model_config", {})
            if model_config:
                model = _model_net_init_registry[model_config["name"]](model_config["config"])
                nx = model.receptive_field
            else:
                nx = 8192  # Default value
        
        # Return all parameters including common NAM parameters
        return {
            "x": x,
            "y": y,
            "knob_settings": knob_settings,
            "nx": nx,
            "ny": config.pop("ny", None),
            "sample_rate": sample_rate,
            "delay": config.pop("delay", 0),  # Include delay parameter
            **config  # Pass through any remaining parameters
        }


# Model Implementation
class MultiKnobModel(BaseNet, ImportsWeights):
    """Neural network model that supports multiple conditioning knobs"""
    
    def __init__(
        self,
        knob_config: Dict[str, Dict[str, Any]],
        base_model: Union[str, BaseNet] = "WaveNet",
        sample_rate: Optional[float] = None,
        **kwargs
    ):
        """Initialize the multi-knob model
        
        Args:
            knob_config: Dictionary mapping knob names to their configurations
                Each knob config should have:
                - embedding_dim: int, dimension of the embedding
                - default_value: float, value to use when knob is not provided
            base_model: Either a string naming the base model type (e.g. "WaveNet")
                       or an instance of a BaseNet model
            sample_rate: Optional sample rate for the model
        """
        super().__init__(sample_rate=sample_rate)
        self.knob_config = knob_config
        
        # Define knob order (fixed order for consistency)
        self.knob_order = sorted(knob_config.keys())  # Use sorted keys for deterministic order
        
        # Create embedding layers for each knob
        self.knob_embeddings = nn.ModuleDict()
        for name, config in knob_config.items():
            self.knob_embeddings[name] = nn.Linear(
                in_features=1,  # Single knob value
                out_features=config['embedding_dim'],  # Embed into higher dimension
                bias=True
            )
            
        # Calculate total embedding dimension for conditioning
        total_embedding_dim = sum(config['embedding_dim'] for config in knob_config.values())
        
        # Initialize base model
        if isinstance(base_model, str):
            if base_model == "WaveNet":
                channels = 32  # Match WaveNet's internal channel count
                head_size = channels // 2  # Half channels for head
                self.base_model = WaveNet(
                    layers_configs=[{
                        "input_size": 1,  # Single input channel
                        "condition_size": total_embedding_dim,  # Total embedding dimension
                        "channels": channels,
                        "head_size": head_size,  # Half channels for head
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": True,  # Enable gated activations
                        "head_bias": True,  # Enable head bias
                    }],
                    head_config={
                        "in_channels": head_size,  # Match head_size from layers
                        "channels": head_size,  # Same size for internal layers
                        "activation": "Tanh",  # Match layer activation
                        "num_layers": 2,  # Two layers for sufficient capacity
                        "out_channels": 1,  # Single output channel
                    },
                    head_scale=0.02,  # Scale factor for the output
                    sample_rate=sample_rate
                )
            else:
                raise ValueError(f"Unknown base model: {base_model}")
        else:
            self.base_model = base_model
            
        # Store the receptive field for padding
        self._receptive_field = self.base_model.receptive_field
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass implementation required by base class"""
        # Extract pad_start from kwargs
        pad_start = kwargs.pop('pad_start', None)
        if pad_start is None:
            pad_start = self.pad_start_default
        
        # Add padding if needed
        if pad_start:
            # Add padding for WaveNet's receptive field
            pad_length = self.receptive_field - 1
            # Ensure x has channel dimension
            if x.ndim == 2:
                x = x.unsqueeze(1)
            elif x.ndim == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            # Add padding with correct dimensions
            x = torch.cat(
                (torch.zeros((len(x), x.shape[1], pad_length)).to(x.device), x),
                dim=2,
            )
        
        # Check if we have enough samples
        if x.shape[-1] < self.receptive_field:
            raise ValueError(f"Input has {x.shape[-1]} samples, which is too few for this model with receptive field {self.receptive_field}!")
        
        # Process the input through our _forward
        output = self._forward(x, *args, **kwargs)
        
        # Squeeze if input was scalar
        if x.ndim == 1:
            output = output[0]
            
        return output

    def _forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Internal forward pass implementation"""
        # Process knob values
        embedded_knobs = []
        for name in self.knob_order:
            # Get knob value from kwargs or use default
            value = kwargs.get(name, None)
            if value is not None:
                # Convert to tensor if needed
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value).to(x.device)
                # Expand scalar value to match input dimensions
                if value.ndim == 0:
                    value = value.expand(x.shape[0], x.shape[-1])
                # Ensure value has batch dimension
                elif value.ndim == 1:
                    value = value.unsqueeze(0)
            else:
                # Use default value
                default_value = self.knob_config[name]['default_value']
                value = torch.full((x.shape[0], x.shape[-1]), default_value).to(x.device)
            
            # Add channel dimension for embedding
            value = value.unsqueeze(1)  # [B, 1, L]
            # Embed the knob value
            embedded = self.knob_embeddings[name](value.transpose(1, 2))  # [B, L, E]
            embedded = embedded.transpose(1, 2)  # [B, E, L]
            # Ensure embedded has the same length as input by interpolating
            if embedded.shape[-1] != x.shape[-1]:
                embedded = torch.nn.functional.interpolate(
                    embedded,
                    size=x.shape[-1],
                    mode='nearest'
                )
            embedded_knobs.append(embedded)
        
        # Combine all knob embeddings
        combined_conditioning = torch.cat(embedded_knobs, dim=1)  # [B, total_E, L]
        
        # Ensure x has channel dimension
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        # Pass through base model's internal layers
        if isinstance(self.base_model, WaveNet):
            # Get the WaveNet's internal layers
            wavenet = self.base_model._net
            # Process through each layer group with conditioning
            x_out = x
            head_input = None
            for layer_group in wavenet._layers:
                head_input, x_out = layer_group(x_out, combined_conditioning, head_input=head_input)
            
            # Apply head scale
            head_input = wavenet._head_scale * head_input
            
            # Apply head if it exists, otherwise use head input
            x_out = wavenet._head(head_input) if wavenet._head is not None else head_input
            
            # Ensure output has correct shape [batch_size, num_samples]
            assert x_out.shape[1] == 1, f"Expected output to have 1 channel, got {x_out.shape[1]}"
            x_out = x_out[:, 0, :]  # Remove channel dimension
            
            return x_out
        else:
            raise NotImplementedError("Only WaveNet base model is currently supported")

    def import_weights(self, weights: Sequence[float]):
        """Import weights for knob embeddings and base model"""
        offset = 0
        
        # Import knob embedding weights
        for name, embedding in self.knob_embeddings.items():
            num_params = embedding.weight.numel() + embedding.bias.numel()
            layer_weights = torch.tensor(weights[offset:offset + num_params])
            
            # Split into weight and bias
            weight_size = embedding.weight.numel()
            embedding.weight.data = layer_weights[:weight_size].reshape(embedding.weight.shape)
            embedding.bias.data = layer_weights[weight_size:].reshape(embedding.bias.shape)
            offset += num_params
            
        # Import base model weights
        self.base_model.import_weights(weights[offset:])

    @property
    def pad_start_default(self) -> bool:
        """Default value for pad_start parameter"""
        return self.base_model.pad_start_default
        
    @property
    def receptive_field(self) -> int:
        """Receptive field of the model"""
        return self._receptive_field

    def _export_config(self):
        """Export model configuration"""
        return {
            "knob_config": self.knob_config,
            "base_model": self.base_model.__class__.__name__,
            "sample_rate": self.sample_rate
        }

    def _export_weights(self) -> np.ndarray:
        """Export weights for knob embeddings and base model"""
        weights = []
        
        # Export knob embedding weights
        for embedding in self.knob_embeddings.values():
            weights.extend(embedding.weight.data.cpu().numpy().flatten())
            weights.extend(embedding.bias.data.cpu().numpy().flatten())
        
        # Export base model weights
        weights.extend(self.base_model._export_weights())
        
        return np.array(weights)

    @classmethod
    def init_from_config(cls, config):
        """Initialize model from configuration dictionary"""
        return cls(**config)


# Register extensions with NAM
register_dataset_initializer("multi_knob", MultiKnobDataset.init_from_config)

# The model will be registered when imported by NAM's extension system
LightningModule.register_net_initializer(
    "MultiKnob",
    MultiKnobModel.init_from_config,
    overwrite=True  # Allow overwriting existing registration
) 