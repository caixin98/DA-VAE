import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.utils.checkpoint import checkpoint
from .convnext import convnextv2_base, convnextv2_nano
from .discriminator import Conv2dSame
import functools

convnextv2_base_pretrained_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt"
convnextv2_nano_pretrained_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt"


class ConvNeXtV2Discriminator(nn.Module):
    """ConvNeXtV2-based discriminator for image generation tasks.
    
    This discriminator uses ConvNeXtV2 models as the backbone and adapts it
    for discriminator tasks by removing the classification head and adding
    appropriate output layers for real/fake classification.
    """
    
    def __init__(
        self,
        num_channels: int = 3,
        model_size: str = "base",  # "base" or "nano"
        use_pretrained: bool = True,
        use_pretrained_weights: bool = True,  # 新增参数控制是否使用预训练权重
        use_checkpointing: bool = True,
        output_size: int = 1
    ):
        """Initialize the ConvNeXtV2 discriminator.
        
        Args:
            num_channels: Number of input channels (default: 3 for RGB)
            model_size: Size of ConvNeXtV2 model ("base" or "nano", default: "base")
            use_pretrained: Whether to load pretrained weights (default: True)
            use_pretrained_weights: Whether to use pretrained weights (default: True)
            use_checkpointing: Whether to use gradient checkpointing (default: True)
            output_size: Size of output (default: 1 for binary classification)
        """
        super().__init__()
        
        self.model_size = model_size
        
        # Initialize ConvNeXtV2 model based on size
        if model_size == "base":
            self.backbone = convnextv2_base(
                in_chans=num_channels,
                num_classes=1000  # Will be replaced
            )
            feature_dim = 1024  # ConvNeXtV2 base has 1024 features
            self.pretrained_url = convnextv2_base_pretrained_url
        elif model_size == "nano":
            self.backbone = convnextv2_nano(
                in_chans=num_channels,
                num_classes=1000  # Will be replaced
            )
            feature_dim = 640  # ConvNeXtV2 nano has 640 features
            self.pretrained_url = convnextv2_nano_pretrained_url
        else:
            raise ValueError(f"Unsupported model_size: {model_size}. Use 'base' or 'nano'.")
        
        # We'll use the backbone's forward_features method directly
        # and handle the classification in our discriminator head
        activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)
        del self.backbone.head
        # Add discriminator-specific layers
        self.to_logits = torch.nn.Sequential(
            Conv2dSame(feature_dim, feature_dim // 2, 1),
            activation(),
            Conv2dSame(feature_dim // 2, 1, kernel_size=1)
        )   

        
        self.use_checkpointing = use_checkpointing
        
        # Load pretrained weights if requested
        if use_pretrained and use_pretrained_weights:
            self._load_pretrained_weights()
        else:
            print(f"ConvNeXtV2 discriminator initialized without pretrained weights (use_pretrained={use_pretrained}, use_pretrained_weights={use_pretrained_weights})")
        
        # Initialize discriminator head weights
        self._init_discriminator_weights()
    
    def _load_pretrained_weights(self):
        """Load pretrained weights from the official ConvNeXtV2 model."""
        try:
            state_dict = load_state_dict_from_url(
                self.pretrained_url,
                map_location='cpu',
                progress=True
            )
            
            # Handle the case where weights are wrapped under 'model' key
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Filter out the classification head weights
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('head.'):
                    filtered_state_dict[key] = value
            
            # Load the filtered weights
            missing_keys, unexpected_keys = self.backbone.load_state_dict(
                filtered_state_dict, strict=False
            )
            
            print(f"Loaded pretrained ConvNeXtV2 {self.model_size} weights")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Continuing with random initialization...")
    
    def _init_discriminator_weights(self):
        """Initialize the discriminator head weights."""
        for module in self.to_logits.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward_features(self, x):
        """Extract features from the backbone."""
        if self.use_checkpointing:
            return checkpoint(
                self.backbone.forward_features, x, use_reentrant=False
            )
        else:
            return self.backbone.forward_features(x)
    
    def forward(self, x):
        """Forward pass through the discriminator.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            logits: Output logits of shape (B, 1, h, w)
        """
        # Extract features
        features = self.forward_features(x)
        # Pass through discriminator head
        logits = self.to_logits(features)
        
        return logits


def convnextv2_discriminator_base(**kwargs):
    """Factory function to create a ConvNeXtV2 base discriminator.
    
    Args:
        **kwargs: Arguments to pass to ConvNeXtV2Discriminator
        
    Returns:
        ConvNeXtV2Discriminator: Configured discriminator model
    """
    return ConvNeXtV2Discriminator(model_size="base", **kwargs)


def convnextv2_discriminator_nano(**kwargs):
    """Factory function to create a ConvNeXtV2 nano discriminator.
    
    Args:
        **kwargs: Arguments to pass to ConvNeXtV2Discriminator
        
    Returns:
        ConvNeXtV2Discriminator: Configured discriminator model
    """
    return ConvNeXtV2Discriminator(model_size="nano", **kwargs)








