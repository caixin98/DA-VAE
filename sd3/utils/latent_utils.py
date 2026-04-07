import torch

def pack_latent_to_chw4(latent, squeeze_batch_dim: bool = True):
    """
    Pack latent from (1, C, H, W) to (C*4, H//2, W//2).
    
    Args:
        latent: Input latent tensor of shape (1, C, H, W)
        
    Returns:
        Tuple of (packed_latent, original_shape_info)
    """
    batch_size, num_channels, height, width = latent.shape
    latent = latent.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latent = latent.permute(0, 1, 3, 5, 2, 4)
    latent = latent.reshape(batch_size, num_channels * 4, height // 2, width // 2)
    return latent, [batch_size, num_channels * 4, height // 2, width // 2] 

def unpack_latent_from_chw4(packed_latent):
    """
    Unpack latent from (B,C*4, H//2, W//2) to (B, C, H, W).
    
    Args:
        packed_latent: Input latent tensor of shape (B, C*4, H//2, W//2)
        
    Returns:
        Unpacked latent tensor of shape (B, C, H, W)
    """
    batch_size, num_channels, height, width = packed_latent.shape
    latent = packed_latent.view(batch_size, num_channels // 4, 2, 2, height, width)
    latent = latent.permute(0, 1, 4, 2, 5, 3)
    latent = latent.reshape(batch_size, num_channels // 4, height * 2, width * 2)
    return latent

