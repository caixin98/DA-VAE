import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, List, Union, Any
import numpy as np
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_processing_base import BatchFeature
from transformers.image_utils import ChannelDimension, ImageInput, PILImageResampling
from transformers.utils.generic import TensorType
from transformers.utils import logging

logger = logging.get_logger(__name__)


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} and width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class TensorImageProcessor(BaseImageProcessor):
    """
    A tensor-based image processor for Qwen2.5-VL model.
    This processor works directly with torch tensors without converting to PIL/numpy.
    Aligned with the original Qwen2VLImageProcessor implementation.
    Inherits from BaseImageProcessor for compatibility with transformers library.
    """
    
    model_input_names = ["pixel_values", "image_grid_thw"]
    
    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: Any = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ):
        """
        Initialize the tensor image processor.
        
        Args:
            do_resize: Whether to resize the image's (height, width) dimensions.
            size: Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample: Resampling filter to use when resizing the image.
            do_rescale: Whether to rescale the image by the specified scale `rescale_factor`.
            rescale_factor: Scale factor to use if rescaling the image.
            do_normalize: Whether to normalize the image.
            image_mean: Mean to use if normalizing the image.
            image_std: Standard deviation to use if normalizing the image.
            do_convert_rgb: Whether to convert the image to RGB.
            min_pixels: The min pixels of the image to resize the image.
            max_pixels: The max pixels of the image to resize the image.
            patch_size: The spatial patch size of the vision encoder.
            temporal_patch_size: The temporal patch size of the vision encoder.
            merge_size: The merge size of the vision encoder to llm encoder.
        """
        super().__init__(**kwargs)
        
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        else:
            size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            
        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]
        self.size = size

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = do_convert_rgb

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        
        # Convert to tensors for efficient processing
        self.mean_tensor = torch.tensor(self.image_mean).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor(self.image_std).view(1, 3, 1, 1)
    
    def _preprocess_tensor(
        self,
        images: torch.Tensor,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
    ):
        """
        Preprocess tensor images. This is the tensor version of the original _preprocess method.
        
        Args:
            images: Input tensor of shape (B, C, H, W)
            do_resize: Whether to resize the image
            size: Size configuration for resizing
            do_rescale: Whether to rescale the image
            rescale_factor: Rescale factor
            do_normalize: Whether to normalize the image
            image_mean: Mean for normalization
            image_std: Std for normalization
            patch_size: Patch size for the vision encoder
            temporal_patch_size: Temporal patch size
            merge_size: Merge size
            
        Returns:
            Tuple of (flatten_patches, (grid_t, grid_h, grid_w))
        """
        # Use default values if not provided
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        
        # Ensure input is 4D (B, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        batch_size, channels, height, width = images.shape
        
        # Resize images if needed
        resized_height, resized_width = height, width
        if do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
            )
            if resized_height != height or resized_width != width:
                images = F.interpolate(
                    images, 
                    size=(resized_height, resized_width), 
                    mode='bilinear', 
                    align_corners=False,
                    antialias=True
                )
        
        # Rescale pixel values if needed
        if do_rescale:
            images = images * rescale_factor
        
        # Normalize images
        device = images.device
        mean_tensor = torch.tensor(image_mean, device=device).view(1, 3, 1, 1)
        std_tensor = torch.tensor(image_std, device=device).view(1, 3, 1, 1)
        images = (images - mean_tensor) / std_tensor
        
        # Handle temporal dimension (for images, this is just 1)
        patches = images
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = patches[-1].unsqueeze(0).repeat(temporal_patch_size - 1, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=0)
        
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        
        # Reshape patches exactly like the original implementation
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Use permute instead of transpose for better compatibility
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
        
        return flatten_patches, (grid_t, grid_h, grid_w)
    
    def preprocess(
        self,
        images: Union[torch.Tensor, ImageInput],
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: Optional[Any] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess images or tensors. This method handles both PIL images and torch tensors.
        
        Args:
            images: Images to preprocess. Can be PIL images, numpy arrays, or torch tensors.
            do_resize: Whether to resize the image
            size: Size configuration for resizing
            min_pixels: Minimum pixels for resizing
            max_pixels: Maximum pixels for resizing
            resample: Resampling filter to use when resizing
            do_rescale: Whether to rescale the image
            rescale_factor: Rescale factor
            do_normalize: Whether to normalize the image
            image_mean: Mean for normalization
            image_std: Std for normalization
            patch_size: Patch size for the vision encoder
            temporal_patch_size: Temporal patch size
            merge_size: Merge size
            do_convert_rgb: Whether to convert to RGB
            return_tensors: Type of tensors to return
            data_format: Channel dimension format
            input_data_format: Input channel dimension format
            
        Returns:
            BatchFeature containing processed tensors
        """
        # If input is torch tensor, use tensor-specific processing
        if isinstance(images, torch.Tensor):
            return self._preprocess_tensors(
                images,
                do_resize=do_resize,
                size=size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
                return_tensors=return_tensors,
            )
        else:
            # For PIL images and other formats, convert to tensor first
            if isinstance(images, (list, tuple)):
                # Convert list of images to tensor
                if len(images) == 0:
                    raise ValueError("Images list cannot be empty")
                
                # Convert all images to tensors
                tensors = []
                for img in images:
                    if isinstance(img, Image.Image):
                        # PIL image
                        img_rgb = img.convert('RGB')
                        img_tensor = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).float()
                    else:
                        # numpy array or other format
                        img_tensor = torch.from_numpy(img).float()
                        if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
                            img_tensor = img_tensor.permute(2, 0, 1)
                    tensors.append(img_tensor)
                
                # Stack tensors
                images_tensor = torch.stack(tensors, dim=0)
            else:
                # Single image
                if isinstance(images, Image.Image):
                    # PIL image
                    img_rgb = images.convert('RGB')
                    images_tensor = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).float().unsqueeze(0)
                else:
                    # numpy array or other format
                    img_tensor = torch.from_numpy(images).float()
                    if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
                        img_tensor = img_tensor.permute(2, 0, 1)
                    images_tensor = img_tensor.unsqueeze(0)
            
            return self._preprocess_tensors(
                images_tensor,
                do_resize=do_resize,
                size=size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
                return_tensors=return_tensors,
            )
    
    def _preprocess_tensors(
        self,
        images: torch.Tensor,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Process input tensor images.
        
        Args:
            images: Input tensor of shape (B, C, H, W) or (C, H, W)
            do_resize: Whether to resize the image
            size: Size configuration for resizing
            min_pixels: Minimum pixels for resizing
            max_pixels: Maximum pixels for resizing
            do_rescale: Whether to rescale the image
            rescale_factor: Rescale factor
            do_normalize: Whether to normalize the image
            image_mean: Mean for normalization
            image_std: Std for normalization
            patch_size: Patch size for the vision encoder
            temporal_patch_size: Temporal patch size
            merge_size: Merge size
            return_tensors: Type of tensors to return
            
        Returns:
            BatchFeature containing processed tensors
        """
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        # Ensure input is 4D (B, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        batch_size = images.shape[0]
        pixel_values, vision_grid_thws = [], []
        
        for i in range(batch_size):
            image = images[i:i+1]  # Keep batch dimension
            patches, image_grid_thw = self._preprocess_tensor(
                image,
                do_resize=do_resize,
                size=size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
            )
            pixel_values.append(patches)
            vision_grid_thws.append(image_grid_thw)
        
        # Stack results
        pixel_values = torch.stack(pixel_values, dim=0)
        vision_grid_thws = torch.tensor(vision_grid_thws, dtype=torch.long)
        
        data = {
            "pixel_values": pixel_values,
            "image_grid_thw": vision_grid_thws
        }
        
        return BatchFeature(data=data, tensor_type=return_tensors)
    
    def __call__(
        self,
        images: Union[torch.Tensor, ImageInput],
        **kwargs,
    ) -> BatchFeature:
        """
        Call method for backward compatibility.
        """
        return self.preprocess(images, **kwargs)
    
    def to(self, device: torch.device) -> 'TensorImageProcessor':
        """Move the processor to a specific device."""
        self.mean_tensor = self.mean_tensor.to(device)
        self.std_tensor = self.std_tensor.to(device)
        return self


def create_tensor_image_processor(
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    do_resize: bool = True,
    do_rescale: bool = True,
    do_normalize: bool = True,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280,
) -> TensorImageProcessor:
    """
    Create a tensor image processor with default settings for Qwen2.5-VL.
    
    Args:
        patch_size: Patch size for the vision model
        temporal_patch_size: Temporal patch size
        merge_size: Merge size
        do_resize: Whether to resize images
        do_rescale: Whether to rescale images
        do_normalize: Whether to normalize images
        min_pixels: Minimum pixels for resizing
        max_pixels: Maximum pixels for resizing
        
    Returns:
        Configured TensorImageProcessor
    """
    return TensorImageProcessor(
        do_resize=do_resize,
        size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
        do_rescale=do_rescale,
        rescale_factor=1.0 / 255.0,
        do_normalize=do_normalize,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
    ) 