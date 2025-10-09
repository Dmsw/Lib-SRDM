import torch 

def gray_world_white_balance(image: torch.Tensor) -> torch.Tensor:
    """
    Apply Gray World white balance to an image.

    Args:
        image (torch.Tensor): Input image tensor with shape (C, H, W) or (N, C, H, W),
                              where C is the number of channels (3 for RGB).

    Returns:
        torch.Tensor: White-balanced image tensor with the same shape as input.
    """
    # Ensure input has 4 dimensions: (N, C, H, W)
    is_batch = image.dim() == 4
    if not is_batch:
        image = image.unsqueeze(0)  # Add batch dimension

    # Compute the mean of each channel (R, G, B)
    channel_means = image.mean(dim=(2, 3), keepdim=True)  # Shape: (N, C, 1, 1)

    # Compute scaling factors to normalize means to the average mean
    avg_mean = channel_means.mean(dim=1, keepdim=True)  # Shape: (N, 1, 1, 1)
    scale_factors = avg_mean / channel_means  # Shape: (N, C, 1, 1)

    # Apply scaling factors to each channel
    balanced_image = image * scale_factors

    # Clip values to valid range [0, 1] or [0, 255] depending on input type
    if image.dtype.is_floating_point:
        balanced_image = balanced_image.clamp(0.0, 1.0)
    else:
        balanced_image = balanced_image.clamp(0, 255)

    # Remove batch dimension if input was not a batch
    if not is_batch:
        balanced_image = balanced_image.squeeze(0)

    return balanced_image


def white_block_awb(img, block_size=10):
    """
    Apply White Block Automatic White Balance (AWB) using PyTorch.

    Args:
        img (PIL Image or Tensor): Input image (RGB), either as a PIL Image or a PyTorch Tensor.
        block_size (int): The size of the block used to compute the average brightness (default is 10).

    Returns:
        Tensor: White-balanced image as a PyTorch tensor.
    """
    # Ensure the image is in [0, 1] range (normalize)
    img = img.clamp(0, 1)
    
    is_batch = img.dim() == 4
    if is_batch:
        img = img.squeeze(0)  # Add batch dimension

    # Get image dimensions
    C, H, W = img.shape

    # Get the brightest block (top-left corner)
    max_value = img.view(C, -1).max(dim=1)[0]  # Max value per channel
    brightest_value, _ = max_value.max(dim=-1)  # Find max value from the 3 channels

    # Calculate scaling factors based on the brightest value
    scaling_factors = brightest_value / max_value
    scaling_factors = scaling_factors.view(-1, 1, 1)  # Reshape for broadcasting

    # Apply the scaling factors to adjust the image channels
    img_adjusted = img * scaling_factors

    # Ensure the adjusted image is still in [0, 1] range
    img_adjusted = img_adjusted.clamp(0, 1)

    if is_batch:
        img_adjusted = img_adjusted.unsqueeze(0)

    return img_adjusted