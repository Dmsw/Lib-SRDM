noise_model="randcov75"
save_name="Spec+HSI_DM4"
dataset="cave"
cuda=1

CUDA_VISIBLE_DEVICES=$cuda python ../scripts/hsi_denoise.py 
 # base_samples: path to dataset
# save_dir: path to save the result
# model_config: path to HSI DM config
# in_channels: number of bands
# range_t: perform unconditional denoising in the last range_t steps
# num_samples: set to the size of dataset
# rgb_model_config: path to RGB DM config
# l1: index for gradient
# l2: fusion weight of RGB DM