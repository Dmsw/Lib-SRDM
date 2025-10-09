import numpy as np
from PIL import Image
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
if '/home/root/project/hsi-denoising/GDM/' not in sys.path:
    sys.path.append('/home/root/project/hsi-denoising/GDM/')

import json

import cv2
import supervision as sv
import torchvision

import torch
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor, sam_hq_model_registry

import pickle


cfg = {
    'image_size': 384,
    'pretrained': '/home/root/project/hsi-denoising/GDM/Grounded-Segment-Anything/ram_plus_swin_large_14m.pth',
}

MAX_SPECTRAL_SIZE = 1000


def sample_by_rgb(rgb_values, spectral_library, srf, n_samples, tau):
    """
    Sample spectra from a spectral library based on the RGB values of a pixel.
    Args:
        rgb_values (np.ndarray): RGB values of the pixel. Shape (N, 3) in [0, 1].
        spectral_library (np.ndarray): Spectral library. Shape (M, B) in [0, 1].
        srf (np.ndarray): Spectral Response Function. Shape (B, 3).
        n_samples (int): Number of samples to take.
        tau (float): Threshold for the RGB values. should be in [0, 1].
    Returns:
        np.ndarray: Sampled spectra.
    """
    # Sample RGB values for acceleration.
    rgb_values = rgb_values[np.random.choice(rgb_values.shape[0], min(n_samples, rgb_values.shape[0]), replace=False)]
    # Calculate the RGB values of the spectra.
    spectral_projected = np.dot(spectral_library, srf)
    # Calculate the distance between the RGB values and the projected spectra.
    distances = np.linalg.norm(rgb_values[:, None] - spectral_projected[None], axis=-1)
    distances = np.min(distances, axis=0)
    assert True, distances

    # Normalize the distances.
    distances = (distances / tau / np.max(distances)) ** 2 / 2
    distances = distances - np.min(distances)
    # calculate the probability
    p = np.exp(-distances)
    p = p / np.sum(p)
    
    # Select the spectra according to the probability
    indices = np.random.choice(np.arange(len(p)), min(n_samples, len(p)), p=p, replace=False)

    return spectral_library[indices]


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

DEVICE = torch.device('cuda')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/home/root/project/hsi-denoising/GDM/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/root/project/hsi-denoising/GDM/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/home/root/project/hsi-denoising/GDM/Grounded-Segment-Anything/sam_hq_vit_h.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_hq_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = "/home/root/dataset/cave/cave/fake_and_real_lemon_slices_ms/fake_and_real_lemon_slices_ms/fake_and_real_lemon_slices_RGB.bmp"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# load ram model
transform = get_transform(image_size=cfg['image_size'])

model = ram_plus(pretrained=cfg['pretrained'],
                image_size=cfg['image_size'],
                vit='swin_l')
model.eval()

model = model.to(DEVICE)

@torch.no_grad()
def get_lib_and_mask(rgb: np.ndarray, sample_per_tag: int=20, use_default=True, gamma: float=0.5, srf=None, lib_path='lib.pkl', tau=0.5, logdir=None, memory_save=False, awb=False):
    """get spectral library and mask according to the input rgb image

    Args:
        rgb (np.ndarray): rgb image in [0, 1] with shape (h, w, 3)
        sample_per_tag (int, optional): spectral sample per tag. Defaults to 20.
        gamma (float, optional): gamma correct for better SAM. Defaults to 0.5.
    """
    if logdir is not None:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    # load lib
    lib = pickle.load(open(os.path.join(lib_path, 'lib.pkl'), 'rb'))
    try:
        with open(os.path.join(lib_path, 'pre_tags.json'), 'r') as f:
            pre_tag = json.load(f)
    except FileNotFoundError:
        pre_tag = []
        print("No pre_tags.json found, use empty list")
    # assert isinstance(lib, dict), "lib should be a dict"
    
    # transform rgb image to meet the requirement of ram model
    rgb_awb = srf.awb(rgb) if awb else rgb
    rgb_gamma = rgb_awb ** gamma
    rgb255 = (rgb_gamma * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(logdir, 'rgb_correct.png'), cv2.cvtColor(rgb255, cv2.COLOR_RGB2BGR))
    rgb_tf = Image.fromarray(rgb255)
    rgb_tf = transform(rgb_tf).unsqueeze(0).to(DEVICE)
    
    # predict tags
    pred = inference(rgb_tf, model)
    tags = pred[0].split(' | ') + pre_tag
    # tags = list(lib.keys())
    
    # get mask through grounding dino and SAM
    bgr = cv2.cvtColor(rgb255, cv2.COLOR_RGB2BGR)
    detections = grounding_dino_model.predict_with_classes(
        image=bgr,
        classes=tags,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    if logdir is not None:
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{tags[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=bgr.copy(), detections=detections, labels=labels)
        cv2.imwrite(os.path.join(logdir, 'detections.png'), annotated_frame)
    
    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    
    if logdir is not None:
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{tags[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=bgr.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(logdir, 'detections_mask.png'), annotated_image)

    dm_lib = {}
    lr_lib = {}
    ret_mask = {}
    for i, (_, mask, confidence, class_id, _) in enumerate(detections):
        mask = np.squeeze(mask)
        tag = tags[class_id]
        print(f"Found tag: {tag}")
        if tag not in lib.keys():
            print(f"Tag {tag} not in library, skip")
            continue
        
        if np.sum(mask) <= 0:
            print(f"Found not pixel for Tag {tag}, skip")
            continue
        
        if memory_save:
            tag_new = f"{tag}-{i}"
        else:
            tag_new = tag
            
        if tag_new not in dm_lib.keys():
            dm_lib[tag_new] = []
            # lr_lib[tag_new] = lib[tag]
            ret_mask[tag_new] = np.zeros_like(mask)
        
        if sum([len(_lib) for _lib in dm_lib[tag_new]]) >= MAX_SPECTRAL_SIZE:
            print("Tag {tag} reaching the max spectral library size!!!")
            continue
                
        dm_lib[tag_new].append(sample_by_rgb(rgb[mask], lib[tag], srf.P.T, sample_per_tag, tau))
        ret_mask[tag_new] = np.bitwise_or(ret_mask[tag_new], mask)
    
    # calculate mask, unknown-mask, and factor
    factor = np.zeros(rgb.shape[:2], dtype=np.float32)
    unknown_mask = np.ones(rgb.shape[:2], dtype=np.bool8)
    for t in dm_lib.keys():
        mask = ret_mask[t]
        assert True, f"{mask.shape} {unknown_mask.shape}, {factor.shape}"
        dm_lib[t] = np.concatenate(dm_lib[t], axis=0)
        unknown_mask = np.bitwise_and(unknown_mask, np.bitwise_not(mask))
        factor += mask.astype(np.float32)

    if use_default:
        ret_mask['default'] = unknown_mask
        factor += unknown_mask

        # select unknown spectra
        spectra = sample_by_rgb(rgb[unknown_mask], lib['default'], srf.P.T, sample_per_tag, tau)
        
        dm_lib['default'] = spectra
    # assert np.all(np.bitwise_xor(unknown_mask , factor>0)), "Unknown mask should be the same as factor"
    
    return dm_lib, lr_lib, ret_mask, factor


if __name__ == '__main__':
    # img = cv2.imread("/home/root/dataset/cave/cave/chart_and_stuffed_toy_ms/chart_and_stuffed_toy_ms/chart_and_stuffed_toy_RGB.bmp")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32) / 255
    # lib, mask, unknow_mask, factor = get_lib_and_mask(img)
    lib_part = pickle.load(open('/home/root/project/hsi-denoising/GDM/library/cave_part/lib.pkl', 'rb'))
    print(len(lib_part.keys()))
    n = sum([len(lib_part[k]) for k in lib_part.keys()])
    print(n)
    lib_all = pickle.load(open('/home/root/project/hsi-denoising/GDM/library/cave_all/lib.pkl', 'rb'))
    print(len(lib_all.keys()))
    n = sum([len(lib_all[k]) for k in lib_all.keys()])
    print(n)
    
    count = 0
    for k in lib_part.keys():
        if k not in lib_all.keys():
            count += 1

    print(f"Found {count} tags not in all library")            

    count = 0            
    for k in lib_all.keys():
        if k not in lib_part.keys():
            count += 1
    
    print(f"Found {count} tags not in part library")
    

