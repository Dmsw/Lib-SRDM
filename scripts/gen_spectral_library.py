import numpy as np
from PIL import Image
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.append('/home/root/project/hsi-denoising/GDM/')

from srf_tools import SRFTool
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
from guided_diffusion.ntire_gen import NTIRETrain, NTIREValidate
from guided_diffusion.icvl import ICVLTrain, ICVLValidation, ICVLTest
from guided_diffusion.cave import CAVEBase

np.random.seed(42)
torch.manual_seed(42)

cfg = {
    'data_root': ['/home/root/dataset/NTIRE22/'],
    'image_size': 384,
    'pretrained': '/home/root/project/hsi-denoising/GDM/Grounded-Segment-Anything/ram_plus_swin_large_14m.pth',
    'sample_per_tag': 500,
    'pre_tag_file': None,               # use pre-tag instead of ram
    'save_dir': 'library/ntire_awb/',
    'pre_tag': [],
    'gamma': 0.6,   # default 0.6
}

if os.path.exists(cfg['save_dir']) is False:
    os.makedirs(cfg['save_dir'])
else:
    print(f"Directory {cfg['save_dir']} already exists.")
    print("Are you sure you want to overwrite it? (y/n)")
    ans = input()
    if ans != 'y':
        print("Exiting...")
        exit(0)

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
GROUNDING_DINO_CONFIG_PATH = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "Grounded-Segment-Anything/groundingdino_swint_ogc.pth"

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

hsi = [] # [ -1, 1 ]
# for path in cfg['data_root']:
#     dataset = np.load(path)
#     hsi.append(dataset['clean_img'])
# hsi.extend(ICVLTrain().hypers)
# hsi.extend(ICVLValidation().hypers)
# hsi.extend(ICVLTest().hypers)
hsi.extend(NTIRETrain(cfg['data_root'][0], 256, True, bgr2rgb=False).hypers)
hsi.extend(NTIREValidate(cfg['data_root'][0], True).hypers)
# hsi.extend(CAVEBase(data_root=cfg['data_root'][0]).hypers)
# hsi.extend(CAVEBase(data_root=cfg['data_root'][1]).hypers)

hsi = np.stack(hsi, axis=0)
hsi = np.transpose(hsi, [0, 2, 3, 1]) * 0.5 + 0.5

srf = SRFTool()
srf.load_d400_srf()

transform = get_transform(image_size=cfg['image_size'])

model = ram_plus(pretrained=cfg['pretrained'],
                 image_size=cfg['image_size'],
                 vit='swin_l')
model.eval()

model = model.to(DEVICE)

lib = dict(default=[])
tags = set()
if cfg['pre_tag_file'] is not None:
    tags = (json.load(open(cfg['pre_tag_file'], 'r')))
n = len(hsi)
for i in range(n):
    img = hsi[i]
    rgb = srf.gen_rgb_numpy(img, max_value=255, normalize=True, quantize=True, depth=8, awb=True, gamma=cfg['gamma'])
    rgb_tf = Image.fromarray(rgb.astype(np.uint8))
    rgb_tf.save(f"hsi_{i}.png")

    if cfg['pre_tag_file'] is None:
        rgb_tf = transform(rgb_tf).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = inference(rgb_tf, model)
        tag = pred[0]
        tag = tag.split(' | ') + cfg['pre_tag']
        
        tags.update(tag)
    else:
        tag = tags + cfg['pre_tag']

    rgb = rgb.astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detections = grounding_dino_model.predict_with_classes(
        image=bgr,
        classes=tag,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{tag[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=bgr.copy(), detections=detections, labels=labels)
    
    cv2.imwrite(f"groundingdino_annotated_image_{i}.jpg", annotated_frame)
    
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

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{tag[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=bgr.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # save the annotated grounded-sam image
    cv2.imwrite(f"grounded_sam_annotated_image_{i}.jpg", annotated_image)
    
    unknown_mask = np.ones(bgr.shape[:2], dtype=np.bool8)
    for _, mask, confidence, class_id, _ in detections:
        img_m = img[mask]
        assert True, f"Image: {img.shape}, Mask: {mask.shape}, Masked Image: {img_m.shape}"
        assert True, mask
        n_spc = img_m.shape[0]
        if n_spc == 0:
            continue
        idx = np.random.choice(range(n_spc), min(cfg['sample_per_tag'], n_spc), replace=False)
        spectra = img_m[idx]
        t = tag[class_id]
        print(f"Tag: {t}, Spectra: {spectra.shape}")
        if t not in lib:
            lib[t] = []
        lib[t].append(spectra)
        
        # update unknown mask
        unknown_mask = np.bitwise_and(unknown_mask, np.bitwise_not(mask))

    # generate default spectral library for unknown material
    spectra = img[unknown_mask]
    n_spc = spectra.shape[0]
    idx = np.random.choice(range(n_spc), min(cfg['sample_per_tag'], n_spc), replace=False)
    lib['default'].append(spectra[idx])

lib = {k: np.concatenate(v, axis=0) for k, v in lib.items()}
for k, v in lib.items():
    print(f"Tag: {k}, Spectra: {v.shape}")
            
print(tags)

# assert tags == set(lib.keys()), "Tags and library keys should be the same"
json.dump(cfg, open(os.path.join(cfg['save_dir'], 'cfg.json'), 'w'))
if cfg['pre_tag_file'] is None:
    tags = list(tags)
    tags.sort()
    json.dump(tags, open(os.path.join(cfg['save_dir'], 'tags.json'), 'w'))
    json.dump(cfg['pre_tag'], open(os.path.join(cfg['save_dir'], 'pre_tags.json'), 'w'))
with open(os.path.join(cfg['save_dir'], 'lib.pkl'), 'wb') as f:
    pickle.dump(lib, f)
