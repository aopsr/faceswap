# Faceswap++

[<img src="https://img.shields.io/discord/1059728184414834688?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white">](https://discord.gg/mDDbHc7Dus)

<font size="3"> Faceswap++ is an improved superset of Faceswap and DeepFaceLab. </font>

This is an experiment-focused repo aimed at developing better deepfaking techniques. All new features are personally conceived, implemented, and tested. Faceswap++ will continue to be in sync with upstream and maintain backwards compatibility, in general.

## Installation
See [INSTALL.md](INSTALL.md)

## Comparison
Feature | Faceswap  | Faceswap++ | DeepFaceLab |
| - | - | - | - |
| TF2 + AMP | ✅ | ✅ | ❌
| Custom models | ✅ | ✅ | ❌
| Custom loss | ✅ | ✅ | ❌
| Easy extract cleanup | ✅ | ✅ | ❌
| Windows, Linux, Mac | ✅ | ✅ | ❌
| Xseg | ❌ | ✅ | ✅
| MKL, etc | ❌ | ✅ | ✅
| LRD | ❌ | ✅ | ✅
| Pretraining | ❌ | ✅ | ✅
| LR Scheduling | ❌ | ✅ | ❌
| Face tracking extract | ❌ | ✅ | ❌
| Mix mask types | ❌ | ✅ | ❌
| Gradient Accumulation | ❌ | ✅ | ❌
| Further optimizations | ❌ | ✅ | ❌

Note: Powers and GAN are not necessary and therefore not included.

## Extract
Extract accuracy is now equivalent to DeepFaceLab after the addition of second pass but also benefits from 5-10x extract speed increase - TF2 and multithreading. Detection (S3FD) and alignment (FAN).

Bisenet-fp generic masker works well for almost all scenes. Xseg was added to accommodate custom masking (place Xseg model in faceswap root).

Faceswap allows extraction direct from video without splitting into frames. Please reencode with Adobe Media Encoder if timestamps are not standard (eg output from Topaz Labs Video Enhance AI) or split into frames. Face data are stored to versatile alignments.json.

Manual and sort tool allow for easy cleanup of extracted faces and alignments.

### Extract Workflow

1. Extract from video or folder of images, no masking
2. Sort faces in Tools -> Sort. Sort by face (identity) and distance (misalignment) and delete bad faces
3. Remove deleted faces from alignments file in Tools -> Alignments -> Remove-Faces
4. Check remaining faces in Tools -> Manual and delete leftover bad faces
5. Apply mask in Tools -> Mask
6. Check masks in Tools -> Manual
7. Delete folder of initially extracted faces
8. Extract clean faces with mask in Tools -> Alignments -> Extract

## Training

### NOTE: SAEHD has been ported in its entirety - DF/LIAE and udt options.

Phaze-A allows for highly optimized model architecture - high quality and resolution with low memory requirements. See example config files for optimized architectures that outperform SAEHD.

Features (everything is fully customizable):
- input and output size
- encoder (Conv or EfficientNetV2)
- bottleneck size and norm
- fully connected layers (inter)
  - dimensions and upscales
  - individual, shared, LIAE style
  - optimized G-block
- decoder
  - split or shared
  - normalization
  - residual blocks

The best performing models are IAE style but with G-block instead of shared fully connected layer. The result is LIAE but with DF src-likeliness.

### Training Workflow

(Optional) Pretraining
- Pretrain on dataset like FFHQ
- Load weights onto new model
  - To reduce identity bleed, only load encoder weights (optionally one of inter or decoder weights as well)
  - Optionally freeze loaded weights for 5-10k iters in the beginning to allow other layers to catch up

For optimized 256px model:
1. RW, random flip until face reasonably formed (150k iters)
2. Increase EMP (default eye and mouth 3), mouth to 10-20 if no teeth separation (optional during RW stage)
3. No RW, no random flip until face is crisp (50k iters)

6GB card (RTX A3000) trains optimized 256px model at BS 8 in 1 day.

## Additional Features

Gradient Accumulation - use this option to multiply effective batch size by N without increasing memory requirement. Useful for training large models that would not normally fit.

LR Scheduling - lr warmup and cosine annealing

Mix mask types - useful for long videos in which bisenet-fp works for the majority frames except a portion, where custom xseg can be used

Face tracking - to use face tracking in manual tool, press 'e' to automatically
1. copy bounding box to next frame
2. align face in next frame
3. shift bounding box to be centered on face

Pressing 'e' will work until 1. the next frame already has a face or 2. the current alignment is bad.

Pressing 'p' will do 'e' but for a inputted number of frames. Press any key to interrupt.
