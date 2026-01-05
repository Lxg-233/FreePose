# [2026]FreePose: Modeling Frequency-Decoupled Motion Trajectories for 3D Human Pose Estimation
[![PyTorch](https://img.shields.io/badge/-PyTorch-ff69b4?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
<img width="4849" height="1711" alt="a(1)(1)(1)(1)(1)-第 7 页" src="https://github.com/user-attachments/assets/af95594b-bd95-4fa6-88e9-0af5937fc15c" />

## Dataset

### Human3.6M

#### Preprocessing

1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d', or direct download our processed data [here](https://drive.google.com/file/d/1WWoVAae7YKKKZpa1goO_7YcwVFNR528S/view?usp=sharing) and unzip it.
2. Slice the motion clips by running the following python code in `tools/convert_h36m.py`:

```
python convert_h36m.py
```

### MPI-INF-3DHP

#### Preprocessing

Please refer to - [MotionAGFormer](https://github.com/taatiteam/motionagformer) for dataset setup.

## Training

After dataset preparation, you can train the model as follows:

### human3.6M

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config <PATH-TO-CONFIG> --checkpoint <PATH-TO-CHECKPOINT>
```

where config files are located at `configs/h36m`.

### MPI-INF-3DHP

Please refer to - [MotionAGFormer](https://github.com/taatiteam/motionagformer) for training.

## Evaluation

You can download and unzip it to [get pretrained weight](https://drive.google.com/file/d/1yhXX7VbAjRubZrxd-Rryq3bwYL1b8ko8/view?usp=sharing).

After downloading the weight, you can evaluate Human3.6M models by:

```python
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```

<img width="671" height="519" alt="WPS图片(1)" src="https://github.com/user-attachments/assets/ed8ceae5-b959-489c-938a-09674931e8fb" />

## Demo

Our demo is a modified version of the one provided by [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer) repository. First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. Next, download our base model checkpoint from [here](https://drive.google.com/file/d/1yhXX7VbAjRubZrxd-Rryq3bwYL1b8ko8/view?usp=sharing) and put it in the './checkpoint' directory. Then, you need to put your in-the-wild videos in the './demo/video' directory. We provide [demo](https://drive.google.com/file/d/1hbK1HDz1nMTGYcczOC5r33Mk8nAtLZCr/view?usp=sharing). You can download and unzip it to get demo file. Run the command below:

```python
python vis.py --video sample_video.mp4 --gpu 0
```

## Acknowledgement

Our code refers to the following repositories:

- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer)

We thank the authors for releasing their codes.
