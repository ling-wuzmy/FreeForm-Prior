## Introduction

This is the implementation of our paper:
<h3 align="center">FreeForm-Prior: Parametric-Guided Model-Free 3D Human Mesh Reconstruction</h3>

## TODO :white_check_mark:

- [x] Provide the training weights.
      
1. Install dependences. This project is developed using >= python 3.8 on Ubuntu 16.04. NVIDIA GPUs are needed. We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment.

  ```bash
    # 1. Create a conda virtual environment.
    conda create -n pytorch python=3.8 -y
    conda activate pytorch

    # 2. Install PyTorch >= v1.6.0 following [official instruction](https://pytorch.org/). Please adapt the cuda version to yours.
    pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

    # 3. Pull our code.
    git clone https://github.com/ling-wuzmy/FreeForm-Prior
    cd FreeForm-Prior

    # 4. Install other packages. This project doesn't have any special or difficult-to-install dependencies.
    sh requirements.sh

    #5. Install vm_ik
    python setup.py develop
  ```
2. Prepare SMPL layer. We use [smplx](https://github.com/vchoutas/smplx#installation).

   1. Install `smplx` package by `pip install smplx`. Already done in the first step.
   2. Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) (female & male) and [here](http://smplify.is.tue.mpg.de/) (neutral) to `${Project}/data/smpl`. Please rename them as `SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`, and `SMPL_NEUTRAL.pkl`, respectively.
   3. Download others SMPL-related from [Google drive](https://drive.google.com/drive/folders/1LRMo_7raQuSRuUKAvXKSlzlvQJ5C0IHR?usp=share_link) or [Onedrive](https://chinapku-my.sharepoint.com/:f:/g/personal/2101111546_pku_edu_cn/EitToj4t0BlMmKAo6CZT2H8BMmkyAKQBjY6kO5h0htKveA?e=b57zU5) and put them to `${Project}/data/smpl`.
3. Download data following the **Data** section. In summary, your directory tree should be like this

  ```
    ${Project}
    ├── assets
    ├── command
    ├── configs
    ├── data  
    ├── experiment 
    ├── inputs 
    ├── vm_ik 
    ├── main 
    ├── models 
    ├── README.md
    ├── setup.py
    `── requirements.sh
  ```

  - `assets` contains the body virtual markers in `npz` format. Feel free to use them.
  - `command` contains the running scripts.
  - `configs` contains the configurations in `yml` format.
  - `data` contains soft links to images and annotations directories.
  - `vm_ik` contains kernel codes for our method.
  - `main` contains high-level codes for training or testing the network.
  - `models` contains pre-trained weights. Download from [there](https://github.com/ShirleyMaxx/VirtualMarker).
  - *`experiment` will be automatically made after running the code, it contains the outputs, including trained model weights, test metrics and visualized outputs.

## Quick demo :star:

1. **Installation.** Make sure you have finished the above installation successfully. vm_ik does not detect person and only estimates relative pose and mesh, therefore please also install [VirtualPose](https://github.com/wkom/VirtualPose) following its instructions. VirtualPose will detect all the person and estimate their root depths. .
  ```bash
  git clone https://github.com/wkom/VirtualPose.git
  cd VirtualPose
  python setup.py develop
  ```

2. **Render Env.** If you run this code in ssh environment without display device, please do follow:
  ```
  1. Install osmesa follow https://pyrender.readthedocs.io/en/latest/install/
  2. Reinstall the specific pyopengl fork: https://github.com/mmatl/pyopengl
  3. Set opengl's backend to osmesa via os.environ["PYOPENGL_PLATFORM"] = "osmesa"
  ```

3. **Model weight.** Download the pre-trained VirtualMarker models `baseline_mix` from [there](https://github.com/ShirleyMaxx/VirtualMarker).

4. **Input image/video.** Prepare `input.jpg` or `input.mp4` and put it at `inputs` folder. Both image and video input are supported. Specify the input path and type by arguments.

5. **RUN.** You can check the output at `experiment/simple3dmesh_infer/exp_*/vis`.
  ```bash
  sh command/simple3dmesh_infer/baseline.sh
  ```
  


### Data

The `data` directory structure should follow the below hierarchy. Please download the images from the official sites. Download all the processed annotation files from [there](https://github.com/ShirleyMaxx/VirtualMarker).

```
${Project}
|-- data
    |-- 3DHP
    |   |-- annotations
    |   `-- images
    |-- COCO
    |   |-- annotations
    |   `-- images
    |-- Human36M
    |   |-- annotations
    |   `-- images
    |-- PW3D
    |   |-- annotations
    |   `-- images
    |-- Up_3D
    |   |-- annotations
    |   `-- images
    `-- smpl
        |-- smpl_indices.pkl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_MALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- mesh_downsampling.npz
        |-- J_regressor_extra.npy
        `-- J_regressor_h36m_correct.npy
```
### Evaluation

To evaluate the model, specify the model path `test.weight_path` in `configs/simple3dmesh_test/baseline_*.yml`. Argument `--mode test` should be set. Results can be seen in `experiment` directory or in the tensorboard.

```bash
sh command/simple3dmesh_test/test_h36m.sh
sh command/simple3dmesh_test/test_pw3d.sh
```





## Acknowledgement
This repo is built on the excellent work [GraphCMR](https://github.com/nkolot/GraphCMR), [SPIN](https://github.com/nkolot/SPIN), [Pose2Mesh](https://github.com/hongsukchoi/Pose2Mesh_RELEASE), [HybrIK](https://github.com/Jeff-sjtu/HybrIK), [CLIFF](https://github.com/haofanwang/CLIFF) and [VitualMarker](https://github.com/ShirleyMaxx/VirtualMarker). Thanks for these great projects.
