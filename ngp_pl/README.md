# ngp_pl

### Advertisement: stay tuned with [my channel](https://www.youtube.com/channel/UC7UlsMUu_gIgpqNGB4SqSwQ), I will upload cuda tutorials recently, and do a stream about this implementation!

### Update 2022 July 14th: Multi-GPU training is available now! With multiple GPUs, now you can achieve high quality under a minute!

Instant-ngp (only NeRF) in pytorch+cuda trained with pytorch-lightning (**high quality with high speed**). This repo aims at providing a concise pytorch interface to facilitate future research, and am grateful if you can share it (and a citation is highly appreciated)!

https://user-images.githubusercontent.com/11364490/177025079-cb92a399-2600-4e10-94e0-7cbe09f32a6f.mp4

https://user-images.githubusercontent.com/11364490/176821462-83078563-28e1-4563-8e7a-5613b505e54a.mp4

*  [Official CUDA implementation](https://github.com/NVlabs/instant-ngp/tree/master)
*  [torch-ngp](https://github.com/ashawkey/torch-ngp) another pytorch implementation that I highly referenced.

# :computer: Installation

This implementation has **strict** requirements due to dependencies on other libraries, if you encounter installation problem due to hardware/software mismatch, I'm afraid there is **no intention** to support different platforms (you are welcomed to contribute).

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB (Tested with RTX 2080 Ti), CUDA 11.3 (might work with older version)
* 32GB RAM (in order to load full size images)

## Software

* Clone this repo by `git clone https://github.com/kwea123/ngp_pl`
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ngp_pl python=3.8` to create a conda environment and activate it by `conda activate ngp_pl`)
* Python libraries
    * Install pytorch by `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)

# :books: Data preparation

1.  Synthetic data

Download preprocessed datasets from [NSVF](https://github.com/facebookresearch/NSVF#dataset). **Do not change the folder names** since there is some hard-coded fix in my dataloader.

2.  Custom data

Please run `colmap` and get a folder `sparse/0` under which there are `cameras.bin`, `images.bin` and `points3D.bin`. [nerf_llff_data](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing) is also supported.

# :key: Training

Quickstart:

1.  Synthetic data

`python train.py --root_dir <path/to/lego> --exp_name Lego`

2.  Custom data

`python train.py --root_dir <path/to/fern> --dataset_name colmap --exp_name fern --scale 2.0 --downsample 0.25`

It will train the lego scene for 30k steps (each step with 8192 rays), and perform one testing at the end. The training process should finish within about 5 minutes (saving testing image is slow, add `--no_save_test` to disable). Testing PSNR will be shown at the end.

If your GPU has larger memory, you can try increasing `batch_size` (and `lr`) and reducing `num_epochs` (e.g. `--batch_size 16384 --lr 2e-2 --num_epochs 20`). In my experiments, this further reduces the training time by 10~25s while maintaining the same quality.

More options can be found in [opt.py](opt.py).

# :mag_right: Testing

Use `test.ipynb` to generate images. Lego pretrained model is available [here](https://github.com/kwea123/ngp_pl/releases/tag/v1.0)

# Comparison with torch-ngp and the paper

I compared the quality (average testing PSNR on `Synthetic-NeRF`) and the inference speed (on `Lego` scene) v.s. the concurrent work torch-ngp (default settings) and the paper, all trained for about 5 minutes:

| Method    | avg PSNR | FPS   | GPU     |
| :---:     | :---:    | :---: | :---:   |
| torch-ngp | 31.46    | 18.2  | 2080 Ti |
| mine      | 32.96    | 36.2  | 2080 Ti |
| instant-ngp paper | **33.18** | **60** | 3090 |

As for quality, mine is slightly better than torch-ngp, but the result might fluctuate across different runs.

As for speed, mine is faster than torch-ngp, but is still only half fast as instant-ngp. Speed is dependent on the scene (if most of the scene is empty, speed will be faster).

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/176800109-38eb35f3-e145-4a09-8304-1795e3a4e8cd.png", width="45%">
  <img src="https://user-images.githubusercontent.com/11364490/176800106-fead794f-7e70-4459-b99e-82725fe6777e.png", width="45%">
  <br>
  <img src="https://user-images.githubusercontent.com/11364490/180444355-444676cf-2af2-49ad-9fe2-16eb1e6c4ef1.png", width="45%">
  <img src="https://user-images.githubusercontent.com/11364490/180444337-3df9f245-f7eb-453f-902b-0cb9dae60144.png", width="45%">
  <br>
  <sup>Left: torch-ngp. Right: mine.</sup>
</p>

More details are in the following section.

# Benchmarks

To run benchmarks, use the scripts under `benchmarking`.

Followings are my results trained using 1 RTX 2080 Ti (qualitative results [here](https://github.com/kwea123/ngp_pl/issues/7)):

<details>
  <summary>Synthetic-NeRF</summary>

|       | Mic   | Ficus | Chair | Hotdog | Materials | Drums | Ship  | Lego  | AVG   |
| :---: | :---: | :---: | :---: | :---:  | :---:     | :---: | :---: | :---: | :---: |
| PSNR  | 35.59 | 34.13 | 35.28 | 37.35  | 29.46     | 25.81 | 30.32 | 35.76 | 32.96 |
| SSIM  | 0.988 | 0.982 | 0.984 | 0.980  | 0.944     | 0.933 | 0.890 | 0.979 | 0.960 |
| LPIPS | 0.017 | 0.024 | 0.025 | 0.038  | 0.070     | 0.076 | 0.133 | 0.022 | 0.051 |
| FPS   | 40.81 | 34.02 | 49.80 | 25.06  | 20.08     | 37.77 | 15.77 | 36.20 | 32.44 |
| Training time | 3m9s | 3m12s | 4m17s | 5m53s | 4m55s | 4m7s | 9m20s | 5m5s | 5m00s |

</details>

<details>
  <summary>Synthetic-NSVF</summary>

|       | Wineholder | Steamtrain | Toad | Robot | Bike | Palace | Spaceship | Lifestyle | AVG | 
| :---: | :---: | :---: | :---: | :---: | :---:  | :---:  | :---: | :---: | :---: |
| PSNR  | 31.64 | 36.47 | 35.57 | 37.10 | 37.87 | 37.41 | 35.58 | 34.76 | 35.80 |
| SSIM  | 0.962 | 0.987 | 0.980 | 0.994 | 0.990 | 0.977 | 0.980 | 0.967 | 0.980 |
| LPIPS | 0.047 | 0.023 | 0.024 | 0.010 | 0.015 | 0.021 | 0.029 | 0.044 | 0.027 |
| FPS   | 47.07 | 75.17 | 50.42 | 64.87 | 66.88 | 28.62 | 35.55 | 22.84 | 48.93 |
| Training time | 3m58s | 3m44s | 7m22s | 3m25s | 3m11s | 6m45s | 3m25s | 4m56s | 4m36s |

</details>

<details>
  <summary>Tanks and Temples</summary>

|       | Ignatius | Truck | Barn  | Caterpillar | Family | AVG   | 
|:---:  | :---:    | :---: | :---: | :---:       | :---:  | :---: |
| *PSNR | 28.22    | 27.57 | 28.00 | 26.16       | 33.94  | 28.78 |
| **FPS | 10.04    |  7.99 | 16.14 | 10.91       | 6.16   | 10.25 |

*Trained with `downsample=0.5` (due to insufficient RAM) and evaluated with `downsample=1.0`

**Evaluated on `test-traj`

</details>

<details>
  <summary>BlendedMVS</summary>

|       | *Jade  | *Fountain | Character | Statues | AVG   | 
|:---:  | :---:  | :---:     | :---:     | :---:   | :---: |
| PSNR  | 25.43  | 26.82     | 30.43     | 26.79   | 27.38 |
| **FPS | 26.02  | 21.24     | 35.99     | 19.22   | 25.61 |
| Training time | 6m31s | 7m15s | 4m50s | 5m57s | 6m48s |

*I manually switch the background from black to white, so the number isn't directly comparable to that in the papers.

**Evaluated on `test-traj`

</details>


# TODO

- [ ] GUI
