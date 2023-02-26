# EpiGRAF: Rethinking training of 3D GANs

[[website]](https://universome.github.io/epigraf)
[[paper]](https://universome.github.io/assets/projects/epigraf/epigraf.pdf)
[[arxiv]](https://arxiv.org/abs/2206.10535)

<div style="text-align:center">
<img src="https://user-images.githubusercontent.com/3128824/174703885-903df621-9cbe-4f52-bb67-990dbae52cd7.gif" alt="Generation examples for EpiGRAF"/>
</div>

Code release progress:
- [x] Training/inference code
- [x] Installation and running instructions
- [x] Megascans rendering scripts & instructions
- [x] Datasets
- [ ] Pre-trained checkpoints
- [ ] Fix the problem with eye-glasses on FFHQ
- [ ] Jupyter notebook demos

Limitations / known problems:
- Eye-glasses are synthesized as carved on the face due to not modeling specular effects for them (2D upsampler-based generators, like EG3D, do this for high-res outputs through the upsampler) and using the classical NeRF renderer. This is a known problem and we are working on a fix.
- Low-resolution artifacts due to patch-wise training and producing tri-planes in the dataset resolution and not higher.
- Patch-wise training under-performs compared to full-resolution training for 2D generators

Please, create an issue if you'll find any problems, bugs or have any questions with our repo.

Checkpoints:
- [x] FFHQ 512x512: [FID: 9.87 | 767 MB](https://disk.yandex.ru/d/XJ0k9-kQyHouwQ)
- [ ] FFHQ 256x256
- [ ] Megascans Plants 256x256
- [ ] Megascans Food 256x256

## Installation

To install and activate the environment, run the following command:
```
conda env create -f environment.yml -p env
conda activate ./env
```
This repo is built on top of [StyleGAN3](https://github.com/NVlabs/stylegan3), so make sure that it runs on your system.

Sometimes, it falls down with the error:
```
AttributeError: module 'distutils' has no attribute 'version'
```
in which case you would need to [install an older verion](https://github.com/pytorch/pytorch/issues/69894#issuecomment-1080635462) of `setuptools`:
```
pip install setuptools==59.5.0
```

## Evaluation

Download the checkpoint above and save it into `checkpoints/model.pkl`.
To generate the videos, run:
```
python scripts/inference.py hydra.run.dir=. ckpt.network_pkl=$(eval pwd)/checkpoints/model.pkl vis=video_grid camera=front_circle output_dir=results num_seeds=9
```
You can control the sampling resolution via the `img_resolution` argument.

To compute FID against the `/path/to/dataset.zip` dataset, run:
```
python scripts/calc_metrics.py hydra.run.dir=. ckpt.network_pkl=$(eval pwd)/checkpoints/model.pkl ckpt.reload_code=false img_resolution=512 metrics=fid50k_full data=/path/to/dataset.zip gpus=4 verbose=true
```

## Data

### Real data

For FFHQ and Cats, we use the camera poses provided by [GRAM](https://yudeng.github.io/GRAM/) --- you can download them with their provided links.
For Cats, we used exactly the same dataset as GRAM, we also upload it [here](https://disk.yandex.ru/d/FXwuU1uWTLwOmQ) (together with our pre-processed camera poses).
For FFHQ, in contrast to some previous works (e.g., EG3D or GRAM), we do not re-crop it and use [the original one](https://github.com/NVlabs/ffhq-dataset) (but with the camera poses provided for the cropped version by GRAM).

### Megascans

We give the links to the Megascans datasets, as well as the rendering code and documentation on how to use it in a [separate repo](https://github.com/universome/megascans-rendering).
We also prepared a script for simpler downloading of the Megascans datasets: you can download it via:
```
python scripts/data_scripts/download_megascans.py food /my/output/dir/
python scripts/data_scripts/download_megascans.py plants /my/output/dir/
```

### How to pre-process the datasets

Data should be stored in a zip archive, the exact structure is not important, the script will use all the found images.
Put your datasets into `data/` directory.
If you want to train with camera pose conditioning (either in Generator or Discriminator), then create a `dataset.json` with `camera_angles` dict of `"<FILE_NAME>": [yaw, pitch, roll]` key/values.
Also, use `model.discriminator.camera_cond=true model.discriminator.camera_cond_drop_p=0.5` command line arguments (or simply override them in the config).
If you want to train on a custom dataset, then create the config for it like `configs/dataset/my_dataset.yaml`, specifying the necessary parameters (see other configs to get the idea on what should be specified).

## Training

### Commands

To launch training, run:
```
python src/infra/launch.py hydra.run.dir=. desc=<EXPERIMENT_NAME> dataset=<DATASET_NAME> dataset.resolution=<DATASET_RESOLUTION>  model.training.gamma=0.1 training.resume=null
```

To continue training, launch:
```
python src/infra/launch.py hydra.run.dir=. experiment_dir=<PATH_TO_EXPERIMENT> training.resume=latest
```

For Megascans Plants, we used class labels (for all the models). To enable class-conditional training, use `training.use_labels=true` command line argument (class annotations are located in `dataset.json`):
```
python src/infra/launch.py hydra.run.dir=. desc=default dataset=megascans_plants dataset.resolution=256  training.gamma=0.05 training.resume=null training.use_labels=true
```

### Tips and tricks
- The model is quite sensitive to the `gamma` hyperparameter ([R1 regularization](https://paperswithcode.com/method/r1-regularization)). If you have the capacity to optimize for it, this might improve the results. We would recommend doing this if you train with a different patch size (i.e., not the default 64x64 one). We use `gamma=0.05` everywhere for quite some time, but then found that `gamma=0.1` works slightly better (~10% in terms of FID) on FFHQ.
- Make sure that camera angles are zero-centered. Our tri-plane projection implementation works reliably only with zero-centered cameras.
- If you train on a new dataset and it does not quite work, try reducing the resolution to 128x128 (or even 64x64). This makes the task easier and the experiments run much faster, which is helpful for debugging. If the model does not converge on 128x128 --- open an issue and provide the details of your dataset, we'll try to help.
- Training with the background separated uses an additional network to model the background. This can be quite heavy that's why we use a quite tiny network (4 layers of 128 neurons with 8 evaluations per ray) to do this. This is likely why our FID scores decrease 10-15% for it. That's why we recommend training with the separated background only after making sure that the normal training works for you.
- If your dataset contains 3D biases (like FFHQ), then we would suggest using Generator Pose Conditioning (GPC) from EG3D: it conditions the generator on the camera pose angles in 50% of the cases during training. You can enable this by specifying `model.generator.camera_cond=true`. Also, do not forget to enable camera pose conditioning for the discriminator as well by providing `model.discriminator.camera_cond=true`.

### Training on a cluster or with slurm

If you use slurm or some cluster training, you might be interested in our cluster training infrastructure.
We leave our A100 cluster config in `configs/env/raven.yaml` as an example on how to structure the config environment in your own case.
In principle, we provide two ways to train: locally and on cluster via slurm (by passing `slurm=true` when launching training).
By default, the simple local environment is used, but you can switch to your custom one by specifying `env=my_env` argument (after your created `my_env.yaml` config in `configs/env`).

<!-- ## Checkpoints

We release the following generator checkpoints for our model:
- [FFHQ 512x512 checkpoint]()
- [Cats 256x256 checkpoint]()
- [M-Plants 256x256 checkpoint]()
- [M-Food 256x256 checkpoint]() -->

## Evalution
At train time, we compute FID only on 2,048 fake images (versus all the available real images), since generating 50,000 images takes too long.
To compute FID for 50k fake images after the training is done, run:
```
python scripts/calc_metrics.py hydra.run.dir=. ckpt.network_pkl=<CKPT_PATH> data=<PATH_TO_DATA> mirror=true gpus=4 metrics=fid50k_full img_resolution=<IMG_RESOLUTION>
```
If you have several checkpoints for the same experiment, you can alternatively pass `ckpt.networks_dir=<CKPTS_DIR>` instead of `ckpt.network_pkl=<CKPT_PATH>`.
In this case, the script will find the best checkpoint out of all the available ones (measured by FID@2k) and computes the metrics for it.

## Inference and visualization

Doing visualizations for a 3D GANs paper is pretty tedious, and we tried to structure/simplify this process as much as we could.
We created a scripts which runs the necessary visualization types, where each visualization is defined by its own config.
Below, we will provide several visualization types, the rest of them can be found in `scripts/inference.py`.
Everywhere we use a direct path to a checkpoint via `ckpt.network_pkl`, but often it is easier to pass `ckpt.networks_dir` which should lead to a directory with checkpoints of your experiment --- the script will then take the best checkpoint based on the `fid2k_full` metric.
You can combine different visualization types (location in `configs/scripts/vis`) with different camera paths (location in `configs/scripts/camera`).

Please see `configs/scripts/inference.yaml` for the available parameters and what they influence.

### Main grid visualization

It's the visualization type we used for the teaser (as an image).
```
python scripts/inference.py hydra.run.dir=. ckpt.network_pkl=<CKPT_PATH> vis=front_grid camera=points output_dir=<OUTPUT_DIR> num_seeds=16 truncation_psi=0.7
```

### A "zoom-in/out-and-fly-around" video

It's the visualization type we used for the teaser (as a video).
```
python scripts/inference.py hydra.run.dir=. ckpt.network_pkl=<CKPT_PATH> vis=video camera=front_circle output_dir=<OUTPUT_DIR> num_seeds=16 truncation_psi=0.7
```

## Reporting bugs and issues

If something does not work as expected — please create an issue or email `iskorokhodov@gmail.com`.

## License

This repo is built on top of [StyleGAN3](https://github.com/nvlabs/stylegan3) and [INR-GAN](https://github.com/universome/inr-gan).
This is why it is likely to be restricted by the [NVidia license](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt) (but no idea to what extent).

<!-- ## Rendering Megascans

To render the megascans, obtain the necessary models from the [website](https://quixel.com/megascans/home), convert them into GLTF, create a `enb.blend` Blender environment, and then run:
```
blender --python render_dataset.py env.blend --background
```
The rendering config is located in `render_dataset.py`. -->

## Bibtex

```
@article{epigraf,
    title={EpiGRAF: Rethinking training of 3D GANs},
    author={Skorokhodov, Ivan and Tulyakov, Sergey and Wang, Yiqun and Wonka, Peter},
    journal={arXiv preprint arXiv:2206.10535},
    year={2022},
}
```
