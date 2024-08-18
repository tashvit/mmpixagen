# Experiments

### Model A1

* Training is carried out with default parameters
* Image size - resized to 256x256
* Standard PatchGAN
* unet_256 Generator

```bash
python -m thirdparty.pix2pix.train --dataroot ./datasets/model_a_data/combined --name pixel_A1 --model pix2pix --direction BtoA
```

### Model A2

* Generator - 'unet_64' 
* Discriminator is 1x1 PatchGAN/pixel 
* Do not resize images to 256x256

```bash
python -m thirdparty.pix2pix.train --dataroot ./datasets/model_a_data/combined --name pixel_A2 --model pix2pix --direction BtoA --netG unet_64 --netD pixel --load_size 64 --crop_size 64 --display_winsize 64
```

### Model A3

* Generator - 'unet_64' 
* Discriminator is 1x1 PatchGAN/pixel 
* Does not resize images to 256x256
* Two-step training
  * Mixed data - Anime Faces, Pokemon, Some game assets (Kenny.nl)
  * Pixel art character front facing data
* Epochs are customized
* Continue training from mixed data.

#### Step 1 - Train on subset of Anime Face data + Pokemon data

```bash
python -m thirdparty.pix2pix.train --dataroot ./datasets/model_a3_data/tl_combined --name pixel_A3 --model pix2pix --direction AtoB --netG unet_64 --netD pixel --load_size 64 --crop_size 64 --display_winsize 64 --n_epochs 40 --n_epochs_decay 40
```

#### Step 2 - Train on pixel art characters

```bash
python -m thirdparty.pix2pix.train --dataroot ./datasets/model_a3_data/pix2pix_combined --name pixel_A3 --model pix2pix --direction AtoB --netG unet_64 --netD pixel --load_size 64 --crop_size 64 --display_winsize 64 --continue_train --epoch_count 41 --n_epochs 60 --n_epochs_decay 60
```

### Model B1

* Generator - 'unet_64' 
* Discriminator is 1x1 PatchGAN/pixel 
* Do not resize images to 256x256

```bash
python -m thirdparty.pix2pix.train --dataroot ./datasets/model_b_data/combined --name pixel_B1 --model pix2pix --direction AtoB --netG unet_64 --netD pixel --load_size 64 --crop_size 64 --display_winsize 64 --no_flip
```

### Model C1

* Generator - 'unet_64' 
* Discriminator is 1x1 PatchGAN/pixel 
* Do not resize images to 256x256

```bash
python -m thirdparty.pix2pix.train --dataroot ./datasets/model_c_data/combined --name pixel_C1 --model pix2pix --direction AtoB --netG unet_64 --netD pixel --load_size 64 --crop_size 64 --display_winsize 64 --no_flip
```

### Model D1
* Generator - 'unet_64' 
* Discriminator is 1x1 PatchGAN/pixel 
* Do not resize images to 256x256

```bash
python -m thirdparty.pix2pix.train --dataroot ./datasets/model_d_data/combined --name pixel_D1 --model pix2pix --direction AtoB --netG unet_64 --netD pixel --load_size 64 --crop_size 64 --display_winsize 64 --no_flip
```

### Model E1

* Generator - 'unet_64' 
* Discriminator is 1x1 PatchGAN/pixel 
* Do not resize images to 256x256

```bash
python -m thirdparty.pix2pix.train --dataroot ./datasets/model_e_data/combined --name sprite_E1 --model pix2pix --direction AtoB --netG unet_64 --netD pixel --load_size 64 --crop_size 64 --display_winsize 64 --no_flip
```

### Model E2

* FOMM model

```bash
cd thirdparty/fomm/data
ln -s ../../../datasets/model_e2_data tinyhero-64
cd ..
python run.py --config config/tinyhero-256.yaml
```