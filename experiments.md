# Experiments

### Model A1

* Training is carried out with default parameters
* Image size - resized to 256x256
* Standard PatchGAN
* unet_256 Generator

```bash
python pix2pix/train.py --dataroot ./pix2pix_data/AB --name pixel --model pix2pix --direction BtoA
```

### Model A2

* Generator - 'unet_64' 
* Discriminator is 1x1 PatchGAN/pixel 
* Do not resize images to 256x256

```bash
python pix2pix/train.py --dataroot ./pix2pix_data/AB --name pixel_A2 --model pix2pix --direction BtoA --netG unet_64 --netD pixel --load_size 64 --crop_size 64 --display_winsize 64
```