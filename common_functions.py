# -- Common/core functions for notebooks and scripts --
import os
import sys
from typing import NamedTuple, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from resizeimage import resizeimage

from thirdparty.pix2pix.data.base_dataset import get_params, get_transform
from thirdparty.pix2pix.models import create_model
from thirdparty.pix2pix.options.test_options import TestOptions
from thirdparty.pix2pix.util import util


class Spec(NamedTuple):
    """
    Model Spec has been created to ensure
     that the file-paths for data and model metadata is available for easy access
    """
    name: str
    raw_data_dir: str
    prepared_data_dir: str
    direction: str
    model_type: str
    dataset_path_prefix: str = ""
    temp_directory: str = ""
    # Fomm specific
    fomm_endpoint: str = ""
    fomm_conf: str = ""
    fomm_checkpoint_data: str = ""
    fomm_driver: str = ""


# Spec object is created for different iterations of models such as A1, A2, A3, B1, C1, D1, etc.
#   these objects contain metadata to load and evaluate models

A1 = Spec(
    name="pixel_A1",
    raw_data_dir="datasets/unprepared_data/input_a1_a2",
    prepared_data_dir="datasets/model_a_data",
    direction="BtoA",
    model_type="pix2pix",
    temp_directory="datasets/unprepared_data/pairs",
)
A2 = Spec(
    name="pixel_A2",
    raw_data_dir=A1.raw_data_dir,
    prepared_data_dir=A1.prepared_data_dir,
    direction=A1.direction,
    model_type="pix2pix"
)
A3_BASE = Spec(
    name="pixel_A3",
    raw_data_dir="datasets/unprepared_data/input_a3/transfer_learning",
    prepared_data_dir="datasets/model_a3_data",
    direction="AtoB",
    model_type="pix2pix",
    dataset_path_prefix="tl_"
)
A3 = Spec(
    name="pixel_A3",
    raw_data_dir="datasets/unprepared_data/input_a3/characters",
    prepared_data_dir="datasets/model_a3_data",
    direction="AtoB",
    model_type="pix2pix",
    dataset_path_prefix="pix2pix_"
)
B1 = Spec(
    name="pixel_B1",
    raw_data_dir="datasets/unprepared_data/input_bcd",
    prepared_data_dir="datasets/model_b_data",
    direction="AtoB",
    model_type="pix2pix"
)
C1 = Spec(
    name="pixel_C1",
    raw_data_dir=B1.raw_data_dir,
    prepared_data_dir="datasets/model_c_data",
    direction="AtoB",
    model_type="pix2pix"
)
D1 = Spec(
    name="pixel_D1",
    raw_data_dir=B1.raw_data_dir,
    prepared_data_dir="datasets/model_d_data",
    direction="AtoB",
    model_type="pix2pix"
)
# Note: E1 raw data is generated with generate_E1_E2.py
E1 = Spec(
    name="sprite_E1",
    raw_data_dir="datasets/unprepared_data/input_e/walk",
    prepared_data_dir="datasets/model_e_data",
    direction="AtoB",
    model_type="pix2pix"
)
E2 = Spec(
    name="sprite_E2",
    raw_data_dir=E1.raw_data_dir,
    prepared_data_dir="datasets/model_e2_data",
    direction="",
    model_type="fomm",
    fomm_endpoint="thirdparty/fomm/create_spritesheet.py",
    fomm_conf="thirdparty/fomm/config/tinyhero-256.yaml",
    fomm_checkpoint_data="thirdparty/fomm/log/tinyhero-256 18_08_24_18.36.14/00000009-checkpoint.pth.tar",
    fomm_driver="thirdparty/fomm/data/spritesheet-driver"
)

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_PATH = os.path.join(ROOT_PATH, "checkpoints")


class Model:
    def __init__(self, spec: Spec, model_type="pix2pix"):
        self._spec = spec
        self._model_type = model_type
        self._model = None
        if self._model_type == "pix2pix":
            self._model = self._load_pix2pix()
        # Nothing to load for the E2 model as it is executed as a separate script

    def _load_pix2pix(self):
        # This section is based on the evaluation notebooks such as eval_A2_A3, eval_B1_C1_D1
        args = ["--dataroot", self._spec.prepared_data_dir,
                "--name", self._spec.name, "--model", self._spec.model_type,
                "--direction", self._spec.direction, "--phase", "test", "--netG",
                "unet_64", "--netD", "pixel", "--load_size", "64",
                "--crop_size", "64", "--display_winsize", "64", "--checkpoints_dir", CHECKPOINT_PATH, ]

        # Parsed test options
        self.opt = TestOptions().parse(args)
        self.opt.num_threads = 0
        self.opt.batch_size = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True
        self.opt.display_id = -1
        model = create_model(self.opt)
        model.setup(self.opt)
        return model

    def evaluate(self, image_path=None, image=None, output_image_path=None, driver="front") -> Image:
        """
        Evaluate model on given input
        :param image_path: Image path to load image from (either this or image need to be provided)
        :param image: 64x64 magenta background image (Pillow) to evaluate
            (either this or image_path need to be provided)
            for fomm models images are converted to 256x256
        :param output_image_path: Path to save output image (if None, will not save to this location)
        :param driver: Driver images directory for FOMM model. One of (front, back, left, right)
        :return: Image (Pillow) evaluated on given input
        """
        if image is None and image_path is None:
            raise ValueError("Either image or image path must be specified")
        if image_path:
            input_image = load_64x64_with_magenta_bg(image_path).convert('RGB')
        else:
            input_image = image
        # --------------------------------------------
        if self._model_type == "fomm":
            if not driver or driver not in ["front", "back", "left", "right"]:
                raise ValueError("Invalid driver name")
            if not image_path:
                raise ValueError("Image path must be specified")
            return self._eval_fomm(image_path, output_image_path, driver)
        if self._model_type == "pix2pix":
            return self._eval_pix2pix(input_image, output_image_path)
        raise ValueError("Model type not supported")

    def _eval_fomm(self, image, output_image_path, driver):
        endpoint = os.path.abspath(os.path.join(ROOT_PATH, self._spec.fomm_endpoint))
        conf = os.path.abspath(os.path.join(ROOT_PATH, self._spec.fomm_conf))
        checkpoint_data = os.path.abspath(os.path.join(ROOT_PATH, self._spec.fomm_checkpoint_data))
        driver_path = os.path.abspath(os.path.join(ROOT_PATH, self._spec.fomm_driver, driver))
        base_path = os.path.dirname(endpoint)
        # current_dir = os.curdir
        # os.chdir(base_path)
        command = (f"{sys.executable} {endpoint} --config '{conf}' --checkpoint '{checkpoint_data}'"
                   f" --source_image '{image}' --driver '{driver_path}' --result {output_image_path}")
        os.system(command)
        # os.chdir(current_dir)
        return Image.open(output_image_path).convert('RGB')

    def _eval_pix2pix(self, input_image, output_image_path):
        tensor = self._im_to_tensor(input_image)
        self._model.set_input({"A": tensor, "B": tensor, "A_paths": "path", "B_paths": "path"})
        self._model.test()
        visuals = self._model.get_current_visuals()
        generated = visuals['fake_B']
        im = util.tensor2im(generated)
        im = Image.fromarray(im)
        if output_image_path:
            im.save(output_image_path)
            im = Image.open(output_image_path)
        else:
            im.save("temp.png")
            im = Image.open("temp.png")
            os.unlink("temp.png")
        return im

    def _im_to_tensor(self, im):
        transform_params = get_params(self.opt, im.size)
        transformer = get_transform(self.opt, transform_params, grayscale=False)
        return torch.unsqueeze(transformer(im), 0)


def load_model(spec: Spec) -> Model:
    return Model(spec, model_type=spec.model_type)


def create_combined_images(arguments: str):
    script = get_path("thirdparty/pix2pix/datasets/combine_A_and_B.py")
    command_line = f"python {script} {arguments}"
    print("----------------------------------------------------------")
    print(command_line)
    os.system(command_line)


def get_path(*paths) -> str:
    # based on
    # reference: https://stackoverflow.com/a/59131433
    target = str(os.path.join(ROOT_PATH, *paths))
    if os.path.isdir(target):
        if target[-1] == "/" or target[-1] == "\\":
            return target
        return target + os.path.sep
    return target


def create_sketch(cv_img, levels=3, magenta_bg=True):
    image = cv_img
    # Create a basic sketch
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    invert_image = cv2.bitwise_not(gray_image)
    blur_image = cv2.GaussianBlur(invert_image, (5, 5), 0)
    invert_blur = cv2.bitwise_not(blur_image)
    sketch = cv2.divide(gray_image, invert_blur, scale=256.0)
    # perform edge detection on the sketch
    b1 = cv2.GaussianBlur(sketch, (5, 5), 0)
    b2 = b1

    for _ in range(0, levels - 1):
        b2 = cv2.GaussianBlur(b2, (5, 5), 0)
    edge = cv2.Canny(b2, 50, 100)
    edge_inv = cv2.bitwise_not(edge)
    edge_inv = cv2.cvtColor(edge_inv, cv2.COLOR_GRAY2BGR)
    if magenta_bg:
        # Define the white color range
        lower_white = np.array([254, 254, 254], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        # Create a mask to detect white regions
        mask = cv2.inRange(edge_inv, lower_white, upper_white)

        # Change white regions to magenta
        edge_inv[mask == 255] = [255, 0, 255]
    return edge_inv


def write_image(image, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(filename)
    else:
        cv2.imwrite(filename, image)


def pil_to_opencv(pil_img):
    """
    Converts PIL image to OpenCV image
    :param pil_img: pillow image
    :return: OpenCV image
    """
    # Image conversion functions are taken from
    # https://stackoverflow.com/a/74205492
    cv2_img = np.array(pil_img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img


def opencv_to_pil(cv2_img):
    """
    Converts OpenCV image to PIL image
    :param cv2_img: OpenCV image
    :return: pillow image
    """
    # Image conversion functions are taken from
    # https://stackoverflow.com/a/74205492
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    return pil_img


def background_to_alpha(cv_image):
    # Taken from https://stackoverflow.com/a/63003020
    # load image
    img = cv_image

    # convert to graky
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    #
    # # negate mask
    mask = 255 - mask

    # apply morphology to remove isolated extraneous noise
    # use border constant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result


def images_to_sprite_sheet(images, max_horiz=8):
    # From https://stackoverflow.com/a/46877433
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')  # noqa
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


def turn_pil_image_to_magenta_bg(pil_image):
    cv_img = pil_to_opencv(pil_image)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
    alpha_bg = background_to_alpha(cv_img)
    img = Image.fromarray(alpha_bg, mode='RGBA')
    magenta_bg = Image.new("RGBA", img.size, (255, 0, 255, 255))
    img_with_magenta_bg = Image.alpha_composite(magenta_bg, img)
    final_img = img_with_magenta_bg.convert("RGB")
    return final_img


def load_with_magenta_background(input_img_path):
    """
    Load image with magenta background
    :param input_img_path: input image path
    :return: image with magenta background
    """
    # Convert input image to RGBA
    img = Image.open(input_img_path).convert("RGBA")
    # New image with a magenta background
    magenta_bg = Image.new("RGBA", img.size, (255, 0, 255, 255))
    # Get input image on top of magenta image
    img_with_magenta_bg = Image.alpha_composite(magenta_bg, img)
    # Convert back to RGB to remove alpha channel
    final_img = img_with_magenta_bg.convert("RGB").convert("RGBA")
    return final_img


def resize_to_256(image_path):
    """
    Turn given image to 256x256
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256), resample=Image.Resampling.NEAREST)
    img.save(image_path)


def load_64x64_with_magenta_bg(image_path):
    """
    Load and resize image with magenta background to 64x64
    :param image_path: image path
    :return: resized image with magenta background
    """
    image = load_with_magenta_background(image_path)
    res = resizeimage.resize_contain(image, [64, 64], bg_color=(255, 0, 255, 255))
    return res


def to_64x64_magenta(image_path, output_path):
    """
    Load and save 64x64 image with magenta background to a given file path
    :param image_path: image path
    :param output_path: location to write magenta background image
    """
    res = load_64x64_with_magenta_bg(image_path)
    # Save image to output path
    res.save(output_path)


def turn_magenta(input_img_path, output_img_path):
    """
    Load and save image with magenta background to a given file path
    :param input_img_path: input image path
    :param output_img_path: location to write magenta background image
    :return:
    """
    final_img = load_with_magenta_background(input_img_path)
    # Save image to output directory
    final_img.save(output_img_path)


def load_test_sheet(sheet_path):
    """
    Load sprite sheet images in given path as a single spritesheet image
    :param sheet_path: path to sprite sheet images
    :return: single 8x4 sprite sheet image (size 512x256), front_facing_image
    """
    sheet_order = [
        # front
        8, 9, 10, 11, 12, 13, 14, 15,
        # right
        24, 25, 26, 27, 28, 29, 30, 31,
        # back
        0, 1, 2, 3, 4, 5, 6, 7,
        # left
        16, 17, 18, 19, 20, 21, 22, 23,
    ]
    images = []
    for i in sheet_order:
        filename = os.path.join(sheet_path, f"{i:02d}.png")
        images.append(Image.open(filename))
    sheet = images_to_sprite_sheet(images)
    front_facing_image = os.path.join(sheet_path, "08.png")  # a front facing image
    if sheet.width > 64 * 8:
        return sheet.resize((sheet.width // 4, sheet.height // 4),
                            resample=Image.Resampling.NEAREST), front_facing_image
    else:
        return sheet, front_facing_image


class Inference:
    def __init__(self, get_temp_file):
        """
        Inference facade
        :param get_temp_file: function to get a temporary image file (takes an optional prefix)
        """
        self._sketch_to_image = load_model(A3)
        self._front_to_right = load_model(B1)
        self._front_to_back = load_model(C1)
        self._right_to_left = load_model(D1)
        self._e1_next_sprite = load_model(E1)
        self._e2_sprite = load_model(E2)
        self._get_temp = get_temp_file

    def sketch_to_image(self, sketch_path, output_path):
        return self._sketch_to_image.evaluate(image_path=sketch_path, output_image_path=output_path)

    def make_directional_images(self, input_image) -> Tuple[Image, Image, Image, Image]:
        """
        Use given input image file to make (back, front, left, right) image tuple
        :param input_image: path to input image file (front facing)
        """
        front_facing = load_64x64_with_magenta_bg(input_image).convert('RGB')
        right_facing = self._front_to_right.evaluate(image=front_facing)
        back_facing = self._front_to_back.evaluate(image=front_facing)
        left_facing = self._right_to_left.evaluate(image=right_facing)
        return back_facing, front_facing, left_facing, right_facing

    def directional_images_sheet(self, input_image) -> Image:
        images = list(self.make_directional_images(input_image))
        return images_to_sprite_sheet(images)

    def create_sheet_e1(self, input_image, output_image=None):
        """
        Create a spritesheet using E1 model
        :param input_image: path to input image file
        :param output_image: path to output image file
        """
        back_facing, front_facing, left_facing, right_facing = self.make_directional_images(input_image)
        direction_images = [front_facing, right_facing, back_facing, left_facing]
        # For each directional image, attempt to generate individual sprite images using next_sprite_image model (E1)
        sprites = []
        for single_image in direction_images:
            curr = single_image
            sprites.append(curr)
            for i in range(7):
                curr = self._e1_next_sprite.evaluate(image=curr)
                sprites.append(curr)
        sheet = images_to_sprite_sheet(sprites)
        if output_image:
            sheet.save(output_image)
        return sheet

    def create_sheet_e2(self, input_image, output_image=None):
        """
        Create a spritesheet using E2 model
        :param input_image: path to input image file
        :param output_image: path to output image file
        """
        back_facing, front_facing, left_facing, right_facing = self.make_directional_images(input_image)
        starting_images = [front_facing, right_facing, back_facing, left_facing]
        images = []
        for direction, directional_image in zip(["front", "right", "back", "left"], starting_images):
            input_temp = self._get_temp(prefix=direction + "_i")
            directional_image.resize((256, 256), resample=Image.Resampling.NEAREST).save(input_temp)
            single_direction = self._get_temp(prefix=direction + "_o")
            img = self._e2_sprite.evaluate(image_path=input_temp, output_image_path=single_direction, driver=direction)
            images.append(img)
        sheet = images_to_sprite_sheet(images, max_horiz=1)
        sheet = sheet.resize((sheet.width // 4, sheet.height // 4), resample=Image.Resampling.NEAREST)
        if output_image:
            sheet.save(output_image)
        return sheet
