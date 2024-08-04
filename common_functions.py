# -- Common/core functions for notebooks and scripts --
from PIL import Image
import cv2
import numpy as np
from resizeimage import resizeimage
import os
from typing import NamedTuple


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


# For different iterations of models such as A1, A2, A3, B1, C1, D1, etc.
#  spec is created

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
    name="pixel_E1",
    raw_data_dir="datasets/unprepared_data/input_e",
    prepared_data_dir="datasets/model_e_data",
    direction="AtoB",
    model_type="pix2pix"
)

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


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
