from PIL import Image
import cv2
import numpy as np
from resizeimage import resizeimage
import os


# https://stackoverflow.com/a/74205492

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
    cv2_img = np.array(pil_img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img


def opencv_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    return pil_img


def load_with_magenta_background(input_img_path):
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
    image = load_with_magenta_background(image_path)
    res = resizeimage.resize_contain(image, [64, 64], bg_color=(255, 0, 255, 255))
    return res


def to_64x64_magenta(image_path, output_path):
    res = load_64x64_with_magenta_bg(image_path)
    # Save image to output directory
    res.save(output_path)


def turn_magenta(input_img_path, output_img_path):
    final_img = load_with_magenta_background(input_img_path)
    # Save image to output directory
    final_img.save(output_img_path)
