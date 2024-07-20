# This code is based on random_character.py script
#  from https://github.com/YingzhenLi/Sprites
#  all content of 'Sprites' repo is copied to 'sprite_generation' for use in mmpixagen
#
#  Modifications
#   - python 3 support
#   - keep only walk animation
#   - extract magic values to have meaningful names
#   - simplified code
#   - magenta background for generated frames

import os

import numpy as np
from PIL import Image
from tqdm import tqdm

FRAME_COUNT = 8
RIGHT_WALK_START = 144
FRONT_WALK_END = 139
FRONT_WALK_START = 131
LEFT_WALK_END = 126
LEFT_WALK_START = 118
BACK_WALK_START = 104
BACK_WALK_END = 113
RIGHT_WALK_END = 152


def gen_char(body, bottom, top, hair):
    """
    Generate a character from given body, bottom, top and hair options
    :param body: index for body type
    :param bottom: index for clothing (bottom wear)
    :param top:  index for clothing (top wear)
    :param hair: index for hair type
    """
    # Get different sections of the image and add it to attributes
    attributes = {'body': str(body), 'bottomwear': str(bottom),
                  'topwear': str(top), 'hair': str(hair)}
    img_list = []
    for attr in ['body', 'bottomwear', 'topwear', 'hair']:
        path = attr + '/'
        filename = attributes[attr] + '.png'
        img_list.append(Image.open(path + filename))
    # shoes
    img_list.append(Image.open('shoes/1.png'))

    # then merge all!
    # Use magenta background
    f = Image.new('RGBA', img_list[0].size, (255, 0, 255, 255))
    for i in range(len(img_list)):
        f = Image.alpha_composite(f, img_list[i].convert('RGBA'))

    # save image
    classname = str(body) + str(bottom) + str(top) + str(hair)
    f.save(f'{classname}.png')

    img = Image.open(f'{classname}.png')
    # crop to 64 * 64
    width = 64
    height = 64
    image_width, image_height = img.size
    n_width = image_width // width
    n_height = image_height // height
    path = 'frames/'
    if not os.path.exists(path):
        os.makedirs(path)
    os.makedirs(path + "/walk/", exist_ok=True)

    for j in range(n_height):
        for i in range(n_width):
            ind = j * n_width + i
            # for walk
            if BACK_WALK_START <= ind < RIGHT_WALK_END:
                box = (i * width, j * height, (i + 1) * width, (j + 1) * height)
                if BACK_WALK_START <= ind < BACK_WALK_END:
                    pose = 'back'
                    k0 = (ind - BACK_WALK_START + 1) % (FRAME_COUNT + 1)
                    # There is a problem with back facing walk's 2nd frame, and it needs to be ignored
                    #   this also make back facing walk to have 8 frames like left, front, right
                    k = k0
                    if k0 == 1:
                        continue
                    elif k0 >= 2:
                        k = k0 - 1
                    crop_and_save_image(box, classname, img, k, path, pose)
                if LEFT_WALK_START <= ind < LEFT_WALK_END:
                    pose = 'left'
                    k = (ind - LEFT_WALK_END + 1) % FRAME_COUNT
                    crop_and_save_image(box, classname, img, k, path, pose)
                if FRONT_WALK_START <= ind < FRONT_WALK_END:
                    pose = 'front'
                    k = (ind - FRONT_WALK_START + 1) % FRAME_COUNT
                    crop_and_save_image(box, classname, img, k, path, pose)
                if RIGHT_WALK_START <= ind < RIGHT_WALK_END:
                    pose = 'right'
                    k = (ind - RIGHT_WALK_START + 1) % FRAME_COUNT
                    crop_and_save_image(box, classname, img, k, path, pose)
    # now remove the png files
    os.remove(f'{classname}.png')


def crop_and_save_image(rect_to_crop, image_identifier, image_object, frame, file_path, pose):
    a = image_object.crop(rect_to_crop)
    a.convert('RGB')
    a.save(f"{file_path}walk/{image_identifier}_{pose}_{frame}.png")


def main():
    n_class = 6
    seed_list = range(0, n_class ** 4)
    # TQDM allow author to view the progress of the frame generation for model (E1, E2) training.
    for seed_value in tqdm(seed_list):
        seed = seed_value
        body = int(seed / n_class ** 3)
        seed = int(np.mod(seed, n_class ** 3))
        bottom = int(seed / n_class ** 2)
        seed = int(np.mod(seed, n_class ** 2))
        top = int(seed / n_class)
        hair = int(np.mod(seed, n_class))

        gen_char(body, bottom, top, hair)


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), 'sprite_generation'))
    main()
