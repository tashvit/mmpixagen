"""
Create a spritesheet using a checkpoint, source image, driver video

This script is based on animate.py, run.py and demo.py of FOMM code (same directory)

Additional changes are located in frames_dataset
"""

from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from skimage.transform import resize
from tqdm.auto import tqdm

from animate import normalize_kp
from frames_dataset import FramesDataset
from modules import platform_util
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        platform_util.model_to_gpu(generator, [0])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        platform_util.model_to_gpu(kp_detector, [0])

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=platform_util.device([0]))

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator, device_ids=[platform_util.device([0])])
        kp_detector = DataParallelWithCallback(kp_detector, device_ids=[platform_util.device([0])])

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,
                   cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.to(platform_util.device([0]))
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.to(platform_util.device([0]))
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='hero.png', help="path to source image")
    parser.add_argument("--driver", default='driver', dest='driver', help="path to driving video source")
    parser.add_argument("--result", default='result.png', help="path to output")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
    opt.cpu = True  # Force CPU

    # Get source png image
    source_image = imageio.imread(opt.source_image)
    source_image = resize(source_image, (256, 256))[..., :3]

    # Get Driving video
    dataset = FramesDataset(is_train=False, root_dir=opt.driver, is_spritesheet=True)
    driving_video = dataset[0]['video']
    print("Frame count =", len(driving_video))
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    # Evaluate
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True,
                                 adapt_movement_scale=True, cpu=opt.cpu)
    predictions = np.concatenate(predictions, axis=1)
    predictions = (255 * predictions).astype(np.uint8)
    # Export as PNG
    imageio.imsave(opt.result, predictions)


if __name__ == "__main__":
    main()
