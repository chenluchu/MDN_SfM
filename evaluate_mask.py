import os
from path import Path
from tqdm import tqdm
from skimage.transform import resize as imresize
import numpy as np
from imageio import imwrite

from options_eval import MonodepthOptions
from eval_utils import ValidationMobileMask, load_models, array_to_tensor, normalize
from eval_utils import ValidationMobileMaskMore as test_framework_more
from utils import *

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    # environment preparation
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    mobile_weights_folder = os.path.join(
        os.getcwd(), "log/{}/models/weights_{}".format(opt.version, opt.idx))
    opt.eval_out_dir = os.path.join(mobile_weights_folder, "predictions")

    output_dir = os.path.join(opt.eval_out_dir, "mobile")
    create_dir(output_dir)
    output_images_dir = os.path.join(output_dir, "mobile_masks")  #  "masks_eigen_full_val"/"zero_motion_epipolar"
    create_dir(output_images_dir)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)
    assert opt.eval_out_dir, 'Cannot find a folder at {}'.format(opt.eval_out_dir)
    print("-> Loading weights from \n{} \n{}".format(opt.load_weights_folder, mobile_weights_folder))

    # calculation preparation
    img_height = 128  # opt.height #net_dict['height']
    img_width = 416  # opt.width #net_dict['width']
    if opt.weights_init == 'pretrained':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        mean = (0.45, 0.45, 0.45)
        std = (0.225, 0.225, 0.225)
    print("-> Computing predictions with size {}x{}".format(img_width, img_height))

    # # Evaluation with split of eigen full val_files
    # fpath = os.path.join(os.path.dirname(__file__), "splits", "eigen_full", "val_files.txt")
    # files = readlines(fpath)
    # del files[481], files[725]  # [481, 726] don't have next frame
    # framework = test_framework_more(dataset_dir, files)

    # dataset and models preparation
    motion_net, pose_net, mobile_net = load_models(opt, device)
    dataset_dir = Path(opt.raw_dataset_dir)
    framework = ValidationMobileMask(dataset_dir)
    mobile_masks = []

    print("-> Evaluating:  Mono evaluation - using median scaling")
    with torch.no_grad():
        for j, sample in enumerate(tqdm(framework)):
            tgt_img, next_tgt = sample

            h, w, _ = tgt_img.shape
            if h != img_height or w != img_width:
                tgt_img = imresize(tgt_img, (img_height, img_width)).astype(np.float32)
                next_tgt = imresize(next_tgt, (img_height, img_width)).astype(np.float32)

            input_color = array_to_tensor(tgt_img)
            normalize(input_color, mean, std)
            motion_input_color = input_color.unsqueeze(0).to(device)

            input_color = array_to_tensor(next_tgt)
            normalize(input_color, mean, std)
            next_motion_input_color = input_color.unsqueeze(0).to(device)

            # PREDICTIONS
            axisangle, translation = pose_net(motion_input_color, next_motion_input_color)
            flows, features_enc = motion_net(motion_input_color, next_motion_input_color)
            mobiles = mobile_net(features_enc, axisangle, translation)
            mobile_mask = mobiles[("mobile", 0, 0)].expand_as(input_color)[0, ...]

            # SAVE MAPS
            if opt.save_pred_masks:
                binary_mask = binary_image(mobile_mask).permute(1, 2, 0).cpu().numpy()
                mask = mobile_mask.permute(1, 2, 0).cpu().numpy()
                viz = np.vstack([tgt_img, 255 * mask, 255 * binary_mask]).astype(np.uint8)
                file = os.path.join(output_images_dir, "{}.png".format(j))
                imwrite(file, viz)

    print("\n-> Done!")


if __name__ == '__main__':
    options = MonodepthOptions()
    evaluate(options.parse())
