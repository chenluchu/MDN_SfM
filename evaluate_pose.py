import os
from skimage.transform import resize as imresize
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from networks import transformation_from_parameters
from options_eval import MonodepthOptions
from utils import write_result, create_dir
from eval_utils import test_framework_KITTI, compute_pose_error, load_models, normalize, array_to_tensor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    # environment preparation
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    dataset_dir = os.path.join(opt.raw_dataset_dir, "odometry_data")
    sequence_set = ['09', '10']

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)
    assert opt.eval_out_dir, 'Cannot find a folder at {}'.format(opt.eval_out_dir)
    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # calculation preparation
    img_height = 128
    img_width = 416
    if opt.weights_init == 'pretrained':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        mean = (0.45, 0.45, 0.45)
        std = (0.225, 0.225, 0.225)

    print("-> Computing predictions with size {}x{}".format(img_width, img_height))

    # dataset and models preparation
    framework = test_framework_KITTI(dataset_dir, sequence_set, opt.sequence_length)
    pose_net = load_models(opt, device, motion=False, mobile=False)
    predictions_array = np.zeros((len(framework), opt.sequence_length, 3, 4))
    errors = np.zeros((len(framework), 2), np.float32)

    print("-> Evaluating")
    with torch.no_grad():
        for j, sample in enumerate(tqdm(framework)):
            imgs = sample['imgs']

            h, w, _ = imgs[0].shape
            if h != img_height or w != img_width:
                imgs = [imresize(img, (img_height, img_width)).astype(np.float32) for img in imgs]

            squence_imgs = []
            for i, img in enumerate(imgs):
                img = array_to_tensor(img)
                normalize(img, mean, std)
                squence_imgs.append(img.unsqueeze(0).to(device))

            global_pose = np.eye(4)
            poses = [global_pose[0:3, :]]

            # gt pose 矩阵是把t点投影到0点，pose_net是把t-1点投影到t点 --> M_gt_i = inv(M_0) @ inv(M_1) @ ... inv(M_i)
            # poses 总长度是 sequence_length， 第i矩阵是第i图片投影到第0图片的投影矩阵，其中第零个矩阵是单位矩阵(投影到自己)
            for iter in range(opt.sequence_length - 1):
                axisangle, translation = pose_net(squence_imgs[iter], squence_imgs[iter + 1])
                pose_mat = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                global_pose = global_pose @ np.linalg.inv(pose_mat.cpu().numpy()[0, ...])
                poses.append(global_pose[0:3, :])

            final_poses = np.stack(poses, axis=0)  # (sequence_length, 3, 4)
            predictions_array[j] = final_poses

            ATE, RE = compute_pose_error(sample['poses'], final_poses)
            errors[j] = ATE, RE

    output_dir = os.path.join(opt.eval_out_dir, "pose")
    create_dir(output_dir)

    if opt.save_pred_poses:
        output_path = os.path.join(output_dir, "poses.npy")
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, predictions_array)

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE', 'RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))
    result_txt = os.path.join(output_dir, "result.txt")
    f = open(result_txt, 'w')
    write_result(f, mean_errors, error_names)
    f.close()


if __name__ == '__main__':
    options = MonodepthOptions()
    evaluate(options.parse())
