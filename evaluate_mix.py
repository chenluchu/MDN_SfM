from tqdm import tqdm
from imageio import imwrite
import cv2

from torchvision import transforms
from torch.utils.data import DataLoader

from networks.layers import get_scale_factor, transformation_from_parameters
from datasets.kitti_dataset import KITTISegDataset
from options_eval import MonodepthOptions
from loss_functions import LossModule
from utils import *
from eval_utils import load_models, get_quantitative_results
from cityscapesscripts.helpers.labels import trainId2label
from detectron2CustomDataset import kitti_decode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def evaluate(options):
    """Evaluates a pretrained model using a specified test set
    """

    # environment preparation
    opt = options.parse()
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    output_images_dir = os.path.join(opt.eval_out_dir, "mobile", "masks_{}_{}".format(opt.version, opt.idx))
    create_dir(output_images_dir)

    # calculation preparation
    img_height = opt.height
    img_width = opt.width
    opt_new = options.parse()
    opt_new.batch_size = 1
    epipolar_loss = LossModule(opt_new).to(device)
    scale_factor = get_scale_factor(1, img_height, img_width).to(device)
    resize = transforms.Resize((img_height, img_width))

    print("-> Computing predictions with size {}x{}".format(img_width, img_height))
    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    # dataset and models preparation
    eval_dataset = KITTISegDataset(opt.data_root, kitti_decode, img_height, img_width)
    eval_loader = DataLoader(eval_dataset, 1, False, num_workers=opt.num_workers, pin_memory=True)  # opt.num_workers
    motion_net, pose_net, mobile_net, instance_model, cfg = load_models(opt, device, detectron2=True)

    metrics = []
    with torch.no_grad():
        for j, inputs in enumerate(tqdm(eval_loader)):

            tgt = inputs[('color', 0)].to(device)
            next_tgt = inputs[('color', 1)].to(device)
            instance_img = inputs["instance_img"]
            inv_K = inputs['inv_K'].to(device)

            # PREDICTIONS
            axis_angle, translation = pose_net(tgt, next_tgt)
            flow, features_enc = motion_net(tgt, next_tgt)
            mobile = mobile_net(features_enc, axis_angle, translation)
            detectron2_input = get_detectron2_input(instance_img, cfg)
            detectron2_out = instance_model(detectron2_input)[0]["instances"]

            # Get results
            full_flow = scale_factor * flow[('flow', 0, 0)]
            cam_T_cam = transformation_from_parameters(axis_angle, translation)
            mobile_mask = mobile[("mobile", 0, 0)]

            _, post_epip, ori_epip = epipolar_loss.epipolar_loss(
                full_flow, mobile_mask, detectron2_out,
                inv_K, cam_T_cam[:, :3, :3], cam_T_cam[:, :3, -1])

            # post process of result
            image = instance_img[0].type(torch.uint8).permute(2, 0, 1)
            classes = detectron2_out.pred_classes.cpu().numpy()
            labels = [trainId2label[cls + 1].name for cls in classes]
            colors = [trainId2label[cls + 1].color for cls in classes]
            seg_img = draw_bounding_boxes(image, detectron2_out.pred_boxes.tensor.type(torch.int),
                                          labels, colors, width=2, font_size=2)

            gt_mask = imread(os.path.join(opt.gt_mask_path, "{}.png".format(j))) / 255
            metric = get_quantitative_results(
                binary_image(mobile_mask[0].permute(1, 2, 0).repeat(1, 1, 3)).cpu().numpy(), gt_mask)
            metrics.append(metric)

            if opt.save_pred_masks:
                # SAVE MAPS
                s = resize(seg_img).permute(1, 2, 0).cpu().numpy()
                m = mobile_mask[0].permute(1, 2, 0).repeat(1, 1, 3)
                bm = 255 * binary_image(m).cpu().numpy()
                m = 255 * m.cpu().numpy()
                e = ori_epip.permute(0, 2, 3, 1).cpu()[0, :].numpy()
                e = 255 * (e / e.max())
                pe = post_epip.permute(0, 2, 3, 1).cpu()[0, :].numpy()
                pe = 255 * (pe / pe.max())
                viz = np.hstack([s, bm, m, pe, e]).astype(np.uint8)
                file = os.path.join(output_images_dir, "{}.png".format(j))
                imwrite(file, viz)

            del inputs
    if opt.save_pred_masks:
        print("Evaluation save to --> ", output_images_dir)

    print(np.mean(np.array(metrics), axis=0, keepdims=True))
    print("\n-> Done!")


if __name__ == '__main__':
    options = MonodepthOptions()
    evaluate(options)
