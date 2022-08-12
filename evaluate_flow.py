from path import Path
from tqdm import tqdm
from imageio import imwrite
import cv2
from skimage.transform import resize as imresize

from options_eval import MonodepthOptions
from eval_utils import ValidationFlow as test_framework
# from eval_utils import ValidationMobileMaskMore as test_framework
from eval_utils import load_models, get_output_img_name, compute_epe, normalize
from utils import *
from loss_utils import post_process_epipolar_1, post_pro_epipolar_weighted, get_epipolar_new
from networks.layers import get_scale_factor, transformation_from_parameters

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def compute_gt_epipolar(p1, p2, inv_K, M):
    Fundamental = torch.matmul(torch.transpose(inv_K, -2, -1), torch.matmul(M, inv_K))
    Fp1 = torch.matmul(Fundamental, p1)
    epipolar = (Fp1 * p2).sum(1, True)
    # epipolar /= (1e-7 + torch.square(Fp1[:, :2]).sum(1, True).sqrt())
    return epipolar


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    '''environment preparation '''
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    output_motions_dir = os.path.join(opt.eval_out_dir, "flow", opt.eval_name)
    create_dir(output_motions_dir)

    assert opt.eval_out_dir, 'Cannot find a folder at {}'.format(opt.eval_out_dir)

    '''calculation preparation '''
    img_height = 128  # opt.height #net_dict['height']
    img_width = 416  # opt.width #net_dict['width']
    scale_factor = get_scale_factor(1, img_height, img_width).to(device)
    flow_warp = FlowWarp(1, img_height, img_width).to(device)
    ones = torch.ones(1, 1, img_height, img_width, device=device)
    weight = gauss_distance_weight(1)[0].to(device)
    if opt.weights_init == 'pretrained':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        mean = (0.45, 0.45, 0.45)
        std = (0.225, 0.225, 0.225)

    print("-> Computing predictions with size {}x{}".format(img_width, img_height))
    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    '''dataset and models preparation '''
    dataset_dir = Path(opt.raw_dataset_dir)
    framework = test_framework(dataset_dir)
    files = None  # readlines(os.path.join(opt.root, "splits/eigen_zhou/train_files.txt"))
    # framework = test_framework(dataset_dir, random.sample(files, 200))
    motion_net, pose_net = load_models(opt, device, mobile=False)

    errors_full = []

    with torch.no_grad():
        for j, sample in enumerate(tqdm(framework)):
            intrinsics = np.copy(sample['intrinsics'])
            ref_img = sample['next_tgt']
            h, w, _ = ref_img.shape
            zoom_y = img_height / h
            zoom_x = img_width / w
            if h != img_height or w != img_width:
                intrinsics[0] *= zoom_x
                intrinsics[1] *= zoom_y
                ref_img = imresize(ref_img, (img_height, img_width)).astype(np.float32)

            K = np.vstack((np.hstack((intrinsics, [[0], [0], [0]])), [0, 0, 0, 1])).astype(np.float32)
            inv_K = np.linalg.inv(K)
            inv_K = torch.from_numpy(inv_K).unsqueeze(0).to(device)

            # NEXT PART
            input_color = torch.from_numpy(ref_img).permute(2, 0, 1) / 255
            normalize(input_color, mean, std)
            next_motion_input_color = input_color.unsqueeze(0).to(device)

            # MAIN PART
            tgt_img = sample['tgt']
            if h != img_height or w != img_width:
                tgt_img = imresize(tgt_img, (img_height, img_width)).astype(np.float32)
            input_color = torch.from_numpy(tgt_img).permute(2, 0, 1) / 255
            normalize(input_color, mean, std)
            motion_input_color = input_color.unsqueeze(0).to(device)

            # PREDICTIONS
            axisangle, translation = pose_net(motion_input_color, next_motion_input_color)
            cam_T_cam = transformation_from_parameters(axisangle, translation)
            flows, features_enc = motion_net(motion_input_color, next_motion_input_color)
            full_flow = flows[('flow', 0, 0)] * scale_factor

            pix_coords_flow, pix_coords_norm_flow, _ = flow_warp(full_flow)
            # flow_warp_img = F.grid_sample(next_motion_input_color,
            #                               pix_coords_norm_flow,
            #                               padding_mode=opt.padding,
            #                               align_corners=opt.align_corners)
            p1 = torch.cat([flow_warp.pix_coords, ones], 1).view(1, 3, -1)
            p2 = torch.cat([pix_coords_flow, ones], 1).view(1, 3, -1)
            epipolar = get_epipolar_new(
                p1, p2, inv_K[:, :3, :3],
                cam_T_cam[:, :3, :3], cam_T_cam[:, :3, -1]).view(1, 1, img_height, img_width).abs()
            post_epip = post_pro_epipolar_weighted(epipolar, weight)
            post_epip /= post_epip.max()
            # post_epip = post_process_epipolar_1(epipolar, d2_output)
            epipolar /= epipolar.max()

            # epipolar map used gt flow_occ
            gt_flow = sample['gt_flow_occ']
            gt_flow_zoomed = imresize(gt_flow[:, :, :2], (img_width, img_height))
            gt_flow_zoomed[..., 0] = gt_flow_zoomed[..., 0] * zoom_x
            gt_flow_zoomed[..., 1] = gt_flow_zoomed[..., 1] * zoom_y
            gt_flow_zoomed = torch.from_numpy(gt_flow_zoomed).permute(2, 0, 1).unsqueeze(0).to(device)
            pix_coords_gt_flow, _, _ = flow_warp(gt_flow_zoomed)
            gt_p1 = torch.cat([flow_warp.pix_coords, ones], 1).view(1, 3, -1)
            gt_p2 = torch.cat([pix_coords_gt_flow, ones], 1).view(1, 3, -1)
            M = torch.from_numpy(sample['gt_transformation']).unsqueeze(0).to(device)
            gt_epipolar = get_epipolar_new(
                gt_p1, gt_p2, inv_K[:, :3, :3], M[:, :3, :3], M[:, :3, -1]).view(1, 1, img_height, img_width).abs()
            gt_epipolar = gt_epipolar / gt_epipolar.max()

            # flow error maps
            full_flow = full_flow.cpu()[0, :].numpy().transpose(1, 2, 0)
            full_flow_zoomed = imresize(full_flow, (w, h))
            full_flow_zoomed[..., 0] = full_flow_zoomed[..., 0] / zoom_x
            full_flow_zoomed[..., 1] = full_flow_zoomed[..., 1] / zoom_y

            noc_mask_flow = sample['gt_flow_noc'][..., 2]
            flow_err_img = get_flow_error_image(gt_flow[..., :2],
                                                (0 < gt_flow[..., 2]),
                                                (0 < noc_mask_flow),
                                                full_flow_zoomed)
            if opt.pred_errors:
                error_all = compute_epe(gt_flow, full_flow_zoomed, gt_flow[..., 2])
                error_noc = compute_epe(gt_flow, full_flow_zoomed, noc_mask_flow)
                errors_full.append([error_all, error_noc])

            if opt.save_pred_motions:
                t = tgt_img
                f = flow_to_image(full_flow)
                fe = imresize(flow_err_img, (img_width, img_height))
                e = 255 * epipolar[0, ...].permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()
                ge = 255 * gt_epipolar[0, ...].permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()
                pe = 255 * post_epip[0, ...].permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()

                # save results
                name = get_output_img_name(files, j) if files else j
                viz = np.hstack([t, f, fe, e, ge, pe]).astype(np.uint8)
                file = os.path.join(output_motions_dir, "{}.png".format(name))
                imwrite(file, viz)

    if opt.pred_errors:
        mean_errors_full = np.array(errors_full).mean(0)

        print("\n  " + ("{:>8} | " * 2).format("epe_all", "epe_noc"))
        print(("&{: 8.3f}  " * 2).format(*mean_errors_full.tolist()) + "\\\\")
        print("\n-> Done!")

        result_txt = os.path.join(output_motions_dir, "result.txt")
        f = open(result_txt, 'w')
        write_result(f, mean_errors_full, ["epe_noc", "epe_occ"])
        f.close()

    if opt.save_pred_motions:
        print("Evaluation save to --> ", output_motions_dir)


if __name__ == '__main__':
    options = MonodepthOptions()
    evaluate(options.parse())
