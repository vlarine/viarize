import sys

sys.path.append('./pytorch3d-lite')
sys.path.append('./MiDaS')
sys.path.append('./AdaBins')

import os
import torch, torchvision
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import math
import argparse
from tqdm import tqdm
from glob import glob

from infer import InferenceHelper
import py3d_tools as p3d
import utils as midas_utils

DEVICE_NAME = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_NAME)
MAX_ADABINS_AREA = 500000
MIN_ADABINS_AREA = 448*448
TRANSLATION_SCALE = 1.0/200.0

# 1.4 Define Midas functions

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

midas_models = {
    'midas_v21_small': 'pretrained/midas_v21_small-70d6b9c8.pt',
    'midas_v21': 'pretrained/midas_v21-f6b98070.pt',
    'dpt_large': 'pretrained/dpt_large-midas-2f21e586.pt',
    'dpt_hybrid': '{model_path}/dpt_hybrid-midas-501f0c75.pt',
    'dpt_hybrid_nyu': '{model_path}/dpt_hybrid_nyu-2ce69ec7.pt',
}


def init_midas_depth_model(midas_model_type='dpt_large', optimize=True):
    midas_model = None
    net_w = None
    net_h = None
    resize_mode = None
    normalization = None

    print(f"Initializing MiDaS '{midas_model_type}' depth model...")
    # load network
    midas_model_path = midas_models[midas_model_type]

    if midas_model_type == 'dpt_large': # DPT-Large
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone='vitl16_384',
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = 'minimal'
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == 'dpt_hybrid': #DPT-Hybrid
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone='vitb_rn50_384',
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode='minimal'
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == 'dpt_hybrid_nyu': #DPT-Hybrid-NYU
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone='vitb_rn50_384',
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode='minimal'
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == 'midas_v21':
        midas_model = MidasNet(midas_model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode='upper_bound'
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif midas_model_type == 'midas_v21_small':
        midas_model = MidasNet_small(midas_model_path, features=64, backbone='efficientnet_lite3', exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode='upper_bound'
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"midas_model_type '{midas_model_type}' not implemented")
        assert False

    midas_transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    midas_model.eval()

    if optimize==True:
        if DEVICE == torch.device('cuda'):
            midas_model = midas_model.to(memory_format=torch.channels_last)
            midas_model = midas_model.half()

    midas_model.to(DEVICE)

    print(f"MiDaS '{midas_model_type}' depth model initialized.")
    return midas_model, midas_transform, net_w, net_h, resize_mode, normalization


@torch.no_grad()
def transform_image_3d(img_filepath, midas_model, midas_transform, infer_helper, device, rot_mat=torch.eye(3).unsqueeze(0), translate=(0.,0.,-0.04), near=2000, far=20000, fov_deg=60, padding_mode='border', sampling_mode='bicubic', midas_weight = 0.3,spherical=False):
    img_pil = Image.open(open(img_filepath, 'rb')).convert('RGB')
    w, h = img_pil.size
    image_tensor = torchvision.transforms.functional.to_tensor(img_pil).to(device)

    use_adabins = midas_weight < 1.0

    if use_adabins:
        # AdaBins
        """
        predictions using nyu dataset
        """

        image_pil_area = w*h
        if image_pil_area > MAX_ADABINS_AREA:
            scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS) # LANCZOS is supposed to be good for downsampling.
        elif image_pil_area < MIN_ADABINS_AREA:
            scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
        else:
            depth_input = img_pil
        try:
            _, adabins_depth = infer_helper.predict_pil(depth_input)
            if image_pil_area != MAX_ADABINS_AREA:
                adabins_depth = torchvision.transforms.functional.resize(torch.from_numpy(adabins_depth), image_tensor.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC).squeeze().to(device)
            else:
                adabins_depth = torch.from_numpy(adabins_depth).squeeze().to(device)
            adabins_depth_np = adabins_depth.cpu().numpy()
        except:
            pass

    # MiDaS
    img_midas = midas_utils.read_image(img_filepath)
    img_midas_input = midas_transform({"image": img_midas})["image"]
    midas_optimize = True

    # MiDaS depth estimation implementation
    print("Running MiDaS depth estimation implementation...")
    sample = torch.from_numpy(img_midas_input).float().to(device).unsqueeze(0)
    if midas_optimize==True and device == torch.device("cuda"):
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()
    prediction_torch = midas_model.forward(sample)
    prediction_torch = torch.nn.functional.interpolate(
            prediction_torch.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    prediction_np = prediction_torch.clone().cpu().numpy()

    # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
    prediction_np = np.subtract(50.0, prediction_np)
    prediction_np = prediction_np / 19.0

    if use_adabins:
        adabins_weight = 1.0 - midas_weight
        depth_map = prediction_np*midas_weight + adabins_depth_np*adabins_weight
    else:
        depth_map = prediction_np

    depth_map = np.expand_dims(depth_map, axis=0)
    depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)

    pixel_aspect = 1.0 # really.. the aspect of an individual pixel! (so usually 1.0)
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    # Transform the points using pytorch3d. With current functionality, this is overkill and prevents it from working on Windows.
    # If you want it to run on Windows (without pytorch3d), then the transforms (and/or perspective if that's separate) can be done pretty easily without it.
    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    if spherical:
        spherical_grid = get_spherical_projection(h, w, torch.tensor([0,0], device=device), -0.4,device=device)#align_corners=False
        stage_image = torch.nn.functional.grid_sample(image_tensor.add(1/512 - 0.0001).unsqueeze(0), offset_coords_2d, mode=sampling_mode, padding_mode=padding_mode, align_corners=True)
        new_image = torch.nn.functional.grid_sample(stage_image, spherical_grid,align_corners=True) #, mode=sampling_mode, padding_mode=padding_mode, align_corners=False)
    else:
        new_image = torch.nn.functional.grid_sample(image_tensor.add(1/512 - 0.0001).unsqueeze(0), offset_coords_2d, mode=sampling_mode, padding_mode=padding_mode, align_corners=False)

    img_pil = torchvision.transforms.ToPILImage()(new_image.squeeze().clamp(0,1.))
    return img_pil


def get_spherical_projection(H, W, center, magnitude,device):
    xx, yy = torch.linspace(-1, 1, W,dtype=torch.float32,device=device), torch.linspace(-1, 1, H,dtype=torch.float32,device=device)
    gridy, gridx  = torch.meshgrid(yy, xx)
    grid = torch.stack([gridx, gridy], dim=-1)
    d = center - grid
    d_sum = torch.sqrt((d**2).sum(axis=-1))
    grid += d * d_sum.unsqueeze(-1) * magnitude
    return grid.unsqueeze(0)


def generate_eye_views(args, in_folder, filename, out_folder, midas_model, midas_transform, infer_helper):
    for i in range(2):
        theta = args.vr_eye_angle * (math.pi/180)
        ray_origin = math.cos(theta) * args.vr_ipd / 2 * (-1.0 if i==0 else 1.0)
        ray_rotation = (theta if i==0 else -theta)
        translate_xyz = [-(ray_origin)*TRANSLATION_SCALE, 0,0]
        rotate_xyz = [0, (ray_rotation), 0]
        rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=DEVICE), 'XYZ').unsqueeze(0)
        transformed_image = transform_image_3d(f'{in_folder}/{filename}', midas_model, midas_transform, infer_helper, DEVICE,
                                                      rot_mat, translate_xyz, args.near_plane, args.far_plane,
                                                      args.fov, padding_mode=args.padding_mode,
                                                      sampling_mode=args.sampling_mode, midas_weight=args.midas_weight,spherical=True)
        name = '.'.join(filename.split('.')[:-1])
        ext = filename.split('.')[-1]
        eye_file_path = f'{out_folder}/{name}' + ('_l' if i==0 else '_r') + '.' + ext
        transformed_image.save(eye_file_path)


def main(args):
    if not os.path.exists(args.out):
        print('There is no output folder: ', args.out)
        return

    if os.path.exists(args.input):
        midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
        infer_helper = InferenceHelper(dataset='nyu', device=DEVICE_NAME)
        if os.path.isdir(args.input):
            for ext in ('*.jpg', '*.png'):
                for filename in tqdm(glob('{}/{}'.format(args.input, ext))):
                    folder = os.path.dirname(filename)
                    filename = os.path.basename(filename)
                    generate_eye_views(args, folder, filename, args.out, midas_model, midas_transform, infer_helper)

        else:
            folder = os.path.dirname(args.input)
            if folder == '':
                folder = '.'
            filename = os.path.basename(args.input)
            generate_eye_views(args, folder, filename, args.out, midas_model, midas_transform, infer_helper)
    else:
        print('There is no file nor folder: ', args.input)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--midas-depth-model', type=str, default='dpt_large',
                        help='Midas model name')
    parser.add_argument('--midas-weight', type=float, default=0.5,
                        help='Midas/AdaBins rate')
    parser.add_argument('--vr-eye-angle', type=float, default=0.5,
                        help='')
    parser.add_argument('--vr-ipd', type=float, default=5.0,
                        help='')
    parser.add_argument('--near-plane', type=int, default=200,
                        help='')
    parser.add_argument('--far-plane', type=int, default=10000,
                        help='')
    parser.add_argument('--fov', type=int, default=40,
                        help='')
    parser.add_argument('--padding-mode', type=str, default='border',
                        help='')
    parser.add_argument('--sampling-mode', type=str, default='bicubic',
                        help='')
    parser.add_argument('--input', type=str, default='frames',
                        help='Input file or folder')
    parser.add_argument('--out', type=str, default='out',
                        help='Output folder')

    args = parser.parse_args()
    main(args)
