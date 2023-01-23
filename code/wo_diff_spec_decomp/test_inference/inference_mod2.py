from model import *
from util import *
from metric import *
from preprocessing import *
import numpy as np
import torch
import torch.nn.functional as F
import pyexr
import argparse

import glob
import time

parser = argparse.ArgumentParser()
parser.add_argument("--inDir", "-i", required=True)
parser.add_argument("--outDir", "-o", required=True)
parser.add_argument("--fileName", required=True)
parser.add_argument("--overlap", type=int, default=5)
parser.add_argument("--isLoadGt", dest="loadGt", action="store_true")
parser.set_defaults(loadGt=False)
# model parameters (fixed)
parser.add_argument("--blockSize", type=int, default=8)
parser.add_argument("--haloSize", type=int, default=3)
parser.add_argument("--numHeads", type=int, default=4)
parser.add_argument("--numSA", type=int, default=5)
parser.add_argument("--inCh", type=int, default=3)
parser.add_argument("--auxInCh", type=int, default=7)
parser.add_argument("--baseCh", type=int, default=256)
parser.add_argument("--numGradientCheckpoint", type=int, default=0)  # how many Trans blocks with gradient checkpoint
args, unknown = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
permutation = [0, 3, 1, 2]


def inference(args, save_path, sample):
    print("-AFGSA %s" % args.fileName)
    G = AFGSANet(args.inCh, args.auxInCh, args.baseCh, num_sa=args.numSA, block_size=args.blockSize,
                 halo_size=args.haloSize, num_heads=args.numHeads, num_gcp=args.numGradientCheckpoint).to(device)
    G.load_state_dict(torch.load(r'..\..\..\models\wo_diff_spec_decomp\G_4spp.pt'))
    with torch.no_grad():
        G.eval()

        print("\tPreparing data")
        # noisy_exr = pyexr.open(args.fileName.replace("spp.exr", "spp_f.exr"))        
        noisy_exr = pyexr.open(args.fileName)
        w = noisy_exr.width / 1000
        h = noisy_exr.height / 1000
        noisy_exr = noisy_exr.get_all()
        noisy_exr = noisy_exr['default']
        

        normal = pyexr.open(args.fileName.replace(sample, "normals"))
        normal = normal.get_all()['default']
        normal = preprocess_normal(np.nan_to_num(normal))

        depth = pyexr.open(args.fileName.replace(sample, "motiond"))
        depth = depth.get_all()['default']
        depth_layer = depth[:,:,2]
        depth = np.expand_dims(depth_layer, axis=2)
        # depth = np.concatenate((depth_layer,depth_layer,depth_layer), axis=2)
        depth = preprocess_depth(np.nan_to_num(depth))

        albedo = pyexr.open(args.fileName.replace(sample, "albedo"))
        albedo = albedo.get_all()['default']
        

        n = 3 # number of parts image is split to
        overlap = 12

        out = []

        _, cols, _ = noisy_exr.shape
        col_split = cols // n

        lim1 = noisy_exr.shape[1] // 4
        lim2 = noisy_exr.shape[1] // 2
        lim3 = noisy_exr.shape[1] // 4 * 3

        start = time.time()

        for i in range(n):
            if i == 0:
                noisy = noisy_exr[:, :lim2, :]
                normal_tmp = normal[:, :lim2, :]
                depth_tmp = depth[:, :lim2, :]
                albedo_tmp = albedo[:, :lim2, :]
            elif i == 1:
                noisy = noisy_exr[:, lim1:lim3, :]
                normal_tmp = normal[:, lim1:lim3, :]
                depth_tmp = depth[:, lim1:lim3, :]
                albedo_tmp = albedo[:, lim1:lim3, :]
            else:
                noisy = noisy_exr[:, lim2:, :]
                normal_tmp = normal[:, lim2:, :]
                depth_tmp = depth[:, lim2:, :]
                albedo_tmp = albedo[:, lim2:, :]

            # albedo = noisy_exr['albedo']

            aux_features = np.concatenate((normal_tmp.copy(), depth_tmp.copy(), albedo_tmp.copy()), axis=2)[np.newaxis, :, :, :]
            aux_features_pre = torch.FloatTensor(aux_features).permute(permutation).to(device)
            aux_features_pre = F.pad(aux_features_pre, (0, aux_features_pre.shape[3] % args.blockSize,
                                                        0, aux_features_pre.shape[2] % args.blockSize), "constant", 0)

            # noisy = noisy_exr['default']
            noisy = noisy[:,:,0:3]
            # pyexr.write(os.path.join(save_path, 'noisy.exr'), noisy)
            noisy = np.nan_to_num(noisy)
            noisy = np.clip(noisy, 0, np.max(noisy))[np.newaxis, :, :, :]
            noisy_pre = preprocess_specular(noisy)
            noisy_pre = torch.FloatTensor(noisy_pre).permute(permutation).to(device)
            noisy_pre = F.pad(noisy_pre, (0, noisy_pre.shape[3] % args.blockSize,
                                        0, noisy_pre.shape[2] % args.blockSize), "constant", 0)

            center = (aux_features_pre.shape[3] + aux_features_pre.shape[3] % args.blockSize) // 2
            split1 = center + args.overlap * args.blockSize
            split2 = center - args.overlap * args.blockSize

            # split into 2 along W dimension to avoid cuda out of memory
            aux_features_pre = (aux_features_pre[:, :, :, :split1], aux_features_pre[:, :, :, split2:])
            noisy_pre = (noisy_pre[:, :, :, :split1], noisy_pre[:, :, :, split2:])

            print("\tStart denoising")
            output = []
            for n, a in zip(noisy_pre, aux_features_pre):
                o = G(n, a)
                output.append(o)
            # combine the 2 imgs to form the final output
            output = np.concatenate((output[0].cpu().numpy()[:, :, :, :center],
                                    output[1].cpu().numpy()[:, :, :, args.overlap * args.blockSize:]), axis=3)
            # post-processing (our model takes img after preprocess_specular as input, thus use postprocess_specular for output)
            output_c_n = np.transpose(postprocess_specular(output), (0, 2, 3, 1))[0]

            out.append(output_c_n)

            # transfer to image
            noisy_c_n_255 = tensor2img(np.transpose(noisy, permutation))[0]
            output_c_n_255 = tensor2img(output, post_spec=True)[0]

        merged = np.empty_like(noisy_exr)
        merged[:, :lim2, :] = out[0]
        merged[:, lim2:, :] = out[2]

        alpha1 = np.linspace(0, 1, lim1)[np.newaxis, :, np.newaxis]
        alpha2 = np.linspace(1, 0, lim1)[np.newaxis, :, np.newaxis] 
        alpha = np.concatenate((alpha1, alpha2),axis=1)
        # alpha = np.expand_dims(alpha, axis=2)
        alpha = np.concatenate((alpha, alpha, alpha),axis=2)
        alpha = np.repeat(alpha, 1080, axis=0)
        merged[:, lim1:lim3, :] = (1-alpha) * merged[:, lim1:lim3, :] + alpha * out[1]

        final_output = merged
        end = time.time()
        print(end-start)

        # save image
        # save output in exr
        pyexr.write(save_path, final_output)
        print("Saved: " + save_path)
        # pyexr.write(os.path.join(save_path, 'output_ours.exr'), output_c_n)
        # save noisy in png
        # save_img(os.path.join(save_path, 'noisy.png'), noisy_c_n_255, figsize=(w, h), dpi=1000)
        # save output in png
        # save_img(os.path.join(save_path, 'output_ours.png'), output_c_n_255, figsize=(w, h), dpi=1000)

        # print("\tDone!")

        if args.loadGt:
            gt_exr = pyexr.open(os.path.join(args.inDir, args.fileName.split('_')[0] + '_gt.exr')).get_all()
            gt = gt_exr['default']
            pyexr.write(os.path.join(save_path, 'gt.exr'), gt)
            gt = np.nan_to_num(gt)
            gt = np.clip(gt, 0, np.max(gt))
            gt_c_n_255 = tensor2img(np.transpose(gt, (2, 0, 1)))
            # save gt in png
            save_img(os.path.join(save_path, 'gt.png'), gt_c_n_255, figsize=(w, h), dpi=1000)

            # rmse: output after post-processing, without tone mapping and * 255 (use postprocess_specular/ postprocess_diffuse)
            # psnr, ssim: output after post-processing, tone mapping and * 255 (use tensor2img)
            rmse = calculate_rmse(output_c_n.copy(), gt.copy())
            psnr = calculate_psnr(output_c_n_255.copy(), gt_c_n_255.copy())
            ssim = calculate_ssim(output_c_n_255.copy(), gt_c_n_255.copy())

            print("\tRMSE: %f \tPSNR: %f \t1-SSIM: %f" % (rmse, psnr, 1-ssim))
            # save results
            with open(os.path.join(save_path, "evaluation_ours.txt"), 'w') as f:
                f.write("RMSE: %f \tPSNR: %f \t1-SSIM: %f" % (rmse, psnr, 1-ssim))


if __name__ == "__main__":
    
    root_dir = "Z:\\work\\exrs\\inputs"
    # samples = ["2spp", "4spp", "8spp"]
    samples = ["4spp"]
    
    out_dir_name = str(args.fileName).replace("*.", "")
    args.outDir = os.path.join(args.outDir,"AFGSA_" + out_dir_name)

    # create_folder(args.outDir)
    save_path = os.path.join(args.outDir, '_'.join(["AFGSA", args.fileName]))
    # create_folder(save_path)

    files = glob.glob(os.path.join(args.inDir, args.fileName))


    for item in os.listdir(root_dir):

        if item == 'bistro1-day':
            continue

        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            scene_directory = item_path

            for sample in samples:
                pattern = "*." + sample + ".exr"

                output_directory = os.path.join(scene_directory.replace("inputs", "outputs"), "AFGSA4spp_" + sample)
                create_folder(output_directory)

                files = glob.glob(os.path.join(scene_directory, pattern))

                for file in files:
                    args.fileName = file
                    args.outDir = output_directory
                    save_path = os.path.join(output_directory, os.path.basename(file))

                    inference(args, save_path, sample)


