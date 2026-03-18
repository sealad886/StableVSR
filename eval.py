import argparse
import os
import warnings

import numpy as np
import pyiqa
import torch
from DISTS_pytorch import DISTS
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large as raft
from torchvision.transforms import CenterCrop, ToTensor
from tqdm import tqdm

from util.flow_utils import get_flow

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation code for StableVSR.")
    # expected folder organization: root/sequences/frames
    parser.add_argument(
        "--out_path",
        type=str,
        default="./StableVSR_results/",
        help="Path to output folder containing the upscaled frames.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Path to folder with GT frames (required).",
    )
    args = parser.parse_args()

    if args.gt_path is None:
        parser.error("--gt_path is required. Provide the path to GT frames.")

    print("Run with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    gt_path = args.gt_path
    rec_path = args.out_path
    seqs = sorted(os.listdir(rec_path))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    of_model = raft(weights=Raft_Large_Weights.DEFAULT).to(device)
    lpips = LPIPS(normalize=True).to(device)
    dists = DISTS().to(device)
    psnr = PSNR(data_range=1).to(device)
    ssim = SSIM(data_range=1).to(device)
    musiq = pyiqa.create_metric("musiq", device=device, as_loss=False)
    niqe = pyiqa.create_metric("niqe", device=device, as_loss=False)
    clip = pyiqa.create_metric("clipiqa", device=device, as_loss=False)

    lpips_dict = {}
    psnr_dict = {}
    ssim_dict = {}
    dists_dict = {}
    musiq_dict = {}
    niqe_dict = {}
    clip_dict = {}
    tlpips_dict = {}
    tof_dict = {}
    tt = ToTensor()

    total = 0
    for root, dirs, files in os.walk(gt_path):
        total += len(files)

    pbar = tqdm(total=total, ncols=100)

    for seq in seqs:

        ims_rec = sorted(os.listdir(os.path.join(rec_path, seq)))
        gt_seq_dir = os.path.join(gt_path, seq)
        if not os.path.isdir(gt_seq_dir):
            print(f"Warning: GT directory not found for '{seq}', skipping.")
            continue
        ims_gt = sorted(os.listdir(gt_seq_dir))

        lpips_dict[seq] = []
        psnr_dict[seq] = []
        ssim_dict[seq] = []
        dists_dict[seq] = []
        musiq_dict[seq] = []
        niqe_dict[seq] = []
        clip_dict[seq] = []
        tlpips_dict[seq] = []
        tof_dict[seq] = []

        for i, (im_rec, im_gt) in enumerate(zip(ims_rec, ims_gt)):
            with torch.no_grad():
                gt = Image.open(os.path.join(gt_path, seq, im_gt))
                rec = Image.open(os.path.join(rec_path, seq, im_rec))
                gt = tt(gt).unsqueeze(0).to(device)
                rec = tt(rec).unsqueeze(0).to(device)

                psnr_value = psnr(gt, rec)
                ssim_value = ssim(gt, rec)
                lpips_value = lpips(gt, rec)
                dists_value = dists(gt, rec)
                musiq_value = musiq(rec)
                niqe_value = niqe(rec)
                clip_value = clip(rec)
                if i > 0:
                    tlpips_value = (lpips(gt, prev_gt) - lpips(rec, prev_rec)).abs()
                    tlpips_dict[seq].append(tlpips_value.item())
                    tof_value = (
                        (
                            get_flow(of_model, rec, prev_rec)
                            - get_flow(of_model, gt, prev_gt)
                        )
                        .abs()
                        .mean()
                    )
                    tof_dict[seq].append(tof_value.item())

            psnr_dict[seq].append(psnr_value.item())
            ssim_dict[seq].append(ssim_value.item())
            lpips_dict[seq].append(lpips_value.item())
            dists_dict[seq].append(dists_value.item())
            musiq_dict[seq].append(musiq_value.item())
            niqe_dict[seq].append(niqe_value.item())
            clip_dict[seq].append(clip_value.item())

            prev_rec = rec
            prev_gt = gt
            pbar.update()

    pbar.close()
    mean_lpips = np.round(
        np.mean([np.mean(lpips_dict[key]) for key in lpips_dict.keys()]), 3
    )
    mean_dists = np.round(
        np.mean([np.mean(dists_dict[key]) for key in dists_dict.keys()]), 3
    )
    mean_psnr = np.round(
        np.mean([np.mean(psnr_dict[key]) for key in psnr_dict.keys()]), 2
    )
    mean_ssim = np.round(
        np.mean([np.mean(ssim_dict[key]) for key in ssim_dict.keys()]), 3
    )
    mean_musiq = np.round(
        np.mean([np.mean(musiq_dict[key]) for key in musiq_dict.keys()]), 2
    )
    mean_niqe = np.round(
        np.mean([np.mean(niqe_dict[key]) for key in niqe_dict.keys()]), 2
    )
    mean_clip = np.round(
        np.mean([np.mean(clip_dict[key]) for key in clip_dict.keys()]), 3
    )
    mean_tlpips = (
        np.round(np.mean([np.mean(v) for v in tlpips_dict.values() if v]) * 1e3, 2)
        if any(tlpips_dict.values())
        else 0.0
    )
    mean_tof = (
        np.round(np.mean([np.mean(v) for v in tof_dict.values() if v]) * 1e1, 3)
        if any(tof_dict.values())
        else 0.0
    )

    print(
        f"PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {mean_lpips}, DISTS: {mean_dists}, MUSIQ: {mean_musiq}, CLIP: {mean_clip}, NIQE: {mean_niqe}, tLPIPS: {mean_tlpips}, tOF: {mean_tof}"
    )
