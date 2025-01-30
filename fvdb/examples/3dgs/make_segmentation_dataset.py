# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.utils.data
import tqdm
import tyro
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from fvdb.nn.gaussian_splatting import GaussianSplat3D
from fvdb.utils.data import ColmapDataset, ColmapParser


@torch.inference_mode()
def make_segmentation_dataset(model: GaussianSplat3D, dataset: ColmapDataset, max_scale: float = 2.0, device="cuda"):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    sam2_checkpoint = "sam2_repo/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2_mask_generator = SAM2AutomaticMaskGenerator(
        sam2, points_per_side=32, pred_iou_thresh=0.90, stability_score_thresh=0.90
    )

    all_scales = []
    all_pixel_to_mask_ids = []
    all_mask_cdfs = []
    all_images = []
    all_cam_to_world = []
    all_intrinsics = []

    for data in tqdm.tqdm(dataloader):
        img = data["image"].to(device).squeeze() / 255.0
        K = data["K"].to(device).squeeze()
        cam_to_world = data["camtoworld"].to(device).squeeze()
        world_to_cam = torch.linalg.inv(cam_to_world).contiguous()

        # Compute a depth image for the current camera as well as the 3D points that correspond to each pixel
        depth, cam_pts = model.render_depth_points(
            image_w=img.shape[1],
            image_h=img.shape[0],
            extrinsics_mats=world_to_cam.unsqueeze(0),
            intrinsics_mats=K.unsqueeze(0),
            rasterize_mode="classic",
            sh_degree=0,
        )
        depth, cam_pts = depth.squeeze(), cam_pts.squeeze()  # [H, W], [H, W, 3]

        # Generate a set of masks for the current image using SAM2
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sam_masks = sam2_mask_generator.generate(img.cpu().numpy())
            sam_masks = sorted(sam_masks, key=(lambda x: x["area"]), reverse=True)
            sam_masks = torch.stack([torch.from_numpy(m["segmentation"]) for m in sam_masks]).to(device)  # [M, H, W]

        # Erode masks to remove noise at the boundary.
        # We're going to compute the scale of each mask by taking the standard deviation of the 3D points
        # within that mask, and the points at the boundary of masks are usually noisy.
        eroded_masks = (
            torch.conv2d(
                sam_masks.unsqueeze(1).float(),
                torch.full((3, 3), 1.0, device=device).view(1, 1, 3, 3),
                padding=1,
            )
            >= 5
        ).squeeze(
            1
        )  # [M, H, W]

        # Compute a 3D scale per mask which corresponds to the variance of the 3D points that fall within that mask
        # Filter out masks whose scale is too large since very scattered 3D points are likely noise
        scales = torch.stack([(cam_pts[mask].std(dim=0) * 2.0).norm() for mask in eroded_masks])  # [M]
        keep = scales < max_scale  # [M]
        eroded_masks = eroded_masks[keep]  # [M', H, W]
        scales = scales[keep]  # [M']

        # Compute a tensor that maps pixels to the set of masks which intersect that pixel (sorted by area)
        # i.e. pixel_to_mask_id[i, j] = [m1, m2, m3, ...] where m1, m2, ... are the integer ids of the masks
        # which contain pixel [i, j] and area(m1) <= area(m2) <= area(m3) <= ...
        max_masks = int(eroded_masks.sum(dim=0).max().item())
        pixel_to_mask_id = torch.full(
            (max_masks, eroded_masks.shape[1], eroded_masks.shape[2]), -1, dtype=torch.long, device=device
        )  # [MM, H, W]
        for m, mask in enumerate(eroded_masks):
            mask_clone = mask.clone()
            for i in range(max_masks):
                free = pixel_to_mask_id[i] == -1
                masked_area = mask_clone == 1
                right_index = free & masked_area
                if len(pixel_to_mask_id[i][right_index]) > 0:
                    pixel_to_mask_id[i][right_index] = m
                mask_clone[right_index] = 0
        pixel_to_mask_id = pixel_to_mask_id.permute(1, 2, 0)  # [H, W, MM]

        # We're going to use the SAM masks to group pixels for contrastive learning.
        # i.e. we're going to project features for each pixel into the image and push features corresponding to pixels
        #      with the same mask together, and pixels with different masks apart.
        # If we sample pixels, uniformly, we're going to overwhelmingly sample pixels in large masks, and small masks
        # will not get supervised. To fix this, we assign a weight to each mask which intersects a pixel. The weight
        # is proportional to the log probability of sampling that mask (under uniform sampling).
        # These weights are encoded as a CDF per-pixel which we use to choose which mask to use for loss computation
        # at training time

        # Get the unique ids of each mask, and the number of pixels each mask occupies (area)
        mask_ids, num_pix_per_mask = torch.unique(pixel_to_mask_id, return_counts=True)  # [N], [N]

        # Sort masks by their area
        mask_area_sort_ids = torch.argsort(num_pix_per_mask)
        mask_ids, num_pix_per_mask = mask_ids[mask_area_sort_ids], num_pix_per_mask[mask_area_sort_ids]  # [N], [N]
        num_pix_per_mask[0] = 0  # Remove the -1 mask which corresponds to no mask, [N]

        # The probability of any pixel landing in a mask is just the area of the mask over the area of the image
        probs = num_pix_per_mask / num_pix_per_mask.sum()  # [N]

        # Gather the probability values into pixel_to_mask_id, which produces a tensor where
        # each pixel has a list of probabilities that correspond to the masks that intersect that pixel
        mask_probs = torch.gather(probs, 0, pixel_to_mask_id.reshape(-1) + 1).view(pixel_to_mask_id.shape)  # [H, W, MM]

        # Compute a CDF for each pixel (which sums to 1) which weighs each mask by its log probability of being sampled
        # i.e. mask_cdf[i, j, k] is a cumulative probability weight used to select mask k for pixel [i, j]
        mask_cdf = torch.log(mask_probs)
        never_masked = mask_cdf.isinf()
        mask_cdf[never_masked] = 0.0
        mask_cdf = mask_cdf / (mask_cdf.sum(dim=-1, keepdim=True) + 1e-6)
        mask_cdf = torch.cumsum(mask_cdf, dim=-1)  # [H, W, MM]
        mask_cdf[never_masked] = 1.0

        all_scales.append(scales.cpu())
        all_pixel_to_mask_ids.append(pixel_to_mask_id.cpu())
        all_mask_cdfs.append(mask_cdf.cpu())
        all_images.append(img.cpu())
        all_cam_to_world.append(cam_to_world.cpu())
        all_intrinsics.append(K.cpu())

    return all_scales, all_pixel_to_mask_ids, all_mask_cdfs, all_images, all_cam_to_world, all_intrinsics


def main(checkpoint_path: str, data_path: str, data_scale_factor: int = 4, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GaussianSplat3D(torch.rand([8, 3]), torch.rand([8, 3])).to(device)
    model.load_state_dict(checkpoint["splats"])

    feats = torch.randn(model.num_gaussians, 128, device=device)

    print(model)

    parser = ColmapParser(data_path, test_every=1, normalize=True, factor=data_scale_factor)
    dataset = ColmapDataset(parser, split="test")

    scales, mask_ids, mask_cds, imgs, cam_to_worlds, intrinsics = make_segmentation_dataset(
        model, dataset, device=device
    )

    torch.save(
        {
            "scales": scales,
            "mask_ids": mask_ids,
            "mask_cdfs": mask_cds,
            "images": imgs,
            "cam_to_worlds": cam_to_worlds,
            "intrinsics": intrinsics,
        },
        "segmentation_dataset.pt",
    )


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
