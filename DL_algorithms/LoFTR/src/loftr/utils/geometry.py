import torch


@torch.no_grad()
#Keypoint warping function in PyTorch that maps 2D keypoints from one image (I0) to another image (I1) using depth maps, 
# intrinsic matrices (K0,K1), and the camera pose transformation (T_0to1)
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]  which keypoints are validly warped
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>    warped keypoints in image I1
    """
    #Convert them to integer pixel coordinates for indexing into the depth map
    kpts0_long = kpts0.round().long()   #shape: [N,L,2], with continuous coordinates (x,y)

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject; Make homogeneous coordinates: [x,y,1] * depth; Undo camera intrinsics: x_cam=K0-¹ * (u,v,1) T * depth
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    #Now keypoints are in camera 0's 3D space
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    #Apply rotation and translation to map points from camera 0 -> camera 1; Extract new Z-coordinate as the reprojeted depth
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project; Multiply with camera intrinsics K1
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    #Normalize by Z to get image coordinates (x,y)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check; Ensure warped points land inside image I1 bounds
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    #Sample the actual depth map ofI1 at the projected location
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    #If relative error < 20%, depth is consistent; Helps reject occluded or mismatched points
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    #A keypoint is valid if: Has depth in I0; Projects inside bounds of I1; Depth agrees between reprojection and depth map
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0