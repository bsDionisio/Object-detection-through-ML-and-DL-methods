import torch, pydegensac, cv2, typing

from torch_dimcheck import dimchecked
from torchtyping import TensorType

from disk import EstimationFailedError
from disk.geom import Pose

@dimchecked
#Input: E=Essential matrix (computed from correspondences+intrinsics); i_coords:Keypoints in image 1; j_coords:Keypoints in image 2
def _recover_pose(E: TensorType[3, 3], i_coords: TensorType['N', 2], j_coords: TensorType['N', 2]):
    #All tensors are cast to float64 and converted to numpy arrays
    E_ = E.to(torch.float64).numpy()
    i_coords_ = i_coords.to(torch.float64).numpy()
    j_coords_ = j_coords.to(torch.float64).numpy()

    #recoverPose decomposes the essential matrix into rotation (R) and translation (T) between the two camera views:
    #It also returns: n_inliers=Number of inlier correspondences consistent with the pose; 
    # Inliers_mask=Boolean mask marking which correspondences are inliers
    n_inliers, R, T, inlier_mask = cv2.recoverPose(
        E_,
        i_coords_,
        j_coords_,
    )

    #R is returned as a 3x3 matrix->converted to a torch.float32 tensor
    R = torch.from_numpy(R).to(torch.float32)
    # T is returned as a 3x1 array->converted ans reshaped (squeeze(1)) into a 3-element vector
    T = torch.from_numpy(T).to(torch.float32).squeeze(1)

    #The function returns a Pose object
    return Pose(R, T)

@dimchecked
#Input: coords=an Nx2 tensor (pixel coordinates of keypoints in the image); K=a 3x3 ameta intrinsics matrix
#Output: another Nx2 tensor: the normalized coordinates
def _normalize_coords(coords: TensorType['N', 2], K: TensorType[3, 3]) -> TensorType['N', 2]:
    #Ensures everything is in float32 for consistency and numerical stability
    coords = coords.to(torch.float32)

    #Focal lengths in pixels
    f = torch.tensor([[K[0, 0], K[1, 1]]])
    #Principal point (optimal center)
    c = torch.tensor([[K[0, 2], K[1, 2]]])

    #Each pixel coordinate is: Shifted so that the image center (c) becomes the origin; 
    # Scaled by focal length so that values are in camera-normalized coordinates
    return (coords - c) / f

class Ransac(typing.NamedTuple):
    reprojection_threshold: float = 1.  #pixel error tolerance for considering a correspondence an inlier
    confidence            : float = 0.9999  #probability that the estimated model is correct (classic RANSAC setting)
    max_iters             : int   = 10_000  #max number of RANSAC iterations
    candidate_threshold   : int   = 10  #minimum nimber of matches required before trying estimation

    @dimchecked
    #This makes the class callable, so an instance of Ransac can be used like a function
    def __call__(
        self,
        left: TensorType['N', 2],   #matched keypoints (pixel coordinates, Nx2)
        right: TensorType['N', 2], 
        K1: TensorType[3, 3],   #Intrinsic matrices of the two cameras
        K2: TensorType[3, 3]
    ):
        #Moving everything to CPU; pydegensac (the library used for RANSAC fundamental matrix estimation) expects NumPy arrays on CPU
        left  = left.cpu()
        right = right.cpu()
        K1 = K1.cpu()
        K2 = K2.cpu()

        #If there are too few correspondences, estimation is impossible -> fail early
        if left.shape[0] < self.candidate_threshold:
            raise EstimationFailedError()

        #Uses Pydegensac (a Python wrapper around Degensac, a fast RANSAC-based fundamental matrix estimator); 
        # Returns: F=Fundamental matrix, mask=Boolean array of inliers chosen by RANSAC
        F, mask = pydegensac.findFundamentalMatrix(
            left.numpy(),
            right.numpy(),
            px_th=self.reprojection_threshold,
            conf=self.confidence,
            max_iters=self.max_iters
        )

        #If Degensac fails completely, abort
        if mask is None:
            raise EstimationFailedError()

        #Bring results back into PyTorch
        mask = torch.from_numpy(mask)
        F    = torch.from_numpy(F).to(torch.float32)

        #Convert the Fundamental matrix (F) into the Essential matrix (E) using camera intrinsics
        E = K2.T @ F @ K1

        try:
            #_recover_pose: uses OpenCV (cv2.recoverPose) to extract rotation and translation from E
            pose = _recover_pose(
                E,
                _normalize_coords( left[mask], K1), #converts pixel correspondences into normalized coordinates
                _normalize_coords(right[mask], K2),
            )
        except cv2.error:
            raise EstimationFailedError()

        #Returns: pose=estimated camera motion (R,T); mask=inlier mask (which matches were consistent with the model)
        return pose, mask