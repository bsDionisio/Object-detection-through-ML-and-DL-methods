import torch, typing
import numpy as np
import multiprocessing as mp
import multiprocessing.dummy as mpd
from typing import Dict

from disk import MatchedPairs, Image, NpArray, EstimationFailedError
from disk.loss.ransac import Ransac
from disk.geom import Pose, PoseError

CPU = torch.device('cpu')

#Used to represent the quality of a pose estimation
class PoseQualityResult:
    #Input: error=An object describing the error of the estimated pose; n_inliers=The number of inlier correspondences used in the pose estimation; 
    # success= Whether the pose estimation was successful. Defaults to True
    def __init__(self, error: PoseError, n_inliers: int, success: bool = True):
        self.error     = error
        self.n_inliers = n_inliers
        self.success   = success

    #Merges the dictionary from self.error.to_dict() with its own fields
    def to_dict(self):
        return {
            **self.error.to_dict(),
            'n_inliers': self.n_inliers,
            'success'  : int(self.success), #Converts success into an integer (1=True, 0=False)
        }

    #Provides a human-readable string version of the object
    def __str__(self):
        return (f'<PoseQualityResult error={self.error}, '
                f'n_inliers={self.n_inliers}, success={self.success}>')

# the error returned when pose estimation fails; Instead of returning None or throwing an exception, 
# the system can return this standardized object
FAILED_RESULT = PoseQualityResult(
    #Large error values, representing "bad" alignment
    error=PoseError(
        # less than 90 to avoid creating an extra bin in histograms
        Δ_θ=89.95,
        Δ_T=179.95,
    ),
    #No valid correspondences founf
    n_inliers=0,
    #Explicitly marks failure
    success=False
)

class Job(typing.NamedTuple):
    matches: MatchedPairs   #Object containing features correspondences between two images (kps1, kps2, and index pairs in matches)
    K1     : torch.Tensor   #Camera intrinsics (calibration matrices) for image1 and image2
    K2     : torch.Tensor
    pose1  : Pose   #Ground truth poses of the two cameras
    pose2  : Pose
    ransac : Ransac #A RANSAC algorithm implementation used to robustly estimate the relative pose

    #Makes the object callable like a function
    def __call__(self):
        m = self.matches

        #It extracts the matched keypoints
        left  = m.kps1[m.matches[0]]    #Keypoints from image 1
        right = m.kps2[m.matches[1]]    #Corresponding keypoints from image 2

        #Runs RANSAC with the matched keypoints and the camera intrinsics
        try:
            pose_estimate, mask = self.ransac(left, right, self.K1, self.K2)
        #If EANSAC fails to estimate a valid pose, it catches the exception and returns the predefined FAILED_RESULT
        except EstimationFailedError:
            return FAILED_RESULT

        #Computes the ground truth relative pose between the two cameras
        gt_pose = Pose.relative(self.pose1, self.pose2, normed=True)
        #Compares it to the estimated pose to compute an error message
        error   = Pose.error(gt_pose, pose_estimate)

        #The RANSAC mask marks which matches are inliers (consistent with the estmated pose); 
        # Converts mask to int64, sums up, and gets a plain Python integer count
        n_inliers = mask.to(torch.int64).sum().item()

        #Wraps everything up in a PoseQualityResult object
        return PoseQualityResult(
            error=error,
            n_inliers=n_inliers,
            success=True,
        )

    #Disables the default NamedTuple pretty-printing; Uses the base object representation
    __repr__ = object.__repr__

    @staticmethod
    #A helper to: Call a Job; Converts the result to a dictionary;
    #Makes it easier to use in parallel execution frameworks, since function need to be serializable
    def execute(job):
        return job().to_dict()

class PoseQuality:
    def __init__(self, ransac=Ransac(), dummy_pool=False, n_proc=6):
        self.ransac = ransac    #ransac=Default RANSAC estimator to use when evaluating pose quality
        self.pool = None    #Will hold the multiprocessing pool once created (starts as None)
        self.dummy_pool = dummy_pool    #A switch between two pool implementations
        self.n_proc = n_proc    #Number of worker processes to spawn in the pool (default=6)
    
    #When used with a with statement, this method is called
    def __enter__(self):
        #Depending on the dummy_pool, it either:
        if self.dummy_pool:
            #Creates a dummy pool
            self.pool = mpd.Pool(processes=self.n_proc)
        else:
            #Or a real multiprocessing pool
            self.pool = mp.Pool(processes=self.n_proc)
        
        #Return self so that the pool can be used inside the with block
        return self
    
    #Called automatically when exiting the with block
    def __exit__(self, *args):
        #Closes the pool (releases worker processes)
        self.pool.close()
        #Resets self.pool to None to avoid accidental reuse
        self.pool = None

    #Makes PoseQuality callable like a function
    def __call__(
        self,
        images: NpArray[Image], #a 2D numpy array of Image objects -> shape (N_scenes, N_per_scene); Each scene has multiply images
        #2D numpy array of MatchedPairs -> shape (N_scenes, N_pairs); 
        decisions: NpArray[MatchedPairs]    #These are feature matches between iamge pairs
    )-> NpArray[Dict[str, float]]:  #A numpy array of dictionaries, each containing numeric pose evaluation metrics
        
        #Ensures the mehtod is only called inside a context manager; Otherwise, self.pool wouldn't exist
        if self.pool is None:
            raise RuntimeError('self.pool is not initialized. PoseQuality needs to be used inside a `with` block.')
        
        #Number of scenes;  Number of images per scene
        N_scenes, N_per_scene = images.shape

        #Checks: Each scene has a set of decisions
        assert decisions.shape[0] == N_scenes
        #The number of decisions matches the number of unique images pairs in that scene
        assert decisions.shape[1] == ((N_per_scene - 1) * N_per_scene) // 2

        #Creates a numpy array to hold Job objects; Same shape as decisions
        jobs = np.zeros(decisions.shape, dtype=object)

        for i_scene in range(N_scenes):
            i_decision = 0
            #Grabs the corresponding MatchedPairs object
            scene_decisions = decisions[i_scene]
            scene_images    = images[i_scene]

            #Iterates over all unique image pairs (i_image1, i_image2) inside each scene
            for i_image1 in range(N_per_scene):
                image1 = scene_images[i_image1]
                #Extracts ground-truth pose
                pose1  = Pose.from_poselike(image1).to(CPU)
                #Extracts intrinsic
                K1     = image1.K.cpu()

                for i_image2 in range(i_image1+1, N_per_scene):
                    image2 = scene_images[i_image2]
                    #Extracts ground-truth pose
                    pose2  = Pose.from_poselike(image2).to(CPU)
                    #Extracts intrinsic
                    K2     = image2.K.cpu()

                    #Creates a job object with all of these information
                    jobs[i_scene, i_decision] = Job(
                        scene_decisions[i_decision].to(CPU),
                        K1, K2,
                        pose1, pose2,
                        self.ransac
                    )

                    i_decision += 1

        #jobs.flat->flattens the 2D jobs array; self.pool.map(Job.execute, jobs.flat)->runs all jobs in parallel using the pool
        #Converts results back into a numpy array with same shape as jobs
        return np.array(self.pool.map(Job.execute, jobs.flat)).reshape(*jobs.shape)