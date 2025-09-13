import typing, torch, math
import numpy as np
import torch.nn.functional as F

from torch_dimcheck import dimchecked
from torchtyping import TensorType

RAD_TO_DEG = 57.29

class PoseError(typing.NamedTuple):
    #Represent an angular error (probbly in radians or degrees)
    Δ_θ    : float
    #Represents a translation error
    Δ_T    : float

    #This overrides the default string representation
    def __repr__(self):
        return f'<PoseError Δ_θ={self.Δ_θ:.3f}, Δ_T={self.Δ_T:.3f}>'

    #This method converts the object into a dictionary
    def to_dict(self):
        return {
            'Δ_θ'    : self.Δ_θ,
            'Δ_T'    : self.Δ_T,
        }

class Pose(typing.NamedTuple):
    R: torch.Tensor     #a rotation matrix (3x3 tensor)
    T: torch.Tensor     #a translation vector (3-element tensor)

    #Defines how the object is printed as a string
    def __str__(self):
        return f'\nR=\n{self.R}\nT=\n{self.T}'

    #This is a PyTorch convention to move tensors between devices (CPU/GPU) or change dtype
    def to(self, *args, **kwargs):
        #Returns a new Pose object with updated tensors (because named tuples are immutable)
        return Pose(
            self.R.to(*args, **kwargs),
            self.T.to(*args, **kwargs),
        )
    
    @property
    #Converts the 3x3 rotation matrix into a 4x4 homogeneous transformation matrix
    def R_4x4(self):
        I = torch.eye(4)
        I[:3, :3] = self.R

        return I

    @property
    #Converts the translation vector into a 4x4 homogeneous translation matrix; Allows cobining rotation ans translation with matrix multiplication
    def T_4x4(self):
        I = torch.eye(4)
        I[:3, 3] = self.T

        return I
        
    #Multiplies the translation matrix by rotation matrix to get the full pose as a 4x4 matrix
    def TR_4x4(self):
        #@ is matrix multiplication in Python
        return self.T_4x4 @ self.R_4x4

    #This means the method receives the class itself (cls) instead of an instance (self)
    @classmethod
    #It's a constructor helper that creates a Pose from any pose-like object; useful for interoperability when working with similar data strunctures
    def from_poselike(cls, poselike):
        return cls(poselike.R, poselike.T)

    #This means this function doesn't depend on a specific instance or class, just works on inputs
    @staticmethod
    #Computes the relative pose of p1 and p2
    def relative(p1, p2, normed=False):
        assert isinstance(p1, Pose)
        assert isinstance(p2, Pose)

        #Reltive rotation; p1.R.T= inveerse rotation of p1; So this is the rotation from p1's frame to p2's frame
        ΔR = p2.R @ p1.R.T
        #Relative translation; This computes the translation difference, expressed in the new frame
        ΔT = p2.T - ΔR @ p1.T

        #If true;    Useful if you only care about direction, not distance
        if normed:
            #The translation vector is converted into a unit direction vector
            ΔT = ΔT / torch.norm(ΔT, p=2)

        #The result is itself a Pose object - rotation ΔR and translation ΔT
        return Pose(ΔR, ΔT)

    #It doesn't depend on a Pose instance, just works with p1 and p2
    @staticmethod
    #Measures how "different" two poses are; This blends rotation error and translation error into a PoseError
    def error(p1, p2):
        assert isinstance(p1, Pose)
        assert isinstance(p2, Pose)

        #This is the relative rotation matrix between p1 and p2
        R_ab       = p1.R.T @ p2.R  #If R_ab=I, then the rotations are identical
        #Uses the trace method to extract the angle of rotation; Standard formula in 3D rotation math
        half_trace = (torch.trace(R_ab) - 1) / 2
        #Calculates the angular error in degrees
        θ_deg      = RAD_TO_DEG * math.acos(torch.clamp(half_trace, -1., 1.).item()) / 2

        #Computes the cosine similarity between the two translation vectors; If the point in the same direction, similarity=1; opposite similarity=-1
        cos_sim = F.cosine_similarity(
            #unsqueeze(0) makes them 2D tensors so cosine_similarity can work
            p1.T.unsqueeze(0),
            p2.T.unsqueeze(0)
        ).squeeze(0)
        #Directional error between the trnaslarion vectors; Converts cosine similarity into an angle in degrees
        Δ_T = RAD_TO_DEG * math.acos(torch.clamp(cos_sim, -1., 1.).item())  #If translation are identical directions->Δ_T=0º; If opposite, Δ_T=180º
            
        #Packs the errors into your earlier PoseError class
        return PoseError(θ_deg, Δ_T)

#Low-level helper for computing a cosine-based error between the vectors
def _normalized_cosine_error(v1: TensorType['N'], v2: TensorType['N']):
    #A tiny constant to avoid division by zero or invalid acos inputs
    EPS = 1e-15

    #Normalization function
    def normalize(v):
        #Ensures the vector is float64 (higher precision)
        v = v.to(torch.float64)
        #Normalizes it to unit length (magnitude=1); Adds EPS to deniminator for stability
        return v / (torch.norm(v) + EPS)

    #Both vectors are now unit vectors
    v1 = normalize(v1)
    v2 = normalize(v2)

    #This ensures the value is equal or bigger than EPS
    cos = max(EPS, 1. - torch.dot(v1, v2).pow(2).item())
    #Error compuation
    err = math.acos(math.sqrt(1. - cos))

    return err

@dimchecked
#Input: 3x3 rotation matrix M (PyTorch tensor)
#Output: Returns a 4-element quaternion [w, x, y, z]
def matrix_to_quaternion(M: TensorType[3, 3]) -> TensorType[4]:
    '''
    adapted from
    https://github.com/vcg-uvic/sfm_benchmark/blob/2b28c76635f754cbc32f30571adf80f3eba13f4c/utils/eval_helper.py#L170
    '''

    #TODO: this appears unused
    #Pulls out scalar values from the 3x3 rotation matrix for convenience
    m00 = M[0, 0].item()
    m01 = M[0, 1].item()
    m02 = M[0, 2].item()
    m10 = M[1, 0].item()
    m11 = M[1, 1].item()
    m12 = M[1, 2].item()
    m20 = M[2, 0].item()
    m21 = M[2, 1].item()
    m22 = M[2, 2].item()

    #This constructs a symmetric 4x4 matrix K; This formulation comes from Bar-Itzhack's method (and related work) for 
    # converting a rotation matrix to a quaternion
    K = np.array([
        [m00 - m11 - m22,       0.0,                         0.0,             0.0],
        [      m01 + m10,       m11 - m00 - m22,             0.0,             0.0],
        [      m02 + m20,             m12 + m21, m22 - m00 - m11,             0.0],
        [      m21 - m12,             m02 - m20,       m10 - m01, m00 + m11 + m22],
    ])
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)    #V stores eigenvectors in columns, so we grab the right one
    #The [3,0,1,2] reorders components so that the quaternion in returned as [w, x, y, z] instead of [x, y, z, w]
    q = V[[3, 0, 1, 2], np.argmax(w)]

    #Converts numpy -> PyTorch tensor
    q = torch.from_numpy(q)

    #A quaternion and its negation represent the same rotation
    if q[0] > 0.0:
        return q
    else:
        return -q