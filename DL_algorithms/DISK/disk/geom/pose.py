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
    def T_4x4(self):
        I = torch.eye(4)
        I[:3, 3] = self.T

        return I
        
    def TR_4x4(self):
        return self.T_4x4 @ self.R_4x4

    @classmethod
    def from_poselike(cls, poselike):
        return cls(poselike.R, poselike.T)

    @staticmethod
    def relative(p1, p2, normed=False):
        assert isinstance(p1, Pose)
        assert isinstance(p2, Pose)

        ΔR = p2.R @ p1.R.T
        ΔT = p2.T - ΔR @ p1.T

        if normed:
            ΔT = ΔT / torch.norm(ΔT, p=2)

        return Pose(ΔR, ΔT)

    @staticmethod
    def error(p1, p2):
        assert isinstance(p1, Pose)
        assert isinstance(p2, Pose)

        R_ab       = p1.R.T @ p2.R
        half_trace = (torch.trace(R_ab) - 1) / 2
        θ_deg      = RAD_TO_DEG * math.acos(torch.clamp(half_trace, -1., 1.).item()) / 2

        cos_sim = F.cosine_similarity(
            p1.T.unsqueeze(0),
            p2.T.unsqueeze(0)
        ).squeeze(0)
        Δ_T = RAD_TO_DEG * math.acos(torch.clamp(cos_sim, -1., 1.).item())
            
        return PoseError(θ_deg, Δ_T)
        
def _normalized_cosine_error(v1: TensorType['N'], v2: TensorType['N']):
    EPS = 1e-15

    def normalize(v):
        v = v.to(torch.float64)
        return v / (torch.norm(v) + EPS)

    v1 = normalize(v1)
    v2 = normalize(v2)

    cos = max(EPS, 1. - torch.dot(v1, v2).pow(2).item())
    err = math.acos(math.sqrt(1. - cos))

    return err

@dimchecked
def matrix_to_quaternion(M: TensorType[3, 3]) -> TensorType[4]:
    '''
    adapted from
    https://github.com/vcg-uvic/sfm_benchmark/blob/2b28c76635f754cbc32f30571adf80f3eba13f4c/utils/eval_helper.py#L170
    '''

    #TODO: this appears unused
    m00 = M[0, 0].item()
    m01 = M[0, 1].item()
    m02 = M[0, 2].item()
    m10 = M[1, 0].item()
    m11 = M[1, 1].item()
    m12 = M[1, 2].item()
    m20 = M[2, 0].item()
    m21 = M[2, 1].item()
    m22 = M[2, 2].item()

    # symmetric matrix K
    K = np.array([
        [m00 - m11 - m22,       0.0,                         0.0,             0.0],
        [      m01 + m10,       m11 - m00 - m22,             0.0,             0.0],
        [      m02 + m20,             m12 + m21, m22 - m00 - m11,             0.0],
        [      m21 - m12,             m02 - m20,       m10 - m01, m00 + m11 + m22],
    ])
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    q = torch.from_numpy(q)

    if q[0] > 0.0:
        return q
    else:
        return -q