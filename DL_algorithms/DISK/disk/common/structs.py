import torch, abc, sys

from torch_dimcheck import dimchecked
from torchtyping import TensorType

# the class/object below is there for making type annotations like
# def my_function(args) -> NpArray[OutputType]
if sys.version_info >= (3, 7):
    class NpArray:
        def __class_getitem__(self, arg):
            pass
else:
    # 3.6 and below don't support __class_getitem__
    class _NpArray:
        def __getitem__(self, _idx):
            pass
    
    NpArray = _NpArray()

class Features:
    @dimchecked
    #Input: Kp=A tensor of shape (N,2); desc=A tensor of shape (N,F) ->Feature descriptors for each keypoint, with dimensionality F;
    # kp_logp=Atensor of shape (N,) -> the log-probabilities or confidence scores of each keypoint
    def __init__(self, kp: TensorType['N', 2], desc: TensorType['N', 'F'], kp_logp: TensorType['N']):
        #Ensures that all tensors live on the same device (CPU or GPU)
        assert kp.device == desc.device
        assert kp.device == kp_logp.device

        #Stores the provided tensors as attributes of the Feature object so they can be accessed later
        self.kp      = kp
        self.desc    = desc
        self.kp_logp = kp_logp

    @property
    #Tells us how many keypoints are stored
    def n(self):
        return self.kp.shape[0]

    @property
    #Tells us the execution device for the data
    def device(self):
        return self.kp.device

    #This method returns a fresh Feature object, whose descriptors and log-probabilities are trainable parameters 
    # starting from their current values
    def detached_and_grad_(self):
        return Features(
            self.kp,
            self.desc.detach().requires_grad_(),
            self.kp_logp.detach().requires_grad_(),
        )

    def requires_grad_(self, is_on):
        #desc and kp_logp are detached from the current computational graph -> used when wanting to treat features 
        # as new learnable tensors in a fresh optimization step
        self.desc.requires_grad_(is_on)
        self.kp_logp.requires_grad_(is_on)

    #It doesn't return a new object, it mutates in place
    def grad_tensors(self):
        return [self.desc, self.kp_logp]

    #Used to transfer the entire Features object to another device (CPU/GPU) or change its dtype
    def to(self, *args, **kwargs):
        return Features(
            self.kp.to(*args, **kwargs),
            self.desc.to(*args, **kwargs),
            self.kp_logp.to(*args, **kwargs) if self.kp_logp is not None else None,
        )

#Abstract base class; It defines a common interface for "match distributions" between two sets of features,
# but does not implement the details itself
class MatchDistribution(abc.ABC):
    @abc.abstractmethod
    #Should return a sample of matches between the two sets of features
    def sample(self) -> TensorType[2, 'K']:
        pass

    @abc.abstractmethod
    #"mle" -> "maximum likelihood estimate"; Should return the most likely matches (deterministic), instead of a sample
    def mle(self) -> TensorType[2, 'K']:
        pass

    @abc.abstractmethod
    #Should return the full dense probability distribution over all possible matches; Returns log-probabilities
    def dense_logp(self):
        pass

    @abc.abstractmethod
    #Returns raw probabilities
    def dense_p(self):
        pass

    @abc.abstractmethod
    #These methods return the two sets of Features objects being matched
    def features_1(self) -> Features:
        pass

    @abc.abstractmethod
    def features_2(self) -> Features:
        pass

    @property
    #Returns the "shape" of the matching problem; features_1().kp.shape[0]=number of keypoints in the first set
    def shape(self):
        return self.features_1().kp.shape[0], self.features_2().kp.shape[1]

    #This method is a convenient wrapper to turn a probabilistic match distribution into an explicit set of matched keypoint pairs
    def matched_pairs(self, mle=False):
        #If mle=True, it takes the most likely matches; otherwise, take a random sample of matches
        matches = self.mle() if mle else self.sample()

        return MatchedPairs(
            self.features_1().kp,   #Keypoint coords from set 1
            self.features_2().kp,   #Keypoint coords from set 2
            matches,                #The (2, K) tensor linking indices across sets
        )


#Represents pairs of matched keypoints between two features sets
class MatchedPairs:
    @dimchecked
    #The object know the two sets of keypoints; Which indicates correspond across them
    def __init__(self, kps1: TensorType['N', 2], kps2: TensorType['M', 2], matches: TensorType[2, 'K']):
        self.kps1    = kps1
        self.kps2    = kps2
        self.matches = matches

    #This lets you move the entire object (all tensors) to another device or dtype
    def to(self, *args, **kwargs):
        return MatchedPairs(
            self.kps1.to(*args, **kwargs),
            self.kps2.to(*args, **kwargs),
            self.matches.to(*args, **kwargs),
        )