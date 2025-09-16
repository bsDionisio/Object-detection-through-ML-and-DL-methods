import torch
from torch import nn
from torch.distributions import Categorical
from torch_dimcheck import dimchecked
from torchtyping import TensorType

from disk import Features, NpArray, MatchDistribution
from disk.geom import distance_matrix

#Define a probabilistic model over correspondences between two sets of features, while enforcing some form of cycle-consistency
class ConsistentMatchDistribution(MatchDistribution):
    def __init__(
        self,
        features_1: Features,
        features_2: Features,
        inverse_T: float,
    ):
        #hold descriptors (vectors that represnt data points)
        self._features_1 = features_1
        self._features_2 = features_2
        self.inverse_T = inverse_T

        #Computes pairwise distances between feature descriptors
        distances = distance_matrix(
            self.features_1().desc,
            self.features_2().desc,
        )
        #Turns distances into affinity scores (closer features = higher affinity)
        affinity = -inverse_T * distances

        #Row-rise distribution (N->M) - probability of matching feature from set 1 to set 2
        self._cat_I = Categorical(logits=affinity)
        #Column-wise distribution (M->N) - probability of matching feature from set 2 back to set 1
        self._cat_T = Categorical(logits=affinity.T)

        self._dense_logp = None
        self._dense_p    = None

    @dimchecked
    def dense_p(self) -> TensorType['N', 'M']:
        if self._dense_p is None:
            #self._cat_I.probs gives P(i->j) (match from feature set 1 to 2);
            #self._cat_T.probs.T gives P(j->i)
            #Multiplying them gives a cycle-consistent probability matrix
            self._dense_p = self._cat_I.probs * self._cat_T.probs.T

        return self._dense_p

    @dimchecked
    #Similar to dense_p, but in log-space
    def dense_logp(self) -> TensorType['N', 'M']:
        if self._dense_logp is None:
            self._dense_logp = self._cat_I.logits + self._cat_T.logits.T

        return self._dense_logp

    @dimchecked
    #This method filters correspondences so that only cycle-consistent ones remain
    #left[i] says "feature in set 1 maps to feature j in set 2"; right[j] says "feature j in set 2 maps back to feature i"
    def _select_cycle_consistent(self, left: TensorType['N'], right: TensorType['M']) -> TensorType[2, 'K']:
        indexes = torch.arange(left.shape[0], device=left.device)
        #Cycle-consistency condition
        cycle_consistent = right[left] == indexes

        paired_left = left[cycle_consistent]

        #Returns stacked indices of consistent matches: shape [2,K]
        return torch.stack([
            right[paired_left],
            paired_left,
        ], dim=0)

    @dimchecked
    def sample(self) -> TensorType[2, 'K']:
        #Draws a random match for each feature in set 1 -> set 2
        samples_I = self._cat_I.sample()
        #Draws a random match for each feature in set 2 -> set 1
        samples_T = self._cat_T.sample()

        #Returns a random set of consistent correspondences between features 1 and 2
        return self._select_cycle_consistent(samples_I, samples_T)

    @dimchecked
    def mle(self) -> TensorType[2, 'K']:
        #Best match for each feature in set 1
        maxes_I = self._cat_I.logits.argmax(dim=1)
        #Best match for each feature in set 2
        maxes_T = self._cat_T.logits.argmax(dim=1)

        # FIXME UPSTREAM: this detachment is necessary until the bug is fixed
        #The detach() calls break the computational graph: That means these values won't propagate gradients
        maxes_I = maxes_I.detach()
        maxes_T = maxes_T.detach()

        #Returns a deterministic, best-guess set of consistent correspondences
        return self._select_cycle_consistent(maxes_I, maxes_T)

    #Just simple getters to access the original feature sets; This makes the API clean and avoids 
    # directly exposing _feature1 and _feature2
    def features_1(self) -> Features:
        return self._features_1

    def features_2(self) -> Features:
        return self._features_2

#This is essentially a wrapper module that procedures a ConsistentMatchDistribution object given two sets of features
class ConsistentMatcher(torch.nn.Module):
    def __init__(self, inverse_T=1.):
        super(ConsistentMatcher, self).__init__()
        #inverse_T is the inverse temperature hyperparameter we saw before (controls sharpness of softmax in distribution)
        self.inverse_T = nn.Parameter(torch.tensor(inverse_T, dtype=torch.float32))

    #Makes debugging and inspection easier
    def extra_repr(self):
        return f'inverse_T={self.inverse_T.item()}'

    #Takes two sets of features; Returns a ConsistentMatchDistribution; 
    # inverse_T here is the learnable parameter, so the distribution depends on it
    def match_pair(self, features_1: Features, features_2: Features):
        return ConsistentMatchDistribution(features_1, features_2, self.inverse_T)