import torch
import numpy as np

from disk import MatchDistribution, Features, NpArray, Image

#This class is implementing a Reinforce-style loss for a model that learns correspondences between two images via keypoint matching
class Reinforce:
    #Input: reward=a callable that takes keypoints from two images and computes a reward for how good a proposed match is; 
    # lm_kp=a scalar weight for a keypoint penalty term, discouraging degenerate solutions (like predicting too many trivial keypoints)
    def __init__(self, reward, lm_kp):
        self.reward = reward
        self.lm_kp   = lm_kp

    #This computes the Reinforce loss for a pair of images and returns: the loss (to optimize); some stats (for logging/debugging)
    def _loss_for_pair(self, match_dist: MatchDistribution, img1: Image, img2: Image):
        #This is external supervision telling the model how good each match is
        #Reward(...)= computes a matrix [N,M] of rewards, one for each potential match between keypoints from img1 and img2;
        elementwise_rewards = self.reward(
            match_dist.features_1().kp, #the keypoints extracted from the two images
            match_dist.features_2().kp,
            img1,
            img2,
        )

        with torch.no_grad():
            # we don't want to backpropagate through this
            sample_p = match_dist.dense_p() # [N, M]; probability distribution over all possible keypoint matches (not differentiable, so detached)

        #Log-probabilities of each match (this is differentiable)
        sample_logp = match_dist.dense_logp() # [N, M]

        #Each keypoint has a log-probability of being selected; This creates a matrix [N,M] 
        # where each entry is the joint log-prob of selecting keypoints i and j
        kps_logp    = match_dist.features_1().kp_logp.reshape(-1, 1) \
                    + match_dist.features_2().kp_logp.reshape(1, -1)

        #This sums all keypoint log-probabilities, yielding a scalar penalty;Encourages the model to balance how many keypoints it proposes (via lm_kp)
        sample_lp_flat = match_dist.features_1().kp_logp.sum() \
                       + match_dist.features_2().kp_logp.sum()
        
        #This is the policy gradient trick: probability * log-probability; Multiplied by rewards later, it gives an unbiased gradient estimate
        sample_plogp = sample_p * (sample_logp + kps_logp)  #[N,M]

        reinforce  = (elementwise_rewards * sample_plogp).sum() #expected reward contribution (encourages good matches)
        kp_penalty = self.lm_kp * sample_lp_flat    #regularization for controlling keypoints
        #loss = -((elementwise_rewards * sample_plogp).sum() \
        #         + self.lm_kp * sample_lp_flat.sum())

        #Negative since we minimize i PyTorch (but we actually want to maximize reward)
        loss = -reinforce - kp_penalty

        #How many keypoints were detected across both images
        n_keypoints = match_dist.shape[0] + match_dist.shape[1]
        #Expected number of pairs (from probability mass)
        exp_n_pairs = sample_p.sum().item()
        #Expected reward under current distribution + keypoint penalty
        exp_reward  = (sample_p * elementwise_rewards).sum().item() \
                    + self.lm_kp * n_keypoints

        stats = {
            'reward'     : exp_reward,
            'n_keypoints': n_keypoints,
            'n_pairs'    : exp_n_pairs,
        }

        #Returns: loss=for backprop; stats=useful diagnostics
        return loss, stats

    #It's implementing a custom training loop that avoids holding huge intermediate tensors in memory, 
    # by backpropagating on-the-fly as each image pair is processed
    def accumulate_grad(
        self,
        images  : NpArray[Image],    # array [N_scenes, N_per_scene], each entry is an image
        features: NpArray[Features], # corresponding extracted features for those images; [N_scenes, N_per_scene]
        matcher,    #an object that can compute a match distribution between features
    ):
        '''
        This method performs BOTH forward and backward pass for the network
        (calling loss.backward() is not necessary afterwards).

        For every pair of covisible images we create a feature match matrix
        which is memory-consuming. In a standard forward -> backward PyTorch
        workflow, those would be all computed (forward pass), then the loss
        would be computed and finally backpropagation would be ran. In our
        case, since we don't need the matrices to stick around, we backprop
        through matching of each image pair on-the-fly, accumulating the
        gradients at Features level. Then, we finally backpropagate from
        Features down to network parameters.
        '''
        assert images.shape == features.shape

        N_scenes, N_per_scene = images.shape
        #Number of image pairs in each scene = C(N_per_scene, 2)
        N_decisions           = ((N_per_scene - 1) * N_per_scene) // 2

        #Container for logging training stats for each pair
        stats = np.zeros((N_scenes, N_decisions), dtype=object)

        # we detach features from the computation graph, so that when we call
        # .backward(), the computation will not flow down to the Unet. We
        # mark them as .requires_grad==True, so they will accumulate the
        # gradients across pairwise matches.
        detached_features = np.zeros(features.shape, dtype=object)
        for i in range(features.size):
            detached_features.flat[i] = features.flat[i].detached_and_grad_()

        # we process each scene in batch independently
        for i_scene in range(N_scenes):
            i_decision = 0
            scene_features = detached_features[i_scene]
            scene_images   = images[i_scene]

            # (N_per_scene choose 2) image pairs
            for i_image1 in range(N_per_scene):
                image1    = scene_images[i_image1]
                features1 = scene_features[i_image1]

                for i_image2 in range(i_image1+1, N_per_scene):
                    image2    = scene_images[i_image2]
                    features2 = scene_features[i_image2]

                    # establish the match distribution and calculate the
                    # gradient estimator
                    match_dist = matcher.match_pair(features1, features2)
                    loss, stats_ = self._loss_for_pair(match_dist, image1, image2)
                    # this .backward() will accumulate in `detached_features`
                    loss.backward()

                    stats[i_scene, i_decision] = stats_
                    i_decision += 1

        # here we "reattach" `detached_features` to the original `features`.
        # `torch.autograd.backward(leaves, grads)` API requires that we have
        # two equal length lists where for each grad-enabled leaf in `leaves`
        # we have a corresponding gradient tensor in `grads`
        leaves = []
        grads  = []
        for feat, detached_feat in zip(features.flat, detached_features.flat):
            leaves.extend(feat.grad_tensors())
            grads.extend([t.grad for t in detached_feat.grad_tensors()])
        #for i in range(features.size):
            #leaves.extend(features.flat[i].grad_tensors())
            #grads.extend([t.grad for t in detached_features.flat[i].grad_tensors()])

        # finally propagate the gradients down to the network
        torch.autograd.backward(leaves, grads)

        return stats