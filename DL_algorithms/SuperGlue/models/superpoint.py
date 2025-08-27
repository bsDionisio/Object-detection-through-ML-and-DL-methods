# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """

    #nms_radius=the radius around a peak to surppress other points
    assert(nms_radius >= 0)

    #Helper that applies a max pooling operation over a square window of size kernel_size around each pixel; 
    # it finds the local maximum in that neighbourhoog
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    #Compares each pixel to the max of its neighbourhood; it's True where the score is a local maximum
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        #Expands current ,ax_mask by marking a suppressed area around each maximum (max-pooling)
        supp_mask = max_pool(max_mask.float()) > 0
        #Zero out scores in suppressed areas
        supp_scores = torch.where(supp_mask, zeros, scores)
        #Compute new maxima in the remaining (non-suppressed) regions
        new_max_mask = supp_scores == max_pool(supp_scores)
        #Keep original maxima and add new maxima outside suppression zones.
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    #Keeps the scores where max_mask == True (local maxima), and zeroes out the rest
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    #Checks if the y-coordinate of each keypoint is not too close to the top or bottom borders
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    #Checks if the x-coordinate of each keypoint is not too close to the left or right borders
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    #Combines both masks: only keep keypoints that are far enough from all sides
    mask = mask_h & mask_w
    #Filters out the keypoints and scores using the combined mask
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    """Selects the top K keypoints based on their score"""
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape #b=batch size; c=descriptor dimension; h,w=height and width of descriptor map
    #Shift keypoints to align with descriptors
    keypoints = keypoints - s / 2 + 0.5
    #This maps the coordinates to the range expected by grid_sample, which takes normalized coordinates in the range [-1;1]
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    #Interpolates the descriptors at the specified keypoint locations using bilinear interpolation
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    #Reshape from (b,c,1,n) to (b,c,n) and apply L2 normalization across the descriptor channels so each vector has unit norm.
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,  #Descriptor vector size per keypoint
        'nms_radius': 4,    #Radius for non-maximum suppression
        'keypoint_threshold': 0.005,    #Min score to consider a point
        'max_keypoints': -1,    #Limit on how many keypoints to keep (-1 = no limit)
        'remove_borders': 4,    #Border margin for discarding keypoints
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        #Backbone (Shared Convolution) - Each pair uses 3x3 convolution with ReLU, 
        #followed by max-pooling layers (not shown here but likely applied in forward) 
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        #convPb outputs 65 channels: 64 spatial bins+1 "no keypoint" bin, 
        #used in softmax classification to detect interest points
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        #Outputs a dense descriptor map of shape [B, 256, H/8, W/8]
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        #Loads pre-trained weights from a local file
        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        #Ensures max_keypoints is a valid number (positive or -1)
        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        #Confirmation that the model loaded successful
        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        #Shared Encoder - Applies a series of convolution and ReLU layers
        #Intermediate x becomes a shared feature map used for both detection and description
        #Downsampling (via self.pool) happens 3 times -> output feature map is 1/8 of input resolution
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        #Computes a 65-channel heatmap, where 64 channels correspond to spatial bins
        #(8x8 grid), and 1 is a background class
        cPa = self.relu(self.convPa(x))
        #Apply softmax along the channel dimension (excluding the 65th "no interest point" class)
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        #Applies non-maximum suppression to retain only the strongest local peaks
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        #Finds pixel coordinates where score exceed the threshold. Colects score values at those keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        #Discard keypoints near the image borders to ensure descriptor realiability
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        #Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        #L2-normalizes the descriptor vectors (unit norm per pixel)
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        #Extract descriptors
        #For each image in the batch: -Use bilinear interpolation to sample descriptors at the exact keypoint locations
        #-Output: a descriptor vector for each keypoint
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }