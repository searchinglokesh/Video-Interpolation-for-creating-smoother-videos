# Video Frame Interpolation

## The Goal

The ultimate goal for Video Frame Interpolation is to improve the frame rate and enhance the visual quality of a video. Video Frame Interpolation's objective is to create many frames in the midst of two neighboring frames of the original video, or it aims to synthesize nonexistent frames in-between the original frames. Traditionally used ones before the introduction of Neural Networks linearly combined the frames to approximate intermediate flows, leading to artifacts around motion boundaries.
Example(Before & Interpolated): https://drive.google.com/drive/folders/1-OMJCP3TH7DWwTAE2Fdr_F8WGp99inrf?usp=sharing
 


## Introduction

Video Frame Interpolation (VFI) aims to synthesize intermediate frames between two consecutive video frames. It's commonly utilized to boost frame rate and visual quality. Furthermore, real-time VFI algorithms working on high-resolution movies have a wide range of applications, including boosting the frame rate of video games and live broadcasts, as well as offering visual enhancement.
There are many practical applications as  video enhancement, Automated Animation, video compression , slow-motion generation , and view synthesis.
Furthermore, real-time VFI algorithms operating on high-resolution videos have a wide range of applications, including boosting the frame rate of video games and live broadcasts, as well as offering video enhancement services to consumers with limited computer capabilities.
VFI is challenging due to the complex, large non-linear motions and illumination changes in the real world. Recently, VFI algorithms have offered a framework to address these challenges and achieved impressive results

### common approaches
Common approaches for these methods involve two steps:
1) warping the input frames according to approximated optical flows and 
2) fusing and refining the warped frames using Convolutional Neural Networks (CNNs).
Flow-based approaches must approximate the intermediate flows Ft->0, Ft->1 from the perspective of the frame given the input frames I(0), I(1) which are expected to synthesize it.
Forward warping based methods and Backward warping based methods are the two types of flow-based VFI algorithms. Because forward warping lacks a consistent implementation and suffers from the conflict of mapping several source pixels to the same position, which results in overlapped pixels and holes, backward warping is more extensively utilized.

However, the following two difficulties make this difficult for conventional flow-based VFI models:
1)  Some extra components are required: To compensate for the inadequacies of intermediate flow estimates, the image depth model, flow refinement model, and flow reversal layer are introduced. As substructures, these technologies also necessitate pre-trained state-of-the-art optical flow models that are not specifically intended for VFI activities.
2)  For the approximated intermediate flows, there is no management: Most earlier interpolation models, to our knowledge, have only been trained with the final reconstruction loss. There is no other explicit supervision for the flow estimating process, which degrades interpolation performance.

## Method and Working

 Real-time Intermediate Flow Estimation (RIFE) Algorithm
 There are two major components in RIFE: 
(1) Efficient intermediate flow estimation with the IFNet. 
(2) Fusion process of the warped frames using a FusionNet. We describe the details of these two components in this subsection. 
We employ a coarse-to-fine strategy with gradually increasing resolutions, Specifically, we first compute a rough prediction of the flow on low resolutions, which is believed to capture large motions easier, then iteratively refine the flow fields with gradually increasing resolutions.
We can apply RIFE recursively to interpolate multiple intermediate frames at different timesteps t ∈ (0, 1). Specifically, given two consecutive input frames I(0), I(1), we apply RIFE once to get intermediate frame     at t = 0.5. We feed I(0) and      to get       , and we can repeat this process recursively to interpolate multiple frames.

## Our Proposed Solution

Although RIFE is fast but in terms of the quality it still lacks behind the DAIN(2019) algorith for interpolation which uses depth data of individual pixel for accurate interpolation,So we propose 2 solutions

1) Stabilising the video before interpolating: Since the interpolation gets data from video and can't accurately discriminate between object and backgorund artifacts are usually created in the output video,this is ooften not due to the incapability of the algorith but due to sudden movements in the input which makes it harder for detection
So it's better to stabilise the video before we run the algorithm to decrease the amount of the artifacts created.So we use another stabilization for this.

2) Replacing the IFNet part with more latest analytical methods: Gunner-Farneback and Lucas-Kanade. These are the newer FusionNet and ContextNet models which are fine-tuned on the basis of the proposed solution.

## Compared algorithms

[1] RIFE - 2021

https://arxiv.org/abs/2011.06294

[2] Softmax splatiing - 2020

https://arxiv.org/pdf/2003.05534v1.pdf

[3] DAIN (Depth-Aware Video Frame Interpolation) - 2019

https://arxiv.org/pdf/1904.00830.pdf

[4] MEMC-Net (Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement) - 2019

https://arxiv.org/pdf/1810.08768.pdf

[5] Super SloMo (Deep Slow Motion Video Reconstruction with Hybrid Imaging System) - 2018

https://arxiv.org/pdf/1712.00080.pdf

[6] SepConv - 2017

https://arxiv.org/pdf/1708.01692.pdf

## Why use RIFE ?

Even though there are multiple video frame interpolation which in some cases turn out to produce much more smoother interpolations or much more smoother and cleaner video without much artifacts. RIFE is unparalleled in it's speed of production and could be further upgraded to actually use in games ,which in turn need very less computational hardware but still produce frame rate to the top GPU's in the industry also decrease the burden on the computating hardware.

## Datasets

- Vimeo90K: http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip - used for training other competitive interpolations too
- Middlebury: https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf

## Examples
https://drive.google.com/drive/folders/1-OMJCP3TH7DWwTAE2Fdr_F8WGp99inrf?usp=sharing

## Hardware

GPU Nvidia GeForce 1660 Ti
CPU Intel i7-9750H
