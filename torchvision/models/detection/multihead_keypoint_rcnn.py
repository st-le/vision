import torch
from torch import nn

from torchvision.ops import MultiScaleRoIAlign

import vision
import vision.torchvision
import vision.torchvision.models
import vision.torchvision.models.detection
import vision.torchvision.models.detection.keypoint_rcnn
import vision.torchvision.models.detection.backbone_utils

from vision.torchvision.models.detection._utils import overwrite_eps
from vision.torchvision.models.utils import load_state_dict_from_url

from vision.torchvision.models.detection.faster_rcnn import FasterRCNN

from vision.torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from vision.torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, model_urls


def multihead_keypointrcnn_resnet50_fpn(pretrained=False, progress=True,
                              num_classes=2, num_keypoints=17,
                              pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - keypoints (``FloatTensor[N, K, 3]``): the ``K`` keypoints location for each of the ``N`` instances, in the
          format ``[x, y, visibility]``, where ``visibility=0`` means that the keypoint is not visible.

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
        - keypoints (``FloatTensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

    Keypoint R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "keypoint_rcnn.onnx", opset_version = 11)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        num_keypoints (int): number of keypoints, default 17
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = MultiheadKeypointRCNN(backbone, num_classes, num_keypoints=num_keypoints, **kwargs)
    if pretrained:
        key = 'keypointrcnn_resnet50_fpn_coco'
        if pretrained == 'legacy':
            key += '_legacy'
        state_dict = load_state_dict_from_url(model_urls[key],
                                              progress=progress)
        if hasattr(model.roi_heads.keypoint_predictor, 'kps_score_lowres_array') or \
            state_dict['roi_heads.keypoint_predictor.kps_score_lowres.weight'].shape[1] != \
            model.roi_heads.keypoint_predictor.kps_score_lowres.weight.shape[1]:
            state_dict.pop('roi_heads.keypoint_predictor.kps_score_lowres.weight')
            state_dict.pop('roi_heads.keypoint_predictor.kps_score_lowres.bias')
            state_dict.pop('roi_heads.box_predictor.bbox_pred.weight')
            state_dict.pop('roi_heads.box_predictor.bbox_pred.bias')
            state_dict.pop('roi_heads.box_predictor.cls_score.weight')
            state_dict.pop('roi_heads.box_predictor.cls_score.bias')

            strict_load = False
        else: 
            strict_load = True

        model.load_state_dict(state_dict, strict=strict_load)
        overwrite_eps(model, 0.0)
    return model


class MultiheadKeypointRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(MultiheadKeypointRCNNPredictor, self).__init__()
        assert isinstance(num_keypoints, list), 'multihead requires a list of keypoints to do configuration'

        input_features = in_channels
        deconv_kernel = 4

        self.kps_score_lowres_array = nn.ModuleList()
        self.up_scale = 2
        self.out_channels_array = []
        for num_kpt in num_keypoints:
            kps_score_lowres = nn.ConvTranspose2d(
                input_features,
                num_kpt,
                deconv_kernel,
                stride=2,
                padding=deconv_kernel // 2 - 1,
            )
            nn.init.kaiming_normal_(
                kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.constant_(kps_score_lowres.bias, 0)
            
            out_channels = num_kpt

            self.kps_score_lowres_array.append(kps_score_lowres)
            self.out_channels_array.append(out_channels)

    def forward(self, x):
        outs = []
        for kps_score_lowres in self.kps_score_lowres_array:
            x_ = kps_score_lowres(x)
            x__ = torch.nn.functional.interpolate(
                x_, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False, recompute_scale_factor=False
            )
            outs.append(x__)

        return torch.cat(outs, axis=1) 


class MultiheadKeypointRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=None, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # keypoint parameters
                 keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None,
                 num_keypoints=[17],
                 keypoint_vis=False):

        assert isinstance(num_keypoints, list)

        assert isinstance(keypoint_roi_pool, (MultiScaleRoIAlign, type(None)))
        if min_size is None:
            min_size = (640, 672, 704, 736, 768, 800)

        if num_classes is not None:
            if keypoint_predictor is not None:
                raise ValueError("num_classes should be None when keypoint_predictor is specified")

        out_channels = backbone.out_channels

        if keypoint_roi_pool is None:
            keypoint_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=14,
                sampling_ratio=2)

        if keypoint_vis:
            keypoint_vis_head_out = 512
            assert isinstance(num_keypoints,list)
            keypoint_vis_predictor = KeypointVisRCNNPredictor(keypoint_vis_head_out, sum(num_keypoints)) # max(num_keypoints)

        if keypoint_head is None:
            keypoint_layers = tuple(512 for _ in range(8))
            keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)

        if keypoint_predictor is None:
            keypoint_dim_reduced = 512  # == keypoint_layers[-1]
            keypoint_predictor = MultiheadKeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)

        super(MultiheadKeypointRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights)

        self.roi_heads.keypoint_roi_pool = keypoint_roi_pool
        self.roi_heads.keypoint_head = keypoint_head
        self.roi_heads.keypoint_predictor = keypoint_predictor  # replace the multihead predictor here!

        self.roi_heads.keypoint_vis_predictor = keypoint_vis_predictor

class KeypointVisRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointVisRCNNPredictor, self).__init__()

        # self.vis_heads = nn.Linear(in_channels, num_keypoints)
        self.vis_heads = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1) # similar to RPNHead
        self.relu = nn.ReLU(inplace=True)

        in_dim = 128*14*14
        self.vis_logits = nn.Linear(in_dim,num_keypoints) # 25088 is the flattened size of the feature output by vis_heads
        nn.init.kaiming_normal_(self.vis_heads.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.vis_heads.bias, 0)

    def forward(self,x):
        x = self.relu(self.vis_heads(x))
        N = x.shape[0]
        x = x.reshape(N,-1) if x.numel() != 0 else x.reshape(-1,128*14*14)
        return self.vis_logits(x)