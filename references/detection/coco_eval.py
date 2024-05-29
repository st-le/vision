import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets import simplecoco


def to_multihead_keypoint_array(self : simplecoco = None, target = None):
    if self.with_keypoint:
        org_target_keypoints_structure = [len(klb) for klb in target["keypoint_labels"]]

        # pass the multi-head structure
        # target['keypoint_muticlass_structure'] = org_target_keypoints_structure
        max_num = max(org_target_keypoints_structure)

        # uncollapse the first dimension
        target_kp_labels = np.array(target['keypoint_labels']).flatten().tolist()
        target_kp_catids = np.array(target['keypoint_catids']).flatten().tolist()
        target_kps = np.array(target['keypoints'])
        assert len(target_kps.shape)==3, "keypoints must be a 3D array"

        vis = np.array(target['keypoint_visibility'])
        assert len(vis.shape)==2, "visibility must be a 2D array"        
        target_kps_vis = np.concatenate((target_kps, np.expand_dims(vis,axis=2)), axis=2)
        target_kp_catids_norep = []
        target_kp_class_range = []
        idx=0
        cid=target_kp_catids[idx]
        target_kp_catids_norep.append(cid)
        rng = [idx,idx]
        while idx<len(target_kp_catids):
            while cid==target_kp_catids[idx]:
                if idx+1<len(target_kp_catids):
                    idx+=1
                    rng[-1] = idx
                else:
                    rng[-1]+=1
                    idx=-1
                    break
            if idx>=0:
                cid=target_kp_catids[idx]
                target_kp_catids_norep.append(cid)
                target_kp_class_range.append(rng)
                rng=[idx,idx]
            else:
                target_kp_class_range.append(rng)
                break                    
        all_keypoint_labels = []
        _ = [all_keypoint_labels.append(vv) for v in self.keypoint_labels.values() for vv in v ]

        all_keypoint_catids = [] 
        for i, k in enumerate(self.keypoint_labels.keys()):
            aa = [self.json_category_id_to_contiguous_id[k]]*self.num_keypoints[i]
            for aaa in aa:
                all_keypoint_catids.append(aaa)

        kpt_keys = self.kpt_keys
        target1=dict()
        for kk in kpt_keys[:-1]:
            target1[kk] = []
        for i, _ in enumerate(target_kp_catids_norep):
            for kk in kpt_keys[:-1]:
                tmp=[]
                acc= 0

                if kk=="keypoint_catids":
                    target1[kk].append(all_keypoint_catids)
                elif kk=="keypoint_labels":
                    target1[kk].append(all_keypoint_labels)
                else:
                    tmp_ = np.zeros((sum(self.num_keypoints), 3))

                    rng_a, rng_b = target_kp_class_range[i]
                    # for ii in range(target_kps.shape[0]):
                    for ii in range(rng_a, rng_b):

                        xx = target_kp_catids[ii]-1
                        kp_cls_lbs = self.keypoint_labels[self.contiguous_category_id_to_json_id[target_kp_catids[ii]]]
                        yy = kp_cls_lbs.index(target_kp_labels[ii])

                        tmp_[sum(self.num_keypoints[:xx])+yy] = target_kps_vis[ii]
                    target1[kk].append(tmp_)
                
        target1['keypoints']=np.stack(target1['keypoints'])
    else:
        raise NotImplementedError("only work with keypoint")

    return target1

def prepare_multi_keypoint_target(self : simplecoco = None, anno=None, chosen_cls=None):
    """ function is modified from SimpleCOCO class getitem 
        Return:
            target_kps (list) with size 3 \times num_kpt
    """

    kpts = []
    kpt_catids = []

    kpts_xy=[]
    kpts_xy_lbl=[]
    kpts_xy_vis=[]
    target = dict()
    if chosen_cls is None:
        boxes = [obj["bbox"] for obj in anno]
        if self.with_keypoint:
            assert isinstance(self.num_keypoints, list), "num_keypoints must be a list"
            kpts = []
            tg_kps=[]
            for obj in anno:
                kpts_xy.append([])
                kpts_xy_lbl.append([])
                kpts_xy_vis.append([])
                kpt_catids.append([])

                tg_kps.append([])

                nkpt = self.num_keypoints[self.json_category_id_to_contiguous_id[obj['category_id']]-1]  
                if 'keypoints' in obj.keys():
                    kpts.append(obj['keypoints']) # get all the keypoints
                
                    for _ in range(len(obj['keypoints'])//3):
                        kpt_catids[-1].append(self.json_category_id_to_contiguous_id[obj['category_id']])
                else: 
                    # still need to add invisible keypoints
                    obj_kpt = [0,0,0] * nkpt # [0,0,0] is an invisible keypoint
                    kpts.append(obj_kpt)  
                    tmp = [self.json_category_id_to_contiguous_id[obj['category_id']]] * nkpt
                    for t in tmp:
                        kpt_catids[-1].append(t) 

                for i in range(0,len(kpts[-1]),3):
                    kpts_xy[-1].append(kpts[-1][i:i+2])
                    kpts_xy_lbl[-1].append(self.keypoint_labels[self.contiguous_category_id_to_json_id[kpt_catids[-1][-1]]][i//3])  
                    kpts_xy_vis[-1].append(kpts[-1][i+2])

                    for v in kpts[-1][i:i+3]:
                        tg_kps[-1].append(v)

            target["keypoints"] = kpts_xy
            target["keypoint_labels"] = kpts_xy_lbl
            target["keypoint_catids"] = kpt_catids
            target["keypoint_visibility"] = kpts_xy_vis

            target = to_multihead_keypoint_array(self, target)
        else:
            NotImplementedError                
    else:
        raise NotImplementedError("multi-keypoint does not work with chosen_cls")
    
    return target['keypoints']

class myCOCOeval(COCOeval):
    def __init__(self,*args, num_keypoints=17, testset=None, **kwargs):
        COCOeval.__init__(self,*args,**kwargs)
        self.num_keypoints=num_keypoints
        self.testset=testset
        if isinstance(num_keypoints, list):
            self.params.kpt_oks_sigmas_single = self.params.kpt_oks_sigmas
            self.params.kpt_oks_sigmas_multi = np.zeros(sum(num_keypoints))
            idx=0
            for nkp in self.num_keypoints:
                self.params.kpt_oks_sigmas_multi[idx:idx+nkp] = self.params.kpt_oks_sigmas_single[:nkp]
                idx+=nkp

    def limit_keypoints(self):
        if isinstance(self.num_keypoints, int):
            self.params.kpt_oks_sigmas = self.params.kpt_oks_sigmas[:self.num_keypoints]

    def computeOks(self, imgId, catId):
        """ basically copied from pycocotools::cocoeval.py::COCOeval::computeOks 
            but add the num_keypoints 
        """
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        if isinstance(self.num_keypoints, list):
            sigmas = p.kpt_oks_sigmas_multi
        else:
            sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)

        tg_kps = prepare_multi_keypoint_target(self.testset, gts, None)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            if not isinstance(self.num_keypoints, list):
                g = np.array(gt['keypoints'])[:self.num_keypoints*3]  
            else:
                g = np.array(tg_kps) if not isinstance(tg_kps, np.ndarray) else tg_kps
                g = g.reshape(-1,3).flatten()
                
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, num_keypoints=None, testset=None):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        self.testset=testset
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            if iou_type == 'keypoints':
                self.coco_eval[iou_type] = myCOCOeval(coco_gt, iouType=iou_type, num_keypoints=num_keypoints, testset=self.testset)
                self.coco_eval[iou_type].limit_keypoints()

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))
