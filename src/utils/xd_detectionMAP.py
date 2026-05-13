from typing import Sequence

import numpy as np


CLASS_LIST = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']
PROPOSAL_THRESHOLDS = np.arange(0.6, 0.7, 0.1)

def smooth(v):
   return v
   # l = min(5, len(v)); l = l - (1-l%2)
   # if len(v) <= 3:
   #   return v
   # return savgol_filter(v, l, 1) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)

def nms(dets, thresh=0.6, top_k=-1):
    """Pure Python NMS baseline."""
    # dets: N*2 and sorted by scores
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]  # start
    x2 = dets[:, 1]  # end
    lengths = x2 - x1 
    keep = []
    while order.size > 0:
        i = order[0] # the first is the best proposal
        keep.append(i) # put into the candidate pool
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) ## the intersection
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter) ## the iou
        inds = np.where(ovr <= thresh)[0]  # the index of remaining proposals
        order = order[inds + 1] # add 1

    return dets[keep], keep

def temporal_iou(s1: int, e1: int, s2: int, e2: int) -> float:
   inter = max(0, min(e1, e2) - max(s1, s2))
   union = (e1 - s1) + (e2 - s2) - inter
   return inter / union if union > 0 else 0.0


def _build_gt_by_class(
   gtsegments: np.ndarray,
   gtlabels: np.ndarray,
   class_to_idx: dict[str, int],
) -> list[list[tuple[int, int, int]]]:
   gt_by_class: list[list[tuple[int, int, int]]] = [[] for _ in class_to_idx]

   for video_idx in range(len(gtsegments)):
      for segment_idx in range(len(gtsegments[video_idx])):
         label = str(gtlabels[video_idx][segment_idx])
         if label not in class_to_idx:
            raise ValueError(f"Unknown XD label {label!r}")

         start, end = gtsegments[video_idx][segment_idx]
         gt_by_class[class_to_idx[label]].append((video_idx, int(start), int(end)))

   return gt_by_class


def _class_scores_and_filtered_predictions(
   predictions: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
   """Compute per-video class confidence and suppress classes with non-positive scores.

   For each ``(T, C)`` prediction matrix, class scores are the mean of the top
   ``max(1, T // 16)`` values in each class column after descending sort. The filtered
   prediction keeps columns whose class score is positive and zeros out the rest.
   """
   filtered_predictions: list[np.ndarray] = []
   class_scores: list[np.ndarray] = []

   for prediction in predictions:
      if prediction.shape[0] == 0:
         score = np.zeros(prediction.shape[1], dtype=prediction.dtype)
      else:
         sorted_scores = -np.sort(-prediction, axis=0) # sort columns by descending order (T,C)
         k = max(1, prediction.shape[0] // 16)
         score = np.mean(sorted_scores[:k, :], axis=0) # average top-k score of each class (C,)

      class_scores.append(score)
      filtered_predictions.append(prediction * (score > 0.0))

   return filtered_predictions, class_scores


def _extract_temporal_proposals(
   video_idx: int,
   scores: np.ndarray,
   class_score: float,
) -> list[list[float]]:
   proposals: list[list[float]] = []
   if scores.size == 0:
      return proposals

   score_min = np.min(scores)
   score_max = np.max(scores)
   for thr in PROPOSAL_THRESHOLDS:
      threshold = score_max - (score_max - score_min) * thr
      active = (scores > threshold).astype(np.float32)
      diff = np.diff(np.concatenate([np.zeros(1), active, np.zeros(1)]))
      starts = np.where(diff == 1)[0]
      ends = np.where(diff == -1)[0]

      for start, end in zip(starts, ends):
         if end - start >= 2:
            segment_score = np.max(scores[start:end]) + 0.7 * class_score
            proposals.append([video_idx, int(start), int(end), float(segment_score)])

   return proposals

def getLocMAP(
    predictions: Sequence[np.ndarray],
    iou_threshold: float,
    gtsegments: np.ndarray,
    gtlabels: np.ndarray,
    excludeNormal: bool,
) -> float:
   """Mean class-wise temporal detection score at a single IoU threshold (XD-Violence).

   Each entry in ``predictions`` is a ``(T, C)`` array of per-frame class scores (e.g.
   probabilities). Ground truth is given as per-video segment intervals and label strings.
   Proposals are obtained by thresholding score traces, then matched to GT with temporal
   IoU >= ``iou_threshold``. The returned value is ``100 * mean`` over the seven XD classes (not
   standard VOC AP); returns ``0`` early if any class yields no proposals.

   Args:
      predictions: One ``(T, C)`` score matrix per video (order matches evaluation split).
      iou_threshold: Temporal IoU threshold in ``[0, 1]`` for a prediction to count as a hit.
      gtsegments: ``object`` ndarray (or compatible): ``gtsegments[i][j]`` gives frame
         interval bounds for video ``i`` (see implementation for open vs closed range).
      gtlabels: Same nesting as ``gtsegments``; string labels from the XD class set.
      excludeNormal: If true, clip ``predictions`` to the first 500 videos (legacy eval).

   Returns:
      Scalar mean metric scaled to a percentage (``float``).
   """
   if excludeNormal:
       predictions = predictions[:500]

   class_to_idx = {label: idx for idx, label in enumerate(CLASS_LIST)}
   gt_by_class = _build_gt_by_class(gtsegments, gtlabels, class_to_idx)
   predictions, class_scores = _class_scores_and_filtered_predictions(predictions)
   ap = []
   for c in range(len(CLASS_LIST)):
      segment_predict = []
      # Get list of all predictions for class c
      for i, prediction in enumerate(predictions):
         tmp = smooth(prediction[:, c]) # (T,)
         segment_predict_multithr = _extract_temporal_proposals(i, tmp, class_scores[i][c])
         if len(segment_predict_multithr) != 0:
            segment_predict_multithr = np.array(segment_predict_multithr)
            segment_predict_multithr = segment_predict_multithr[np.argsort(-segment_predict_multithr[:,-1])] # sort segment scores by descending order
            _, keep = nms(segment_predict_multithr[:, 1:-1], 0.6) # [[s[i],e[i]]]
            segment_predict.extend(list(segment_predict_multithr[keep]))
      segment_predict = np.array(segment_predict) # elements of [video_index, start_index, end_index, score]

      # Sort the list of predictions for class c based on score
      if len(segment_predict) == 0:
         return 0.0
      segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]

      # Create gt list
      segment_gt = list(gt_by_class[c])
      gtpos = len(segment_gt)

      # Compare predictions and gt
      tp, fp = [], []
      for i in range(len(segment_predict)):
         matched = False
         best_iou = 0.0
         best_gt_idx = None
         for j, (gt_video_idx, gt_start, gt_end) in enumerate(segment_gt):
            if int(segment_predict[i][0]) == gt_video_idx:
               iou = temporal_iou(gt_start, gt_end, int(segment_predict[i][1]), int(segment_predict[i][2]))
               if iou >= iou_threshold:
                  matched = True
                  if best_gt_idx is None or iou > best_iou:
                     best_iou = iou
                     best_gt_idx = j
         if matched and best_gt_idx is not None:
            del segment_gt[best_gt_idx]
            tp.append(1.0) # true positive
            fp.append(0.0) # false positive
         else:
            tp.append(0.0)
            fp.append(1.0)

      tp = np.array(tp, dtype=float)
      fp = np.array(fp, dtype=float)
      tp_c = np.cumsum(tp)
      fp_c = np.cumsum(fp)
      if gtpos == 0 or np.sum(tp) == 0:
         prc = 0.
      else:
         prc = np.sum((tp_c / (fp_c + tp_c)) * tp) / gtpos # average precision: integration over PR curve
      ap.append(prc)
      # print(np.round(prc, 4))
   return 100 * np.mean(ap)


def getDetectionMAP(predictions, segments, labels, excludeNormal=False):
   iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
   # iou_list = [0.5]
   dmap_list = []
   for iou in iou_list:
      # print('Testing for IoU {:.1f}'.format(iou))
      dmap_list.append(getLocMAP(predictions, iou, segments, labels, excludeNormal))
   return dmap_list, iou_list

