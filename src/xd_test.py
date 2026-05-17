import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option

def test(
    model: CLIPVAD,
    testdataloader: DataLoader,
    maxlen: int,
    prompt_text: list[str],
    gt: np.ndarray,
    gtsegments: np.ndarray,
    gtlabels: np.ndarray,
    device: str | torch.device,
) -> tuple[float, float, int]:
    """
    Evaluate a trained VadCLIP model on the XD-Violence test split (coarse and fine-grained).

    For each sample, runs forward with optional multi-window padding for long clips, then
    concatenates frame-level scores across the dataset. Reports branch-1 (classifier) and
    branch-2 (alignment) ROC-AUC / AP against frame-level ``gt`` (with 16x repetition to
    match CLIP stride), and detection mAP from softmax alignment logits vs ``gtsegments`` /
    ``gtlabels``.

    Args:
        model: Trained ``CLIPVAD`` instance.
        testdataloader: Typically ``batch_size=1`` loader yielding ``(visual_feat, label,
            feat_length)`` per batch (see ``XDDataset`` in test mode).
        maxlen: Temporal length ``T`` fed to the model (``visual_length`` / ``args.visual_length``).
        prompt_text: Ordered list of class prompt strings matching model head dimension ``C``.
        gt: Frame-level binary ground-truth array aligned with repeated predictions.
        gtsegments: Per-video anomaly segment intervals for mAP (pickled ``.npy`` structure).
        gtlabels: Per-segment class strings for mAP (pickled ``.npy`` structure).
        device: ``\"cuda\"``, ``\"cpu\"``, or ``torch.device``.

    Returns:
        ``(ROC1, AP2, 0)`` — ROC-AUC for branch 1, Average Precision for branch 2, placeholder
        third value (average mAP not returned; see printed metrics).
    """
    model.to(device)
    model.eval()

    element_alignment_map_stack = []

    with torch.no_grad():
        ap1 = []
        ap2 = []
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0) # (1,L,T,D)->(1,T,D) splited segments of video
            length = item[2] # original length of the video

            length = int(length)
            len_cur = length
            # Short clips used to be [L, D] here; if dataset returns [1, L, D], do not add another dim.
            if len_cur < maxlen and visual.dim() == 2:
                visual = visual.unsqueeze(0) # (1,T,D)

            visual = visual.to(device)

            split_num = math.ceil(length / maxlen)
            lengths = torch.zeros(split_num)
            for j in range(split_num):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, anomaly_condifence, alignment_map = model(visual, padding_mask, prompt_text, lengths)
            anomaly_condifence = anomaly_condifence.reshape(anomaly_condifence.shape[0] * anomaly_condifence.shape[1], anomaly_condifence.shape[2]) # (L*T,1)
            alignment_map = alignment_map.reshape(alignment_map.shape[0] * alignment_map.shape[1], alignment_map.shape[2]) # (L*T，C)

            anomaly_prob1 = anomaly_condifence[0:len_cur].squeeze(-1)
            anomaly_prob2 = (1 - alignment_map[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))

            ap1.append(anomaly_prob1)
            ap2.append(anomaly_prob2)

            element_alignment_map = alignment_map[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_alignment_map = np.repeat(element_alignment_map, 16, 0)
            element_alignment_map_stack.append(element_alignment_map)
        
        ap1 = torch.cat(ap1, dim=0)
        ap2 = torch.cat(ap2, dim=0)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print(f"AUC1: {ROC1:.4f} AP1: {AP1:.4f}")
    print(f"AUC2: {ROC2:.4f} AP2: {AP2:.4f}")

    dmap, iou = dmAP(element_alignment_map_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1 , averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = list(label_map.values())
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load("H:\\BaiduNetdiskDownload\\model_xd.pth")
    model.load_state_dict(model_param)

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)