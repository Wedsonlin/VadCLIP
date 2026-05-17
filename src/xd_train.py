import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
from argparse import ArgumentParser

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_batch_label_vector
from utils.logger import Logger
import xd_option

def BCE(
    anomaly_confidence: torch.Tensor,
    instance_label: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Video-level binary classification error (BCE) after Top-K MIL on sigmoid frame scores (coarse branch).

    Per clip, averages the top ``floor(T_i/16)+1`` frame scores on indices ``[0, lengths[i])``,
    then compares to binary targets derived from ``instance_labels``.

    Args:
        anomaly_confidences: Per-frame anomaly probability map after sigmoid; shape
            ``(B, T)`` or ``(B, T, 1)``.
        instance_labels: Multi-hot clip labels from ``get_batch_label``; shape ``(B, C)``.
            Column ``0`` is the normal class (XD): targets become ``1 - instance_labels[:, 0]``,
            shape ``(B,)``, values ``0`` or ``1``.
        lengths: Valid frame count per clip (padding excluded); shape ``(B,)``, integer dtype,
            each entry ``<= T``.
        device: Device for internal temporaries (e.g. ``"cuda"``, ``torch.device("cuda")``).

    Returns:
        Scalar tensor — mean binary cross-entropy over batch (both operands implicitly ``(B,)``).
    """
    video_level_scores = torch.zeros(0).to(device)
    binary_label = (1 - instance_label[:, 0]).reshape(-1).to(device) # (B,)
    lengths = lengths.to(device) # (B,)
    anomaly_condifence = anomaly_confidence.squeeze(-1) # (B,T)

    for i in range(anomaly_condifence.shape[0]):
        topk_scores, _ = torch.topk(
            anomaly_condifence[i, 0 : lengths[i]],
            k=int(lengths[i] / 16 + 1),
            largest=True,
        )
        score = torch.mean(topk_scores, dim=-1, keepdim=True) # (1,)
        video_level_scores = torch.cat([video_level_scores, score], dim=0) # (B,)

    loss = F.binary_cross_entropy(video_level_scores, binary_label)
    return loss

def NCE(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """
    MIL-style Noise Contrastive Estimation (NCE): Top-K temporal pooling per clip, then soft-target cross-entropy.

    Args:
        logits: Frame-class alignment scores (before softmax), shape ``(B, T, C)``.
        labels: Multi-hot clip labels, shape ``(B, C)``; row-normalized internally to a probability
            distribution before ``cross_entropy``.
        lengths: Valid frames per sample (padding excluded), shape ``(B,)``, integer dtype; each
            entry ``<= T``. Must be on the same device as ``logits`` for slicing.

    Returns:
        Scalar tensor (``float`` dtype): mean over the batch of ``F.cross_entropy`` between
        softmax logits over classes ``C`` and row-normalized ``labels``.
    """
    instance_logits = []
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        topk_score, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0) # (K, C)
        instance_logits.append(torch.mean(topk_score, dim=0, keepdim=True))

    instance_logits = torch.cat(instance_logits, dim=0) # (B,C)
    loss = F.cross_entropy(instance_logits, labels, reduction='mean')
    return loss

def CTS(text_features: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """
    Contrastive term pushing abnormal-class text embeddings away from the normal class.

    Assumes ``text_features`` is ``(C, D)`` with row ``0`` the normal prompt and rows ``1:C``
    the anomaly prompts. Rows are L2-normalized; loss is the mean absolute cosine similarity
    between normal and each abnormal embedding.

    Args:
        text_features: Class text embeddings from ``encode_textprompt``, shape ``(C, D)``.
        device: Target device for the computation.

    Returns:
        Scalar tensor — mean of ``|⟨û, â_j⟩|`` over abnormal indices ``j`` (unit vectors).
    """
    text_features = F.normalize(text_features, p=2, dim=-1).to(device)
    normal_text_feature = text_features[0]
    abnormal_text_features = text_features[1:]
    loss = torch.mean(torch.abs(abnormal_text_features @ normal_text_feature))

    return loss

def train(
    model: CLIPVAD, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    args: ArgumentParser, 
    label_map: dict[str, str], 
    device: torch.device | str, 
    logger: Logger | None = None
    ) -> None:
    model.to(device)

    gt = np.load(args.gt_path) # frame-leval binary ground truth (whether the frame is anomalous or not)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True) # segment-level ground truth (start and end frame of the anomalous segment)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True) # video-level ground truth (label of the anomalous video)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate) # more suitable for short training epochs
    label_classes = list(label_map.values())
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for epoch in range(args.max_epoch):
        model.train()
        step = 0
        for item in train_loader:
            video_clip_feature, text_labels, feature_length = item
            video_clip_feature = video_clip_feature.to(device)
            feature_length = feature_length.to(device)
            instance_label = get_batch_label_vector(text_labels, label_map).to(device)

            text_feature, anomaly_confidence, alignment_map = model(video_clip_feature, None, label_classes, feature_length) 

            bce_loss = BCE(anomaly_confidence, instance_label, feature_length, device) 

            nce_loss = NCE(alignment_map, instance_label, feature_length, device)

            cts_loss = CTS(text_feature, device)
            loss = bce_loss + nce_loss + cts_loss * 1e-4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print('epoch: ', epoch+1, '| step: ', step, '| bce_loss: ', f'{bce_loss.item():.4f}', '| nce_loss: ', f'{nce_loss.item():.4f}', '| cts_loss: ', f'{cts_loss.item():.4f}')
                if logger is not None:
                    logger.log_train({
                        "bce_loss": bce_loss.item(),
                        "nce_loss": nce_loss.item(),
                        "cts_loss": cts_loss.item(),
                    })
        
        scheduler.step()
        AUC, AP, mAP = test(model, test_loader, args.visual_length, label_classes, gt, gtsegments, gtlabels, device)
        if logger is not None:
            logger.log_eval({
                "AUC": AUC,
                "ap": AP,
                "mAP": mAP,
            })
        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)

        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    logger = Logger(project="WSAVD", name="xd-violence_train-test")
    train(model, train_loader, test_loader, args, label_map, device, logger)