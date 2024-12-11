# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from itertools import filterfalse

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from .mask_model import MaskPLS


class SemLoss(torch.nn.Module):
    def __init__(
        self,
        ignore_class: int,
        sem_distil: bool,
        loss_scales: dict,
        input_mode: MaskPLS.DecoderInputMode = MaskPLS.DecoderInputMode.GRID,
    ) -> None:
        super().__init__()
        self.ignore_class = ignore_class
        self.sem_distil = sem_distil
        self.weight_dict = loss_scales
        self.input_mode = input_mode

        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_class)

    def forward(self, outputs, targets) -> dict:
        padding = outputs["padding"]
        bb_sem_logits = outputs["bb_sem_logits"]
        bb_sem_embed_logits = outputs["bb_sem_embed_logits"]

        sem_labels = [torch.from_numpy(i).type(torch.long).to(padding.device) for i in targets["semantic_labels"]]
        sem_labels = torch.cat(sem_labels)

        if self.input_mode == MaskPLS.DecoderInputMode.GRID:
            # If the input to the loss function (which is the same as the input/output from the decoder) is the grid centers,
            # (i.e. not the original xyz coordinates), we need to convert the targets to the grid centers as well.
            input_vdbtensor = targets["vdbtensor"]

            # map target semantic labels to the grid
            points = targets["xyz"]

            # get mapping of the coordinates to the grid for feature mapping
            coord_ijks = input_vdbtensor.grid.world_to_grid(points).round().int()
            inv_idx = input_vdbtensor.grid.ijk_to_inv_index(coord_ijks, cumulative=True)
            sem_labels = sem_labels[inv_idx.jdata]

        sem_targets = sem_labels

        outputs = []
        batch_size = bb_sem_logits.shape[0]
        for b in range(batch_size):
            bb_sem_logit = bb_sem_logits[b][~padding[b]]

            if not self.training:
                bb_sem_logit = bb_sem_logit[targets["inv_ind"][b]]

            outputs.append(bb_sem_logit)

        outputs = torch.cat(outputs)

        loss_sem_ce = self.cross_entropy(outputs, sem_targets)
        loss_sem_lov = self.lovasz_softmax(F.softmax(outputs, dim=1), sem_targets, ignore=self.ignore_class)

        loss_dict = dict()

        # we check if loss is enabled for cleaner coding
        loss_dict["loss_backbone_sem_ce"] = self.weight_dict.get("loss_backbone_sem_ce", 0.0) * loss_sem_ce

        # we check if loss is enabled for cleaner coding
        loss_dict["loss_backbone_sem_lov"] = self.weight_dict.get("loss_backbone_sem_lov", 0.0) * loss_sem_lov

        if self.sem_distil and "loss_backbone_sem_distil" in self.weight_dict and targets["sem_embed"][0] is not None:
            input1_list = []
            input2_list = []

            for b in range(batch_size):
                sem_embed_logits = bb_sem_embed_logits[b][~padding[b]]

                masks_ids = targets["masks_ids"][b]
                num_masks = len(masks_ids)

                instance_ids = [np.unique(targets["ins_labels"][b][m_ids.cpu().numpy()]) for m_ids in masks_ids]
                for instance_id in instance_ids:
                    assert len(instance_id) == 1

                sem_embed = targets["sem_embed"][b]
                sem_embed_ins = list(sem_embed["ins"])

                embeds_ids = [sem_embed_ins.index(instance_id[0]) for instance_id in instance_ids]

                sem_embeds = sem_embed["embeds"][embeds_ids]

                for i in range(num_masks):
                    input1_list.append(sem_embed_logits[masks_ids[i]])
                    input2_list.append(sem_embeds[i][None, ...].repeat(len(masks_ids[i]), 1))

            input1 = torch.cat(input1_list)
            input2 = torch.cat(input2_list)

            target = torch.ones(len(input2)).to(input2.device)
            loss_distil = F.cosine_embedding_loss(input1, input2, target=target)

            loss_dict["loss_backbone_sem_distil"] = loss_distil * self.weight_dict["loss_backbone_sem_distil"]

        return loss_dict

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax(self, probas, labels, classes="present", ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss

    def lovasz_softmax_flat(self, probas, labels, classes="present"):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.0
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (torch.autograd.Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, torch.autograd.Variable(self.lovasz_grad(fg_sorted))))
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        # Probabilities from SparseTensor.features already flattened
        N, C = probas.size()
        probas = probas.contiguous().view(-1, C)
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = labels != ignore
        vprobas = probas[torch.nonzero(valid).squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = filterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == "raise":
                raise ValueError("Empty mean")
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n
