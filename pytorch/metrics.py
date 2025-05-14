# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Training utilities."""

import torch
import torch.nn.functional as F
import numpy as np
import logging


def update_pretrain_metrics_train(contrast_loss, contrast_acc, contrast_entropy,
                                  loss, logits_con, labels_con):
    """Updated pretraining metrics."""
    contrast_loss.update_state(loss.item())

    contrast_acc_val = torch.argmax(labels_con, dim=1) == torch.argmax(logits_con, dim=1)
    contrast_acc_val = contrast_acc_val.float().mean().item()
    contrast_acc.update_state(contrast_acc_val)

    prob_con = F.softmax(logits_con, dim=1)
    entropy_con = -torch.sum(prob_con * torch.log(prob_con + 1e-8), dim=1).mean().item()
    contrast_entropy.update_state(entropy_con)


def update_pretrain_metrics_eval(contrast_loss_metric,
                                 contrastive_top_1_accuracy_metric,
                                 contrastive_top_5_accuracy_metric,
                                 contrast_loss, logits_con, labels_con):
    contrast_loss_metric.update_state(contrast_loss.item())

    pred_top1 = torch.argmax(logits_con, dim=1)
    true_top1 = torch.argmax(labels_con, dim=1)
    contrastive_top_1_accuracy_metric.update_state(true_top1, pred_top1)

    contrastive_top_5_accuracy_metric.update_state(labels_con, logits_con)


def update_finetune_metrics_train(supervised_loss_metric, supervised_acc_metric,
                                  loss, labels, logits):
    supervised_loss_metric.update_state(loss.item())

    pred_labels = torch.argmax(logits, dim=1)
    true_labels = torch.argmax(labels, dim=1)
    acc = (pred_labels == true_labels).float().mean().item()
    supervised_acc_metric.update_state(acc)


def update_finetune_metrics_eval(label_top_1_accuracy_metrics,
                                 label_top_5_accuracy_metrics, outputs, labels):
    pred_top1 = torch.argmax(outputs, dim=1)
    true_top1 = torch.argmax(labels, dim=1)
    label_top_1_accuracy_metrics.update_state(true_top1, pred_top1)

    label_top_5_accuracy_metrics.update_state(labels, outputs)


def _float_metric_value(metric):
    """Gets the value of a float-value metric object."""
    return float(metric.result())


def log_and_write_metrics_to_summary(all_metrics, global_step):
    for metric in all_metrics:
        metric_value = _float_metric_value(metric)
        logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
        # In PyTorch, you typically log via tensorboardX or torch.utils.tensorboard
        # Here’s a placeholder for that:
        # writer.add_scalar(metric.name, metric_value, global_step)
