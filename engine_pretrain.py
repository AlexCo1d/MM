# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

from typing import Iterable

import torch

import Utils.misc as misc
import Utils.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        with torch.cuda.amp.autocast():
            loss = model(batch)
            loss_values = []
            loss_values_reduce = []
            loss_dict = {}
            for i in range(len(loss)):
                loss_values.append(loss[i].item())
            loss = torch.sum(torch.stack(loss), dim=0)
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            for i, l in enumerate(loss_values):
                loss_dict.update({'loss' + str(i + 1): l})
            metric_logger.update(**loss_dict)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            for l in loss_values:
                loss_values_reduce.append(misc.all_reduce_mean(l))

            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)

                for i, l in enumerate(loss_values_reduce):
                    log_writer.add_scalar('train_loss' + str(i + 1), l, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
