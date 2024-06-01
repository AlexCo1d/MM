"""
VQA task training script
alexyhzou
"""

import argparse
import os
import sys

from model.Former_Llama import Former_Llama

sys.path.append("..")
import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist
import utils
import Utils.misc as misc
import Utils.lr_sched as lr_sched
from Dataset import create_dataset
from model.Former_T5 import Former_T5
from vqaTools.vqaEvaluate import compute_vqa_acc


def train(model, data_loader, optimizer, epoch, device, args):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10
    for i, b in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args)
        loss = model(b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, args):
    """
    :param model: VQA model
    :param data_loader: test_loader
    :param device: device
    :param args: arguments
    :return: a dict that contains the result of the evaluation {question, pred, answer, answer_type}
    """
    # test
    if args.distributed:
        model = model.module
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 10

    result = []

    for n, b in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        text_output = model.predict_answers(b)
        for idx, answer in enumerate(text_output):
            # 构造结果字典
            result_dict = {
                'image_name':b['image_name'][idx], #获取图片名
                "question": b['text_input'][idx],  # 当前问题
                "pred": text_output[idx],  # 预测的答案
                "answer": b['text_output'][idx],  # 实际答案
                "answer_type": b['answer_type'][idx]  # 答案类型
            }
            result.append(result_dict)
    return result

def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())

    #### Loading Dataset ####
    print('Creating vqa {} datasets'.format(args.dataset_use))
    train_dataset, test_dataset = create_dataset(args.dataset_use, args.dataset_path)
    print('train dataset size: ', len(train_dataset))
    print('test dataset size: ', len(test_dataset))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_test = torch.utils.data.RandomSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler_train, batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler_test, batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    #### Creating Model ####
    print("Creating model")
    model = Former_Llama(llm_model=args.LLM_path, vit_path=args.vit_path if args.checkpoint is None else '', freeze_vit=True)
    model = model.to(device)
    # print(model)

    eff_batch_size = args.batch_size * misc.get_world_size()


    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.05)


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    start_epoch = 0
    acc_list = []
    print("\nStart training\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.evaluate:
            vqa_result = evaluation(model, test_loader, device, args)
            json.dump(vqa_result,
                      open(os.path.join(args.result_dir, 'vqa_result_%s.json' % (args.dataset_use)), 'w'))
            acc = compute_vqa_acc(vqa_result, epoch, args=args)
            print(f'{args.dataset_use} acc: {acc}')
            break

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            utils.cosine_lr_schedule(optimizer, epoch, args.epochs, args.lr, args.min_lr)

            train(model, train_loader, optimizer, epoch, device, args)

        if utils.is_main_process():

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'epoch': epoch,
            }
            prefix = args.checkpoint.split('/')[-1].split('.')[0]
            # for evaluation and output the result
            if args.output_dir and epoch >= 20 and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
                torch.save(save_obj, os.path.join(args.output_dir, '%s_rad_%02d.pth' % (prefix, epoch)))
                vqa_result = evaluation(model, test_loader, device, args)
                json.dump(vqa_result,
                          open(os.path.join(args.result_dir, '%s_vqa_result_%s.json' % (prefix, epoch)), 'w'))
                acc = compute_vqa_acc(vqa_result, epoch, args=args)
                acc_list.append({'epoch:':epoch,'acc:':acc})
            torch.save(save_obj, os.path.join(args.output_dir, 'last_epoch_weight.pth'))

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # print the epoch with best acc in acc_list
    json.dump(acc_list,
              open(os.path.join(args.result_dir, 'vqa_metric.json'), 'w'))
    # print("Best acc: ", max(acc_list, key=lambda x: x[1][0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='radvqa', help='choose medical vqa dataset(radvqa, pathvqa, slake)')
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--vit_path', default='',
                        help='path for loading pretrained ViT model')
    parser.add_argument('--LLM_path', default='', type=str, help='path for loading pretrained LLM model')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.set_defaults(evaluate=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--output_dir', default='./output_dir', type=str)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    args = parser.parse_args()

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # set log, set console print info to file
    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)

    print("args: ", args)
    main(args)
