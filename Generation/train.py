"""
VQA task training script
alexyhzou
"""

import argparse
import os
import sys

import transformers

from Generation.metric_eval import compute_metrics
from VQA.pmc_eval import evaluation_pmc
from model.Former_Caption_Llama import Former_Llama_Cap

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
        loss = model(b, dataloader=data_loader)
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
        t_model = model.module
        t_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 10

    result = []

    for n, b in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        text_output = t_model.predict_answers(b, dataloader=data_loader)
        for idx, answer in enumerate(text_output):
            # 构造结果字典
            result_dict = {
                "gen": text_output[idx],  # 预测的答案
                "gt": b['text_output'][idx],  # 实际答案
            }
            result.append(result_dict)
    t_model.train()
    return result



def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())

    #### Loading Dataset ####
    print('Creating vqa {} datasets'.format(args.dataset_use))
    train_dataset, test_dataset, _ = create_dataset(args)
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
        # print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler_train, batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler_test, batch_size=args.eval_batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    #### Creating Model ####
    print("Creating model")
    model = Former_Llama_Cap(img_size=args.img_size, llm_model=args.LLM_path, vit_path=args.vit_path if args.checkpoint is None else '',
                             freeze_vit=args.freeze_vit, is_lora=args.is_lora)
    model = model.to(device)
    # print(model)

    eff_batch_size = args.batch_size * misc.get_world_size()

    optimizer = transformers.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.02, betas=(0.9, 0.98))

    start_epoch = 0

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        if args.deepspeed:
            import deepspeed
            model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, config=args.deepspeed_config,
                                                          model_parameters=model.parameters(),
                                                          optimizer=optimizer)

        model_without_ddp = model.module

    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
        if 'optimizer' in checkpoint and args.load_optim:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if args.start_epoch != -1:
            start_epoch = args.start_epoch

    start_time = time.time()
    if args.evaluate:
        print("\nStart evaluation\n")
        if args.dataset_use != 'pmcvqa':
            gen_result = evaluation(model, test_loader, device, args)
            json.dump(gen_result,
                      open(os.path.join(args.result_dir, 'gen_result_%s.json' % (args.dataset_use)), 'w'))
            metrics = compute_metrics(gen_result, args=args, dataloader=test_loader, epoch=checkpoint['epoch'])
            print(f'{args.dataset_use} acc: {metrics}')
        else:
            evaluation_pmc(model, test_loader, device, args)
    else:
        print("\nStart training\n")
        # misc.set_requires_grad_llm(model_without_ddp, False)
        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            utils.cosine_lr_schedule(optimizer, epoch, args.epochs, args.lr, args.min_lr)
            # if epoch == args.warmup_epochs:
            #     misc.set_requires_grad_llm(model_without_ddp, True)
            train(model, train_loader, optimizer, epoch, device, args)
            ###
            if epoch >= args.epochs - 5 or epoch <= args.warmup_epochs:
                train(model, test_loader, optimizer, epoch, device, args)

            if utils.is_main_process():

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                prefix = args.checkpoint.split('/')[-1].split('.')[0]
                # for evaluation and output the result
                if args.output_dir and epoch >= 10 and (epoch % args.eval_freq == 0 or epoch >= args.epochs - 5):
                    torch.cuda.empty_cache()
                    torch.save(save_obj,
                               os.path.join(args.output_dir, '%s_%s_%02d.pth' % (prefix, args.dataset_use, epoch)))
                    gen_result = evaluation(model, test_loader, device, args)
                    json.dump(gen_result,
                              open(os.path.join(args.result_dir, '%s_vqa_result_%s.json' % (prefix, epoch)), 'w'))
                    metrics = compute_metrics(gen_result, args=args, dataloader=test_loader, epoch=epoch)
                    print({'results:': metrics})
                    json.dump({'results:': metrics},
                              open(os.path.join(args.result_dir, 'vqa_metric.json'), 'a'))
                else:
                    torch.save(save_obj, os.path.join(args.output_dir, 'last_epoch_weight.pth'))

            if args.distributed:
                dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='radvqa', help='choose medical vqa dataset(radvqa, pathvqa, slake)')
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--load_optim', action='store_true')
    parser.set_defaults(load_optim=False)
    parser.add_argument('--vit_path', default='',
                        help='path for loading pretrained ViT model')
    parser.add_argument('--LLM_path', default='', type=str, help='path for loading pretrained LLM model')
    parser.add_argument('--classifier_vqa', action='store_true')
    parser.set_defaults(classifier_vqa=False)
    parser.add_argument('--is_lora', action='store_true')
    parser.set_defaults(is_lora=False)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--freeze_vit', action='store_true')
    parser.set_defaults(freeze_vit=False)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=5, type=int)
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
    parser.add_argument('--start_epoch', default=-1, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=2e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # deepspeed
    parser.add_argument('--deepspeed', action='store_true', help='use DeepSpeed for distributed training')
    parser.set_defaults(deepspeed=False)
    parser.add_argument('--deepspeed_config', type=str, default='./ds_config.json', help='DeepSpeed configuration file')

    args = parser.parse_args()

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # set log, set console print info to file
    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)

    print("args: ", args)
    main(args)
