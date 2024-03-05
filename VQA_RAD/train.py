import argparse
import os
import sys

import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist
import utils
from VQA_RAD.Dataset import create_dataset
from VQA_RAD.model_VQA import MyVQAModel
from utils.pos_embed import interpolate_pos_embed


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    for i, (image, question, answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question, answer, train=True, alpha=alpha)

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
def evaluation(model, data_loader, device, config):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        topk_ids, topk_probs = model(image, question, answer_list, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())
            _, pred = topk_prob.max(dim=0)
            result.append({"qid": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
    return result


def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())

    #### Loading Dataset ####
    print('Creating vqa {} datasets'.format(args.dataset_use))
    datasets = create_dataset(args.dataset_use, args.dataset_path)[0]
    print('train dataset size: ', len(datasets[0]))
    print('test dataset size: ', len(datasets[1]))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            datasets, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(datasets)

    train_loader, test_loader = torch.utils.data.DataLoader(datasets, sampler_train, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True,collate_fn=datasets.collate_fn)

    #### Creating Model ####
    print("Creating model")
    model = MyVQAModel(text_encoder=args.text_encoder, text_decoder=args.text_decoder)
    model = model.to(device)
    # print(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.05)

    # TODO: load model from checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        # state_dict = checkpoint

        # # reshape positional embedding to accomodate for image resolution change
        # pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        # state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        if not args.evaluate:
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                if 'text_encoder' in key:
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < 6:
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num - 6)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    state_dict[decoder_key] = state_dict[key]

                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    start_epoch = 0
    print("\nStart training\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            utils.cosine_lr_schedule(optimizer, epoch, args.epochs, args.lr, args.min_lr)

            train(model, train_loader, optimizer, epoch, device, config)

        if args.evaluate:
            break

        if utils.is_main_process():

            save_obj = {
                'model': model_without_ddp.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'config': config,
                # 'epoch': epoch,
            }
            prefix = args.checkpoint.split('/')[-1].split('.')[0]
            if args.is_save_path and epoch > 20:
                torch.save(save_obj, os.path.join(args.output_dir, '%s_rad_%02d.pth' % (prefix, epoch)))
            vqa_result = evaluation(model, test_loader, device, config)
            json.dump(vqa_result, open(os.path.join(args.result_dir, '%s_vqa_result_%s.json' % (prefix, epoch)), 'w'))

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # compute acc
    res_file_path = '%s/result/%s_vqa_result_<epoch>.json' % (args.output_dir, prefix)
    compute_vqa_acc(answer_list_path=config[args.dataset_use]['test_file'][0], epoch=config['max_epoch'], res_file_path=res_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='rad', help='choose medical vqa dataset(rad, pathvqa, slake)')
    parser.add_argument('--dataset_path',help='path to the dataset')
    parser.add_argument('--is_save_path', default=False)
    parser.add_argument('--checkpoint', default='/mnt/sda/lpf/weights/output/V2/pretrain/std/med_pretrain_29.pth')
    parser.add_argument('--output_suffix', default='', help='output suffix, eg. ../rad_29_1')
    parser.add_argument('--output_dir', default='', help='the final output path, need not to assign')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--output_dir', default='./output_dir',type=str)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    args = parser.parse_args()


    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # set log, set console print info to file
    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)

    print("args: ", args)
    main(args)
