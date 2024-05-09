import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from model import Former_Retrieval
import Utils.misc as misc
from Retrieval.retrieval_dataset import retrieval_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args()
    if args.distributed:
        misc.init_distributed_mode(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = retrieval_dataset(args.data_path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = Former_Retrieval.Former_RT()
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model.eval()
    ret = model(dataloader)
    if misc.is_main_process():
        _report_metrics(ret,args)


@torch.no_grad()
def _report_metrics(ret, args):
    scores_i2t = ret["i2t"] # num_image x num_candidates
    scores_t2i = ret["t2i"] # num_text x num_candidates
    scores_i2i = ret["i2i"] # num_image x num_candidates
    with open(os.path.join(args.data_path, f'candidate.csv')) as f:
        candidate = pd.read_csv(f)
    with open(os.path.join(args.data_path, f'I2IR_query.csv')) as f:
        IR_query = pd.read_csv(f)
    with open(os.path.join(args.data_path, f'T2IR_query.csv')) as f:
        TR_query = pd.read_csv(f)

    # # Images->Text
    # ranks = np.zeros(scores_i2t.shape[0])
    # for index, score in enumerate(scores_i2t):
    #     inds = np.argsort(score)[::-1]
    #     # Score
    #     rank = 1e20
    #     for i in img2txt[index]:
    #         tmp = np.where(inds == i)[0][0]
    #         if tmp < rank:
    #             rank = tmp
    #     ranks[index] = rank
    #
    # # Compute metrics
    # tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    #
    # # Text->Images
    # ranks = np.zeros(scores_t2i.shape[0])
    #
    # for index, score in enumerate(scores_t2i):
    #     inds = np.argsort(score)[::-1]
    #     ranks[index] = np.where(inds == txt2img[index])[0][0]
    #
    # # Compute metrics
    # ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    #
    # tr_mean = (tr1 + tr5 + tr10) / 3
    # ir_mean = (ir1 + ir5 + ir10) / 3
    # r_mean = (tr_mean + ir_mean) / 2
    #
    # agg_metrics = (tr1 + tr5 + tr10) / 3
    #
    # eval_result = {
    #     "txt_r1": tr1,
    #     "txt_r5": tr5,
    #     "txt_r10": tr10,
    #     "txt_r_mean": tr_mean,
    #     "img_r1": ir1,
    #     "img_r5": ir5,
    #     "img_r10": ir10,
    #     "img_r_mean": ir_mean,
    #     "r_mean": r_mean,
    #     "agg_metrics": agg_metrics,
    # }
    # with open(
    #         os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    # ) as f:
    #     f.write(json.dumps(eval_result) + "\n")
    def compute_precision_at_k(scores, query, candidate, k=1):
        precisions = []
        for idx, score in enumerate(scores):
            top_k_indices = np.argsort(score)[-k:][::-1]  # 从高到低
            matched = 0
            for top_idx in top_k_indices:
                # 检查候选项中的one-hot编码是否与查询匹配
                if candidate.iloc[top_idx][query['variable'].iloc[idx]] == 1:
                    matched += 1
            precision = matched / k
            precisions.append(precision)
        return np.mean(precisions)

    eval_result = {
        "t2i_r1": compute_precision_at_k(scores_t2i, IR_query, candidate, k=1),
        "t2i_r5": compute_precision_at_k(scores_t2i, IR_query, candidate, k=5),
        "t2i_r10": compute_precision_at_k(scores_t2i, IR_query, candidate, k=10),
        "i2i_r1": compute_precision_at_k(scores_i2i, TR_query, candidate, k=1),
        "i2i_r5": compute_precision_at_k(scores_i2i, TR_query, candidate, k=5),
        "i2i_r10": compute_precision_at_k(scores_i2i, TR_query, candidate, k=10),
    }

    print(eval_result)
    return eval_result


if __name__ == "__main__":
    main()
