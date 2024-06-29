import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from model import Former_Retrieval
import Utils.misc as misc
from Retrieval.retrieval_dataset import retrieval_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--vit_type', type=str, default="eva_vit")
    parser.add_argument('--task', type=str, default="retrieval")
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

    dataset_rt = retrieval_dataset(args.data_path, task='retrieval')
    dataset_zs = retrieval_dataset(args.data_path, task='zero-shot')
    dataloader_rt = torch.utils.data.DataLoader(dataset_rt, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_zs = torch.utils.data.DataLoader(dataset_zs, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model = Former_Retrieval.Former_RT(vit_type=args.vit_type)
    model = model.to(device)
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
    model.eval()
    ret_zs = model(dataloader_zs)
    ret_rt = model(dataloader_rt)
    # save the result
    np.save(os.path.join('./result_rt.npy'), ret_rt)
    np.save(os.path.join('./result_zs.npy'), ret_zs)
    if misc.is_main_process():
        _report_metrics(ret_rt, ret_zs, args)


@torch.no_grad()
def _report_metrics(ret, ret_zs, args):
    # with open(os.path.join(args.data_path, f'candidate.csv')) as f:
    #     candidate = pd.read_csv(f)
    # with open(os.path.join(args.data_path, f'I2IR_query.csv')) as f:
    #     IR_query = pd.read_csv(f)
    # with open(os.path.join(args.data_path, f'T2IR_query.csv')) as f:
    #     TR_query = pd.read_csv(f)

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
    def compute_precision_at_k(scores, classes, k=1):
        precisions = []

        for i, score_row in enumerate(scores):
            sample_class = classes[i]
            top_k_indices = np.argsort(score_row)[-k:][::-1]  # Get top k indices
            correct_retrievals = sum(classes[idx] == sample_class for idx in top_k_indices)
            precision = correct_retrievals / k
            precisions.append(precision)

        return np.nanmean(precisions)

    scores_i2t = ret["i2t"]
    scores_t2i = ret["t2i"]
    with open(os.path.join(args.data_path, 'df_200.csv')) as f:
        df = pd.read_csv(f)
    classes = df['Class']

    eval_result = {
        "i2t_r1": compute_precision_at_k(scores_i2t, classes, k=1),
        "i2t_r2": compute_precision_at_k(scores_i2t, classes, k=2),
        "i2t_r5": compute_precision_at_k(scores_i2t, classes, k=5),
        "i2t_r10": compute_precision_at_k(scores_i2t, classes, k=10),
        "i2t_r50": compute_precision_at_k(scores_i2t, classes, k=50),
        "t2i_r1": compute_precision_at_k(scores_t2i, classes, k=1),
        "t2i_r2": compute_precision_at_k(scores_t2i, classes, k=2),
        "t2i_r5": compute_precision_at_k(scores_t2i, classes, k=5),
        "t2i_r10": compute_precision_at_k(scores_t2i, classes, k=10),
        "t2i_r50": compute_precision_at_k(scores_t2i, classes, k=50),
    }

    print(eval_result)

    i2t= ret_zs["i2t"]
    true_labels = np.repeat(np.arange(5), 200)  # 假设有序的标签
    predicted_labels = np.argmax(i2t, axis=1) // 200
    accuracy = accuracy_score(true_labels, predicted_labels)

    # 计算每类的精确度、召回率和 F1-Score
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    return eval_result


if __name__ == "__main__":
    main()
