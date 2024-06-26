import argparse
from nlgeval import NLGEval
import pandas as pd

def metrics(path, nlgeval):
    data = pd.read_json(path)
    gt= data['gt'].tolist()
    gt=[gt]
    gen = data['gen'].tolist()
    metrics = nlgeval.compute_metrics(ref_list=gt, hyp_list=gen)
    print(f'{path}: ', metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('eval-metric-nlp')
    # Here we use nargs='+' to indicate one or more arguments
    parser.add_argument('paths', type=str, nargs='+', help='paths to the generated text files')
    args = parser.parse_args()

    nlgeval_ = NLGEval()  # Initialize NLGEval once and use it for all files
    for path in args.paths:
        metrics(path, nlgeval_)




