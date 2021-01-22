import json, pickle, tarfile, os, argparse
import numpy as np
import xgboost

from sklearn.metrics import roc_auc_score

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model-dir', type=str, default=None)
    parser.add_argument('--input-data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--output-file', type=str, default='evaluation.json')
    
    args, _ = parser.parse_known_args()
    print(f'Received arguments {args}')
    return args


if __name__ == "__main__":
    args = arg_parse()
    model_path = os.path.join(args.input_model_dir, 'model.tar.gz')
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    model = pickle.load(open("xgboost-model", "rb"))

    test_csv_path = os.path.join(args.input_data_dir,'test.csv')
    test = np.loadtxt(test_csv_path, delimiter=',')
    test_x = xgboost.DMatrix(test[:,1:])
    pred_y = model.predict(test_x)
    print(pred_y)
    answer_y = test[:,0]
    
    
    auc = roc_auc_score(answer_y,pred_y)
    
    print(f'auc: {auc}')
    
    report_dict = {
        "classification_metrics": {
            "auc": {
                "value": auc,
            },
        },
    }

    eval_result_path = os.path.join(args.output_dir, args.output_file)
    with open(eval_result_path, "w") as f:
        f.write(json.dumps(report_dict))
    exit()