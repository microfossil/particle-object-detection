"""
Plot the evaluation metrics from the slurm output
"""
import re
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt

def get_eval_results(filename):
    all = []
    current = OrderedDict()
    with open(filename, 'r') as file:
        for line in file.readlines():
            parts = re.split('[:\/]', line)
            if len(parts) > 3:
                if parts[0] == "INFO" and parts[1] == "tensorflow" and parts[2].startswith("\t+ DetectionBoxes"):
                    current[parts[3]] = float(parts[4][:-1])
                    if parts[3] == "AR@100 (large)":
                        all.append(current)
                        current = OrderedDict()
    df = pd.DataFrame(all)
    return df

def plot_tf2_objdet_evaluation(filename):
    df = get_eval_results(filename)
    df.plot(cmap='jet', title="test")
    plt.ylim(0, 1.2)
    plt.xlim(0, len(all) * 1.5)
    plt.grid()
    for key in all[-1].keys():
        if all[-1][key] > 0:
            plt.text(len(all) * 1.05, all[-1][key], key)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_tf2_objdet_evaluation("/media/mar76c/scratch1/visage-ml-tf2objdet/launch/COTS_Heron_v4/hpc_20210611_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8_1024/slurm-53161435.out")
    # plot_tf2_objdet_evaluation("/media/mar76c/scratch1/visage-ml-tf2objdet/launch/COTS_Heron_v4/hpc_20210611_efficientdet_d0_coco17_tpu-32_1024/slurm-53161433.out")
    from glob import glob
    import os

    slurms = sorted(glob(os.path.join("/run/user/1112115/gvfs/smb-share:server=bracewellscratch1.csiro.au,share=scratch1/mar76c/visage-ml-tf2objdet/launch/COTS_Heron_v4", "**/*.out")))

    di = []
    dimax = []
    for slurm in slurms:
        df = get_eval_results(slurm)
        if len(df) > 0:
            name = os.path.basename(os.path.dirname(slurm))
            d = {'model': name}
            d.update(df.iloc[-1,:])
            print(d)
            di.append(d)
            d = {'model': name}
            d.update(df.max())
            dimax.append(d)
    result = pd.DataFrame(di)
    result.to_csv("final.csv")
    print(result)
    result = pd.DataFrame(dimax)
    result.to_csv("max.csv")
    print(result)




"""
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP: 0.191834
I0515 16:32:27.606204 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Precision/mAP: 0.191834
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP@.50IOU: 0.365337
I0515 16:32:27.608245 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Precision/mAP@.50IOU: 0.365337
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP@.75IOU: 0.164852
I0515 16:32:27.610277 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Precision/mAP@.75IOU: 0.164852
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (small): -1.000000
I0515 16:32:27.612145 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Precision/mAP (small): -1.000000
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (medium): 0.068334
I0515 16:32:27.613875 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Precision/mAP (medium): 0.068334
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (large): 0.235055
I0515 16:32:27.615824 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Precision/mAP (large): 0.235055
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@1: 0.336823
I0515 16:32:27.617710 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Recall/AR@1: 0.336823
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@10: 0.450361
I0515 16:32:27.619485 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Recall/AR@10: 0.450361
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100: 0.491516
I0515 16:32:27.621570 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Recall/AR@100: 0.491516
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (small): -1.000000
I0515 16:32:27.623480 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Recall/AR@100 (small): -1.000000
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (medium): 0.314159
I0515 16:32:27.625007 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Recall/AR@100 (medium): 0.314159
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (large): 0.536961
I0515 16:32:27.627150 46912496467904 model_lib_v2.py:992] 	+ DetectionBoxes_Recall/AR@100 (large): 0.536961
INFO:tensorflow:	+ Loss/RPNLoss/localization_loss: 0.002336
I0515 16:32:27.629041 46912496467904 model_lib_v2.py:992] 	+ Loss/RPNLoss/localization_loss: 0.002336
INFO:tensorflow:	+ Loss/RPNLoss/objectness_loss: 0.003794
I0515 16:32:27.630482 46912496467904 model_lib_v2.py:992] 	+ Loss/RPNLoss/objectness_loss: 0.003794
INFO:tensorflow:	+ Loss/BoxClassifierLoss/localization_loss: 0.009455
I0515 16:32:27.631928 46912496467904 model_lib_v2.py:992] 	+ Loss/BoxClassifierLoss/localization_loss: 0.009455
INFO:tensorflow:	+ Loss/BoxClassifierLoss/classification_loss: 0.022028
I0515 16:32:27.633294 46912496467904 model_lib_v2.py:992] 	+ Loss/BoxClassifierLoss/classification_loss: 0.022028
INFO:tensorflow:	+ Loss/regularization_loss: 0.000000
I0515 16:32:27.634685 46912496467904 model_lib_v2.py:992] 	+ Loss/regularization_loss: 0.000000
INFO:tensorflow:	+ Loss/total_loss: 0.037613
I0515 16:32:27.636078 46912496467904 model_lib_v2.py:992] 	+ Loss/total_loss: 0.037613
"""