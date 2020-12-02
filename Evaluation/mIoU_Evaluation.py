from argparse import ArgumentParser
from PIL import Image
import os
import glob
import time
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

res = 1080

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gts', default="./c/idd/1/predictionsFullSizeValIDDGt/")
    parser.add_argument('--preds', default="./c/idd/1/predictionsFullSizeValIDD/")
    parser.add_argument('--prefix', default="_gtFine_polygons.png")
    parser.add_argument('--res', type=int, default=720)
    parser.add_argument('--num-workers', type=int, default=8)
    
    args = parser.parse_args()

    return args


def add_to_confusion_matrix(gt, pred, mat):
	if (pred.shape[0] != gt.shape[0]):
		print("Image widths of " + pred + " and " + gt + " are not equal.")
	if (pred.shape[1] != gt.shape[1]):
		print("Image heights of " + pred + " and " + gt + " are not equal.")
	if ( len(pred.shape) != 2 ):
		print("Predicted image has multiple channels.")
	W  = pred.shape[0]
	H = pred.shape[1]
	P = H*W
	
	for h in range(gt.shape[0]):
		for w in range(gt.shape[1]):
			gtr = None
			prr = None

			# if list(gt[h][w]) == [255,1,1] or list(gt[h][w]) == [255,0,0]:
			# 	gtr = 1
			if list(gt[h][w]) == [255,1,255] or list(gt[h][w]) == [255,0,255]:
				gtr = 1
			else:
				gtr = 0

			# if list(pred[h][w]) == [255,1,1] or list(gt[h][w]) == [255,0,0]:
			# 	prr = 1
			if list(pred[h][w]) == [255,1,255] or list(pred[h][w]) == [255,0,255]:
				prr = 1
			else:
				prr = 0
			
			mat[gtr, prr] += 1

	return mat

def eval_ious(mat):
    ious = np.zeros(3)
    for l in range(2):
        tp = np.longlong(mat[l,l])
        fn = np.longlong(mat[l,:].sum()) - tp

        notIgnored = [i for i in range(2) if not i==l]
        fp = np.longlong(mat[notIgnored,l].sum())
        denom = (tp + fp + fn)
        if denom == 0:
            print('error: denom is 0')

        ious[l] =  float(tp) / denom

    return ious[:-1]

def process_pred_gt_pair(pair):
    global res
    W,H = 1920, 1080
    if res == 720:
        W,H = 1280, 720
    if res == 480:
        W,H = 858, 480
    if res == 240:
        W,H = 426, 240



    gt, pred = pair
    confusion_matrix = np.zeros(shape=(2,2),dtype=np.ulonglong)

    gt = Image.open(gt)
    if gt.size != (W, H):
        gt = gt.resize((W, H), resample = Image.NEAREST)
    gt = np.array(gt)

    pred = Image.open(pred)
    if pred.size != (W, H):
        pred = pred.resize((W, H), resample=Image.NEAREST)
    pred = np.array(pred)

    add_to_confusion_matrix(gt, pred, confusion_matrix)

    return confusion_matrix

import tqdm


def main(args):
    global res
    res = args.res
    confusion_matrix    = np.zeros(shape=(2,2),dtype=np.ulonglong)
    gts_folders         = glob.glob(args.gts + '/*')
    pred_folders        = glob.glob(args.preds + '/*')

    pairs   = []
    print(args.gts)
    for g in gts_folders:
        name = g.split("\\")[-1]
        print(name)
        pairs.append((args.gts + name ,args.preds + name ))
        if len(pairs) == 30:
            break  

    pool = Pool(args.num_workers)

    results = list(tqdm.tqdm(pool.imap(process_pred_gt_pair, pairs), total=len(pairs)))
    pool.close()
    pool.join()

    for i in range(len(results)):
        confusion_matrix += results[i]

    print(confusion_matrix)

    os.makedirs('eval_results', exist_ok=True)

    np.save(f'eval_results/cm_{res}',confusion_matrix)

    ious = eval_ious(confusion_matrix)
    np.save(f'eval_results/ious_{res}', np.array(ious))

    print(f'mIoU:\t\t\t\t{ious.mean()*100}')
        
if __name__ == '__main__':
    args = get_args()
    main(args)