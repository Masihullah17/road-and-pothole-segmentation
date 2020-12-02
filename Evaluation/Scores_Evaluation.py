from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.misc
import random
import seg_utils as seg

import time
from cv2 import resize


def eval_image(gt_image, cnn_image):
    thresh = np.array(range(0, 256))/255.0

    # road_color = np.array([255,1,255])
    background_color = np.array([255,1,1])
    pothole_color = np.array([255,1,255])

    # road_color_gt = np.array([255,0,255])
    background_color_gt = np.array([255,0,0])
    pothole_color_gt = np.array([255,0,255])

    # Converting image into True or False
    # gt_road = np.all(gt_image == road_color_gt, axis=2)
    gt_bg = np.all(gt_image == background_color_gt, axis=2)
    gt_ph = np.all(gt_image == pothole_color_gt, axis=2)
    valid_gt = gt_bg + gt_ph

    cnn_road = np.all(cnn_image == pothole_color, axis=2)

    # Getting the False predictions and positive & negative predictions    
    FN, FP, posNum, negNum = seg.evalExp(gt_ph, cnn_road,
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def evaluate():
    image_dir = './c/kitti/without_93.34'

    eval_dict = {}

    thresh = np.array(range(0, 256))/255.0
    total_fp = np.zeros(thresh.shape)
    total_fn = np.zeros(thresh.shape)
    total_posnum = 0
    total_negnum = 0

    # Getting the names of images from the directory
    path = image_dir + "/predictionsFullSizeValKITTI"
    images = os.listdir(path)
    
    for i, name in enumerate(images):
        # Getting names of image, ground truth and predicted image
        image_file = path+name

        gt_file = image_dir + "/predictionsFullSizeValKITTIGt/" + name
        result = image_dir + "/predictionsFullSizeValKITTI/" + name
        

        print(gt_file)
        print(result)

        # Reading all three images
        #input_image = scp.misc.imread(image_file, mode='RGB')
        gt_image = scipy.misc.imread(gt_file, mode='RGB')
        output_im = scipy.misc.imread(result, mode='RGB')

        # gt_image = resize(gt_image, (960,540))

        # Getting the False predictions and positive & negative predictions
        FN, FP, posNum, negNum = eval_image(gt_image, output_im)

        # Summing up all values inorder to get final result from all images
        total_fp += FP
        total_fn += FN
        total_posnum += posNum
        total_negnum += negNum

    print(total_posnum)

    # Calculating the scores
    eval_dict = seg.pxEval_maximizeFMeasure(total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)
    
    # Printing stuff
    eval_list = []
    #print(eval_dict)

    eval_list.append(('MaxF1',
                      100*eval_dict['MaxF']))
    eval_list.append(('Average Precision',
                      100*eval_dict['AvgPrec']))
    eval_list.append(('Precision',
                      100*sum(eval_dict['precision'])/len(eval_dict['precision'])))
    eval_list.append(('Recall',
                      100*sum(eval_dict['recall'])/len(eval_dict['recall'])))
    eval_list.append(('FPR',
                      100*eval_dict['FPR_wp'][0]))
    eval_list.append(('FNR',
                      100*eval_dict['FNR_wp'][0]))
    

    return eval_list

if __name__ == '__main__':
    evals = evaluate()
    print("\nEvaluation Results : ")
    for i in evals:
        print(i)