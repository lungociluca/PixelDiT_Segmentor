import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import scipy.io as sio
import argparse
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, sys.path[0]+"/../..")


voc_path = "../../datasets/VOCdevkit/VOC2012"

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

embs = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def compare(start,step,TP,P,T,input_type,threshold, name_list, predict_folder, num_cls, gt_folder):
    for idx in range(start,len(name_list),step):
        name = name_list[idx]
        if input_type == 'png':
            predict_file = os.path.join(predict_folder,'%s.png'%name)
            predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
            if num_cls == 81:
                predict = predict - 91
        elif input_type == 'npy':
            predict_file = os.path.join(predict_folder,'%s.mat'%name)
            # predict_file = os.path.join(predict_folder,'%s.mat'%str(idx))
            # if not os.path.exists(predict_file):
            #     print(predict_file)
            #     continue
            predict_dict=sio.loadmat(predict_file)
            
            try:
                h, w = list(predict_dict.values())[-1].shape
            except:
                # count+=1
                print(name)
                continue
            tensor = np.zeros((num_cls,h,w),np.float32)
            for key in list(predict_dict.keys())[3:]:
                # index=embs.index(key)
                tensor[int(key)+1] = predict_dict[key]/255
            tensor[0,:,:] = threshold 
            predict = np.argmax(tensor, axis=0).astype(np.uint8)
            # print('predict file', predict_file, predict.mean(), predict.std())

        gt_file = os.path.join(gt_folder,'%s.png'%name)
        gt = np.array(Image.open(gt_file))
        cal = gt<255
        if predict.shape != gt.shape:
            continue
        mask = (predict==gt) * cal
        
        for i in range(num_cls):
            P[i].acquire()
            P[i].value += np.sum((predict==i)*cal)
            P[i].release()
            T[i].acquire()
            T[i].value += np.sum((gt==i)*cal)
            T[i].release()
            TP[i].acquire()
            TP[i].value += np.sum((gt==i)*mask)
            TP[i].release()

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    # count=0
    txt = f'{predict_folder}/text.txt'
    
    # f=open(txt,"a+")
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold, name_list, predict_folder, num_cls, gt_folder))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    loglist['std'] = np.std(np.array(IoU)) * 100
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('\n')
        print(f'FP = {fp*100}, FN = {fn*100}')
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()

import cv2
import torch.nn.functional as F
# cam visual_code
def show_cam_on_image(img, mask):
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img.size[1],img.size[0]), mode='bilinear', align_corners=False).squeeze().squeeze()
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def save_segmentation(orig_image: Image, mask: torch.Tensor, file_id: str, args):
    if not os.path.exists(args.save_samples_path):
        os.makedirs(args.save_samples_path)
    save_path = os.path.join(args.save_samples_path, file_id + '.jpg')
    
    cam = show_cam_on_image(orig_image, mask)
    segmented_image = Image.fromarray(cam[:,:,::-1])
    segmented_image.save(save_path)

def save_sample_segmentations(args):
    predict_folder = args.cam_npy_dir
    masks_paths = [os.path.join(predict_folder, x) for x in os.listdir(predict_folder)[:args.save_count]]
    masks_ids = [x.split(os.sep)[-1].replace('.mat', '') for x in masks_paths]
    for mask_path, mask_id in zip(masks_paths, masks_ids):
        orig_image_path = os.path.join(voc_path, 'JPEGImages', mask_id + '.jpg')
        orig_image = Image.open(orig_image_path).convert("RGB")
        saved_dict = sio.loadmat(mask_path)
        for class_id in list(saved_dict.keys())[3:]:
            mask_tensor = torch.Tensor(saved_dict[class_id] / 255)
            save_segmentation(orig_image, mask_tensor, mask_id + str(class_id), args)



class Args:
    val_split = 1.
    comment='train1464'
    curve=True
    end=51
    base_dir = 'sio_maps'
    image_dir = os.path.join(base_dir,'images')
    cam_npy_dir = os.path.join(base_dir, 'images')
    gt_dir=os.path.join(voc_path, 'SegmentationClassAug')
    list='../../ConsistencySegmenter/dataset/voc12/val_id.txt' # TODO
    # list='/root/autodl-tmp/wjl/ptp_diffusion/voc12/train_aug_id.txt'
    logfile=os.path.join(base_dir,'eval.txt')
    num_classes=21
    start=50
    t=None
    type='npy'
    sample_images=True
    save_samples_path = os.path.join(base_dir, 'samples')
    save_count=30

def test():
    args = Args()

    if args.sample_images:
        save_sample_segmentations(args)

    if not os.path.exists(args.cam_npy_dir):
        os.makedirs(args.cam_npy_dir)

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    prediction_ids = [x.split('.')[0] for x in os.listdir(args.cam_npy_dir)]
    name_list = [x for x in df['filename'].values if x in prediction_ids]

    val_split_size = int(args.val_split * len(name_list))
    val_name_list = name_list[:val_split_size]
    test_name_list = name_list[val_split_size:]

    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, args.t, printlog=True)
        writelog(args.logfile, loglist, args.comment)
        best_thr = None
    else:
        l = []
        max_mIoU = 0.0
        best_thr = 0.0
        print(f"Finding threshold on a data split of {len(val_name_list)}")
        for i in range(args.start, args.end):
            t = i/100.0
            loglist = do_python_eval(args.cam_npy_dir, args.gt_dir, val_name_list, args.num_classes, args.type, t)
            l.append(loglist['mIoU'])
            # print('%d/%d background score: %.3f\tmIoU: %.3f%%'%(i, args.end, t, loglist['mIoU']))
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
            # else:
            #     break
        print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
        writelog(args.logfile, {'mIoU':l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment)
    
    print(f"Using best threshold of {best_thr} for test split of size {len(test_name_list)}")
    loglist_test = do_python_eval(args.cam_npy_dir, args.gt_dir, test_name_list, args.num_classes, args.type, best_thr)
    print(f"Final score: {loglist_test['mIoU']}, std: {loglist_test['std']}")
    
if __name__ == '__main__':
    test()