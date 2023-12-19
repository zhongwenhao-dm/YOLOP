import os
from yacs.config import CfgNode as CN


# 修改为cityscapes数据集需要修改：
# 数据路径、DATASET.DATASET、POLE_ONLY

_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = (0,1)     
_C.WORKERS = 8
_C.PIN_MEMORY = False
_C.PRINT_FREQ = 20
_C.AUTO_RESUME =False       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']
_C.MODEL.PRETRAINED = "/home/user/ZWH/code/YOLOP-main/runs/CityscapesDataset/_2023-05-21-16-04/epoch-127.pth"
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.PRETRAINED_DET_DA_LL = ""  #/home/user/ZWH/code/YOLOP-main/weights/End-to-end.pth
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)


# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.05  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
_C.LOSS.LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
_C.LOSS.PO_SEG_GAIN = 0.2  # pole object segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.2 # lane line iou loss gain


# DATASET related params
_C.DATASET = CN(new_allowed=True)
# BDD 10k路径
# _C.DATASET.DATAROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/bdd100k_images_10k/bdd100k/images/10k'       # the path of images folder
# _C.DATASET.LABELROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/det_annotations_10k'      # the path of det_annotations folder
# _C.DATASET.MASKROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/da_seg_annotations_10k/bdd_seg_gt'                # the path of da_seg_annotations folder
# _C.DATASET.LANEROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/ll_seg_annotations_10k/bdd_lane_gt'                # the path of ll_seg_annotations folder
# _C.DATASET.POLEROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/po_seg_annotations/bdd_pole_gt'                # the path of po_seg_annotations folder
# _C.DATASET.DISPARITYROOT = ''
# BDD temp_data路径
_C.DATASET.DATAROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/temp_data/image'       # the path of images folder
_C.DATASET.LABELROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/temp_data/det'      # the path of det_annotations folder
_C.DATASET.MASKROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/temp_data/drivable'                # the path of da_seg_annotations folder
_C.DATASET.LANEROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/temp_data/lane'                # the path of ll_seg_annotations folder
_C.DATASET.POLEROOT = '/media/dmgz/MyPassport/3090_ZWH/code/YOLOP-main/gt/temp_data/pole'                # the path of po_seg_annotations folder
_C.DATASET.DISPARITYROOT = ''
# cityscapes数据集路径
# _C.DATASET.DATAROOT = '/media/user/MyPassport/data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit'       # the path of images folder
# _C.DATASET.LABELROOT = ''      # the path of det_annotations folder
# _C.DATASET.MASKROOT = ''                # the path of da_seg_annotations folder
# _C.DATASET.LANEROOT = ''                # the path of ll_seg_annotations folder
# _C.DATASET.POLEROOT = '/media/user/MyPassport/data/cityscapes/pole_seg_gt/gtFine' 
# # _C.DATASET.DISPARITYROOT = '/media/user/MyPassport/data/cityscapes/disparity_trainvaltest/disparity'
# _C.DATASET.DISPARITYROOT = ''

_C.DATASET.DATASET = 'BddDataset_pole'  # CityscapesDataset BddDataset BddDataset_pole
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [720,1280]   # CityscapesDataset[1024,2048] BddDataset[720, 1280]

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add


# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 240

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 2
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train three segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and three segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.POLE_ONLY = False          # Only train po_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task




_C.TRAIN.PLOT = True                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 2
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True
_C.TEST.NMS_CONF_THRESHOLD  = 0.2
_C.TEST.NMS_IOU_THRESHOLD  = 0.6


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir
    
    # if args.conf_thres:
    #     cfg.TEST.NMS_CONF_THRESHOLD = args.conf_thres

    # if args.iou_thres:
    #     cfg.TEST.NMS_IOU_THRESHOLD = args.iou_thres
    


    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )
    #
    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()
