import numpy as np
import json

from .AutoDriveDataset import AutoDriveDataset, AutoDriveDataset_pole
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm

single_cls = True       # just detect vehicle

class CityscapesDataset(AutoDriveDataset_pole):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for pole_city in tqdm(list(self.pole_city_list)):
            pole_list = pole_city.iterdir()
            for pole in tqdm(list(pole_list)):
                pole_path = str(pole)
                image_path = pole_path.replace(str(self.pole_root), str(self.img_root)).replace("gtFine_labelIds", "leftImg8bit")
                disparity_path = pole_path.replace(str(self.pole_root), str(self.disparity_root)).replace("gtFine_labelIds", "disparity")
                # 其他的mask、lane、det文件路径都固定，因为只使用cityscapes数据集来训练杆状物检测
                # 其他部分的模型参数都固定不训练
                mask_path = '/home/user/ZWH/code/YOLOP-main/gt/da_seg_annotations_10k/bdd_seg_gt/val/7d2f7975-e0c1c5a7.png'
                label_path = '/home/user/ZWH/code/YOLOP-main/gt/det_annotations_10k/val/7d2f7975-e0c1c5a7.json' 
                lane_path = '/home/user/ZWH/code/YOLOP-main/gt/ll_seg_annotations_10k/bdd_lane_gt/val/7d2f7975-e0c1c5a7.png'
                # print(pole_path)

                # import pdb; pdb.set_trace()
                with open(label_path, 'r') as f:
                    label = json.load(f)
                data = label['frames'][0]['objects']
                data = self.filter_data(data)
                gt = np.zeros((len(data), 5))
                for idx, obj in enumerate(data):
                    category = obj['category']
                    if category == "traffic light":
                        color = obj['attributes']['trafficLightColor']
                        category = "tl_" + color
                    if category in id_dict.keys():
                        x1 = float(obj['box2d']['x1'])
                        y1 = float(obj['box2d']['y1'])
                        x2 = float(obj['box2d']['x2'])
                        y2 = float(obj['box2d']['y2'])
                        cls_id = id_dict[category]
                        if single_cls:
                            cls_id=0
                        gt[idx][0] = cls_id
                        box = convert((width, height), (x1, x2, y1, y2))
                        gt[idx][1:] = list(box)
                    

                rec = [{
                    'image': image_path,
                    'label': gt,
                    'mask': mask_path,
                    'lane': lane_path,
                    'pole': pole_path,
                    'disparity': disparity_path
                }]

                gt_db += rec
        print('database build finish')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass