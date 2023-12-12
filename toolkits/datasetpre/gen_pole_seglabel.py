from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import os
import json 
import numpy as np

from tqdm import tqdm

def poly2patch(poly2d, closed=False, alpha=1., color=None):
    moves = {'L': Path.LINETO,
             'C': Path.CURVE4}
    
    points = [p[:2] for p in poly2d]
    codes = [moves[p[2]] for p in poly2d]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)


    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else 'none',
        edgecolor=color,  # if not closed else 'none',
        lw=1 if closed else 2 * 1, alpha=alpha,
        antialiased=False, snap=True)


def draw_pole(objects, ax):
    plt.draw()

    for obj in objects:
        color = (1,1,1)
        alpha = 1.0
        poly2d = obj['poly2d'][0]
        # import pdb; pdb.set_trace()
        ver = poly2d['vertices']
        t = poly2d['types']
        poly2ds = []
        
        for i in range(len(ver)):
            
            poly = []
            poly.append(ver[i][0])
            poly.append(ver[i][1])
            poly.append(t[i])
            poly2ds.append(poly)
        

        ax.add_patch(poly2patch(poly2ds, closed=True, alpha=alpha, color=color))

    ax.axis('off')



def main(mode="train"):
    # /home/user/ZWH/code/YOLOP-main/gt/po_seg_annotations/bdd_pole_gt
    image_dir = "bdd/bdd100k/images/100k/{}".format(mode)
    file_json = "/home/user/ZWH/code/YOLOP-main/gt/bdd100k_sem_seg_labels_trainval/bdd100k/labels/sem_seg/polygons/sem_seg_{}.json".format(mode)
    out_dir = '/home/user/ZWH/code/YOLOP-main/gt/po_seg_annotations/bdd_pole_gt/{}'.format(mode)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    # json_pd 是一个list，每个元素代表一个图片，如json_pd[0]是一张图片
    # json_pd[0]是一个字典，dict_keys(['name', 'timestamp', 'labels'])
    # json_pd[0]['labels']是一个list,存储了图片内的不同的对象，需要从中找到category为pole的对象
    # json_pd[0]['labels'][0]是一个字典，存储单个对象的属性dict_keys(['id', 'category', 'poly2d'])
    json_pd = json.load(open(file_json))

    
    # 获得只含有pole的图片label信息
    img_label_all = []
    for img in json_pd:
        img_info = {}
        img_info['name'] = img['name']
        img_info['timestamp'] = img['timestamp']

        pole_labels = []
        for label in img['labels']:
            if label['category'] == "pole":
                pole_labels.append(label)

        img_info['labels'] = pole_labels
        img_label_all.append(img_info)


    # 开始对每张图片中的pole生成对应的mask  例如：img_label_all[0]['labels'][1]
    
    for img in tqdm(img_label_all):
        dpi = 80
        w = 16
        h = 9
        image_width = 1280
        image_height = 720
        fig = plt.figure(figsize=(w, h), dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        ax.set_xlim(0, image_width - 1)
        ax.set_ylim(0, image_height - 1)
        ax.invert_yaxis()
        ax.add_patch(poly2patch(
            [[0, 0, 'L'], [0, image_height - 1, 'L'],
            [image_width - 1, image_height - 1, 'L'],
            [image_width - 1, 0, 'L']],
            closed=True, alpha=1., color=(0, 0, 0)))
        data = img['labels']
        draw_pole(data, ax)

        img_name = img['name'].replace('.jpg', '.png')
        out_path = os.path.join(out_dir, img_name)
        fig.savefig(out_path, dpi=dpi)
        plt.close()

    
    
    print("generate pole mask done!!")


if __name__ == '__main__':
    main(mode='train')