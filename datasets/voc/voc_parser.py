import os
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time
from utils.io_utils import dump_pickle, load_pickle
from pprint import pprint
from datasets.configs import names2class

data_root = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_data(data, split='trainval'):
    st = time.time()
    data_path = os.path.join(data_root, '{}_{}.pkl'.format(data, split))
    if os.path.exists(data_path):
        data = load_pickle(data_path)
    else:
        data = parse_ori_voc(data, split)
    print('load time:', time.time() - st)
    return data


def parse_ori_voc(data, split='trainval', use_diff=False):
    if data == 'VOC2012' and split == 'test':
        print('VOC2012 has not test file')
        exit(1)

    # save results
    all_anns = []
    classes_count = {}

    # todo: use absolute data path
    data_path = '/nfs/xs/Datasets/{}/VOCdevkit/{}'.format(data, data)
    # data_path = os.path.join(data_root, data)
    anns_path = os.path.join(data_path, 'Annotations')  # xml
    imgs_path = os.path.join(data_path, 'JPEGImages')

    img_files = []
    with open(os.path.join(data_path, 'ImageSets', 'Main', '{}.txt'.format(split))) as f:
        for line in f:  # VOC2012: 2008-2011
            img_files.append(line.strip() + '.jpg')

    for img in tqdm(img_files):
        # print(img)
        et = ET.parse(os.path.join(anns_path, img.replace('.jpg', '.xml')))
        element = et.getroot()  # 通过 find 来找到每个结构的数据
        element_objs = element.findall('object')
        element_width = int(element.find('size').find('width').text)
        element_height = int(element.find('size').find('height').text)
        if len(element_objs) > 0:
            # img attrs
            ann_data = {
                'filepath': os.path.join(imgs_path, img),
                'width': element_width,
                'height': element_height,
            }
            # label attrs
            boxes, labels, difficulties = [], [], []
            for obj in element_objs:
                # label
                class_name = obj.find('name').text
                labels.append(names2class[class_name])
                if class_name not in classes_count:  # stats
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1
                # box
                obj_bbox = obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                boxes.append([x1, y1, x2, y2])
                # difficulty
                difficulties.append(int(obj.find('difficult').text))

            boxes, labels, difficulties = np.array(boxes), np.array(labels), np.array(difficulties)

            if not use_diff:
                keep = np.where(np.array(difficulties) == 0)[0]
                # print(keep)
                boxes = boxes[keep]
                labels = labels[keep]

            ann_data['boxes'] = boxes
            ann_data['labels'] = labels
            # pprint(ann_data)
            all_anns.append(ann_data)

    pprint(classes_count)

    out_path = os.path.join(data_root, '{}_{}.pkl'.format(data, split))
    dump_pickle(all_anns, out_path)

    return data


def parse_data_test(data, split):
    st = time.time()
    parse_ori_voc(data, split)
    print('time:', time.time() - st)


def parse_all_voc():
    # 执行还是挺快的
    parse_data_test('VOC2007', split='trainval')  # 5011
    parse_data_test('VOC2007', split='test')  # 4952
    parse_data_test('VOC2012', split='trainval')  # 11540, 30s


import xml.etree.ElementTree as ET
import collections
from pprint import pprint
from utils.io_utils import dump_json


# 递归解析 xml 结构化 dict
def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)  # node: annotation; list(node): annotation 下的 children node

    # 如果存在 child，对这些 child 逐个使用 parse_voc_xml
    if children:
        # define_dict:  默认 v 为 list，如果 len(v)=1 提出来 val
        def_dic = collections.defaultdict(list)  # <object> 有多个 children
        # 递归得到的 一组 dict
        for dc in map(parse_voc_xml, children):  # 对每个子 node 使用此函数
            # 将每个子 key:val 传给 def_dic，也用 append 方式添加，处理 key 相同问题
            for ind, v in dc.items():
                def_dic[ind].append(v)
        # 单个元素的 list 转化为 val
        voc_dict = {
            node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}
        }

    # 如果不存在 child，如 <folder>VOC2007</folder> 直接对 text 进行 strip
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text  # 最简单的 dict

    return voc_dict


def demo_parse_voc_xml():
    # 测试 torchvision.datasets.voc
    root = ET.parse('000001.xml').getroot()
    voc_dict = parse_voc_xml(root)
    pprint(voc_dict)
    # dump_json(voc_dict, '000001.json')


if __name__ == '__main__':
    demo_parse_voc_xml()
    pass
