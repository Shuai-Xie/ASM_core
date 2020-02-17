import torch


# convert float points of image to int points in a image
def pt_float2int(pt, img_w, img_h):
    # x,y
    return max(0, min(round(pt[0] * img_w), img_w - 1)), \
           max(0, min(round(pt[1] * img_h), img_h - 1))


def pt_int2float(pt, img_w, img_h):
    # x,y
    return max(0, min(pt[0] / img_w, 1)), \
           max(0, min(pt[1] / img_h, 1))


# convert normalized pts to (x1,y1,x2,y2)
def cvt_4floatpts_to_box(points, img_w, img_h):
    x1, y1 = pt_float2int(points[0], img_w, img_h)  # 左上
    x2, y2 = pt_float2int(points[1], img_w, img_h)  # 右上
    x3, y3 = pt_float2int(points[2], img_w, img_h)  # 右下
    x4, y4 = pt_float2int(points[3], img_w, img_h)  # 右下

    xmin, ymin = min([x1, x2, x3, x4]), min([y1, y2, y3, y4])
    xmax, ymax = max([x1, x2, x3, x4]), max([y1, y2, y3, y4])

    return xmin, ymin, xmax, ymax  # tuple


# convet points of bbox to (x,y,w,h) bbox
def cvt_4floatpts2xywh(points, img_w=640, img_h=360):
    x1, y1 = pt_float2int(points[0], img_w, img_h)  # 左上
    x2, y2 = pt_float2int(points[1], img_w, img_h)  # 右上
    x3, y3 = pt_float2int(points[2], img_w, img_h)  # 右下
    x4, y4 = pt_float2int(points[3], img_w, img_h)  # 右下

    xmin, ymin = min([x1, x2, x3, x4]), min([y1, y2, y3, y4])
    xmax, ymax = max([x1, x2, x3, x4]), max([y1, y2, y3, y4])

    return [xmin, ymin, xmax - xmin, ymax - ymin]  # [x,y,w,h]


# convert (x1,y1,x2,y2) to normalized pts
# 左上角 (x1,y1), 顺时针
def cvt_box_to_4floatpts(box, img_w, img_h):
    x1, y1 = pt_int2float((box[0], box[1]), img_w, img_h)
    x2, y2 = pt_int2float((box[2], box[1]), img_w, img_h)
    x3, y3 = pt_int2float((box[2], box[3]), img_w, img_h)
    x4, y4 = pt_int2float((box[0], box[3]), img_w, img_h)

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def box_iou(box1, box2):
    """
    如果是多个 box 相比，要求 box1,box2 的 dim1 相等
    box1,box2 一定要是 float 类型
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # coords of inter section
    inter_rect_x1 = torch.max(b1_x1, b2_x1)  # 要求 b1,b2 shape 一样
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) \
                 * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-16  # 避免 div 0

    iou = torch.div(inter_area, union_area)
    return iou


def demo_iou():
    box1 = torch.tensor([
        [1, 1, 3, 3],
        [1, 1, 5, 5],
    ], dtype=torch.float32)
    box2 = torch.tensor([
        [2, 2, 3, 3],
        [2, 2, 8, 8],
        [3, 3, 5, 5]
    ], dtype=torch.float32)

    for b in box1:
        b = b.repeat((box2.size()[0], 1))
        iou = box_iou(b, box2)
        print(iou)
        exit(0)


if __name__ == '__main__':
    pass
