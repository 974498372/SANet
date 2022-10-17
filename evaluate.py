import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

from PIL import Image
from tqdm import tqdm

from SANet import SANet
from utils.utils_map import get_map


def parse_args():
    # Setting parameters
    parser = ArgumentParser()

    parser.add_argument('-g','--cuda', action="store_true", help='if use cuda')
    parser.add_argument('--model-path', type=str, default='logs\weight.pth', help='The path of weight')
    parser.add_argument('--input-shape', type=int, default=640, help='size of image')
    parser.add_argument('--confidence', type=float, default=0.001,help='confidence')
    parser.add_argument('--nms-iou', type=float, default=0.65, help='the value of iou in nms')
    parser.add_argument('--letterbox-image', type=bool, default=True, help='undistorted resize')
    parser.add_argument('--MINOVERLAP', type=float, default=0.5, help='calculate the iou threshold for mAP')
    parser.add_argument('--map-vis', type=bool, default=False, help='visualization of mAP calculations')
    parser.add_argument('--VOCdevkit-path', type=str, default='VOCdevkit', help='path of the dataset')
    parser.add_argument('--map-out-path', type=str, default='map_out', help='path of the result')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cuda            = args.cuda
    model_path      = args.model_path
    input_shape     = args.input_shape
    confidence      = args.confidence
    nms_iou         = args.nms_iou
    letterbox_image = args.letterbox_image
    MINOVERLAP      = args.MINOVERLAP
    map_vis         = args.map_vis
    VOCdevkit_path  = args.VOCdevkit_path
    map_out_path    = args.map_out_path

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names = ['Target']

    
    print("Load model.")
    dba = SANet(model_path=model_path, input_shape=input_shape, confidence = confidence, 
                nms_iou = nms_iou, letterbox_image=letterbox_image, cuda=cuda)
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".png")
        image       = Image.open(image_path)
        if map_vis:
            image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".png"))
        dba.get_map_txt(image_id, image, class_names, map_out_path)
    print("Get predict result done.")
        

    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")


    print("Get map.")
    get_map(MINOVERLAP, True, path = map_out_path)
    print("Get map done.")
