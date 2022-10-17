from argparse import ArgumentParser

from PIL import Image

from SANet import SANet


def parse_args():
    # Setting parameters
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'dir_predict'])
    parser.add_argument('-g','--cuda', action="store_true", help='if use cuda')
    parser.add_argument('--model-path', type=str, default='logs\weight.pth', help='The path of weight')
    parser.add_argument('--input-shape', type=int, default=640, help='size of image')
    parser.add_argument('--confidence', type=float, default=0.5,help='confidence')
    parser.add_argument('--nms-iou', type=float, default=0.3, help='the value of iou in nms')
    parser.add_argument('--letterbox-image', type=bool, default=True, help='undistorted resize')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    mode            = args.mode
    cuda            = args.cuda
    model_path      = args.model_path
    input_shape     = args.input_shape
    confidence      = args.confidence
    nms_iou         = args.nms_iou
    letterbox_image = args.letterbox_image
    Sa = SANet(model_path=model_path, input_shape=input_shape, confidence = confidence, 
                nms_iou = nms_iou, letterbox_image=letterbox_image, cuda=cuda)
    
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = Sa.detect_image(image)
                # r_image.show()
                r_image.save("Misc_87.png")

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm
        
        dir_origin_path = "img/"
        dir_save_path   = "img_out/"

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = Sa.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")