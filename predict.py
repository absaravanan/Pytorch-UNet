import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import numpy as np
import cv2

from unet import UNet
from unet import fast_unet

from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms
import imutils
import random

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    # left_square, right_square = split_img_into_squares(img)

    # left_square = hwc_to_chw(left_square)
    # right_square = hwc_to_chw(right_square)
    img = hwc_to_chw(img)

    # X_left = torch.from_numpy(left_square).unsqueeze(0)
    # X_right = torch.from_numpy(right_square).unsqueeze(0)

    X_img = torch.from_numpy(img).unsqueeze(0)
    
    if use_gpu:
        # X_left = X_left.cuda()
        # X_right = X_right.cuda()
        X_img = X_img.cuda()

    with torch.no_grad():
        # output_left = net(X_left)
        # output_right = net(X_right)
        output = net(X_img)

        # left_probs = output_left.squeeze(0)
        # right_probs = output_right.squeeze(0)

        probs = output.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        
        # left_probs = tf(left_probs.cpu())
        # right_probs = tf(right_probs.cpu())

        probs = tf(probs.cpu())

        # left_mask_np = left_probs.squeeze().cpu().numpy()
        # right_mask_np = right_probs.squeeze().cpu().numpy()

        full_mask = probs.squeeze().cpu().numpy()

    # full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    # if use_dense_crf:
    #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files



def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def overlay_mask(img, mask):
    # res = cv2.bitwise_and(img,img,mask = mask)
    img[mask!=0] = (0,0,255)
    # img = cv2.resize(img,(500, 350))
    # cv2.imwrite("sample.jpg", img)
    # cv2.imshow("1", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img



if __name__ == "__main__":
    args = get_args()
    in_files = args.input

    image_dir = "/home/ai/Downloads/quality/Aadhaar_Quality_dataset/"
    out_files = []
    for r, d, f in os.walk(image_dir):
        for file in f:
            # print (os.path.join(r,file))
            out_files.append(os.path.join(r,file))
    # out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)
    # net = fast_unet(n_channels=3, n_classes=1)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(out_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        # angle = random.randint(0,359)
        angle = random.choice([0,90,180,270])
        img = img.rotate(angle)

        print (img.size)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        import time

        start = time.time()
        img = img.resize((512, 512))


        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        # mask = np.expand_dims(mask, axis=0)
        # mask = np.transpose(mask, axes=[1, 2, 0])

        print (time.time() - start )

        if args.viz:
            
            # img = img.resize((img.size[1], img.size[1]))
            img = np.array(img)
            mask = np.float32(mask)
            # print (img.shape)
            # print (mask.shape)

            print("Visualizing results for image {}, close to continue ...".format(fn))
            img = overlay_mask(img, mask)
            # cv2.imwrite("tmp/"+os.path.basename(fn), img)
            plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])
            print("Mask saved to {}".format(out_files[i]))
