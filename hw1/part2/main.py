import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    filename = os.path.basename(args.image_path)
    number_str = filename.split('.')[0]
    image_index = int(number_str) 
    img_rgb = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    guidance_images = [img_gray]
    W = []
    with open(args.setting_path, 'r') as f:
        for idx, line in enumerate(f):
            info = line.strip().split(',')
            if idx > 0 and idx < 6:
                info[:] = [float(x) for x in info]
                W.append(info)
            elif idx == 6:
                sigma_s = int(info[1])
                sigma_r = float(info[3])
    
    for i in range(len(W)):
        guidance = img_rgb[:, :, 0] * W[i][0] + img_rgb[:, :, 1] * W[i][1] + img_rgb[:, :, 2] * W[i][2]
        guidance_images.append(guidance)

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    
    cost = []

    for i in range(len(guidance_images)):
        guidance_image = guidance_images[i]
        gf_out = JBF.joint_bilateral_filter(img_rgb, guidance_image)
        cost.append(np.sum(np.abs(bf_out.astype('int32') - gf_out.astype('int32'))))
        
        if i == 0:
            print('cv2.COLOR_BGR2GRAY cost: ', cost[i])
        else:
            print('R*%.1f+G*%.1f+B*%.1f: ' % (W[i-1][0], W[i-1][1], W[i-1][2]), cost[i])
            
    max_index = cost.index(max(cost))
    min_index = cost.index(min(cost))
    
    
    if max_index !=0 :
        guidance_max = img_rgb[:, :, 0] * W[max_index-1][0] + img_rgb[:, :, 1] * W[max_index-1][1] + img_rgb[:, :, 2] * W[max_index-1][2]
    else:
        guidance_max = guidance_images[0]
    
    if min_index !=0 :
        guidance_min = img_rgb[:, :, 0] * W[min_index-1][0] + img_rgb[:, :, 1] * W[min_index-1][1] + img_rgb[:, :, 2] * W[min_index-1][2]
    else:
        guidance_min = guidance_images[0]
    
    cv2.imwrite('./report_img/guidance_max_%d.png' % image_index, guidance_max)
    cv2.imwrite('./report_img/guidance_min_%d.png' % image_index, guidance_min)

    gf_out_max = JBF.joint_bilateral_filter(img_rgb, guidance_max).astype(np.uint8)
    gf_out_min = JBF.joint_bilateral_filter(img_rgb, guidance_min).astype(np.uint8)
    gf_out_max = cv2.cvtColor(gf_out_max,  cv2.COLOR_RGB2BGR)
    gf_out_min = cv2.cvtColor(gf_out_min,  cv2.COLOR_RGB2BGR)
    cv2.imwrite('./report_img/JBF_max_%d.png' % image_index, gf_out_max)
    cv2.imwrite('./report_img/JBF_min_%d.png' % image_index, gf_out_min)
    

if __name__ == '__main__':
    main()
    