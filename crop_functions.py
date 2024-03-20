from glob import glob
import pandas as pd
import cv2
print(cv2.__version__)
from tqdm import tqdm
import random
# from time import sleep



def getImgName(TorV, img_file_str):
    imgname = []
    for f in glob(f'./{TorV}/{img_file_str}/*.jpg'):
        # ./Training/img_pear_normal1/V006_80_0_00_01_01_25_0_b07_20201005_0016_S01_1.jpg
        f = f.split('/')
        f = f[-1]
        imgname.append(f)
    return imgname



def cropPear(TorV, filename, label_file_str, img_file_str):
    for idx, f in tqdm(enumerate(filename)):
        # sleep(0.1)
        print(idx, f)
        label = pd.read_json(f'./{TorV}/{label_file_str}/{f}.json')
        bbox = label.iloc[-1, -1][0]
        disease = label.iloc[8, 1]
        crop = label.iloc[9, 1]
        print(f"crop_code:{crop}, disease_code:{disease}")

        try:
            srcBGR = cv2.imread(f"./{TorV}/{img_file_str}/{f}")
            srcRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
            dstRGB = srcRGB[bbox.get('ytl'):bbox.get('ybr'), bbox.get('xtl'):bbox.get('xbr')].copy()
         
            dstBGR = cv2.cvtColor(dstRGB, cv2.COLOR_RGB2BGR)
            if disease == 0:
                disease_num = "00"
            elif disease == 1:
                disease_num = '01'
            else:
                disease_num = '02'
        
            cv2.imwrite(f'./{TorV}/cropped/pear/{disease_num}/{f}.jpg', dstBGR)
            print("saved")

        except:
            pass


def cropPear_aug(TorV, filename, label_file_str, img_file_str):
    for idx, f in tqdm(enumerate(filename)):
        # sleep(0.1)
        print(idx, f)
        label = pd.read_json(f'./{TorV}/{label_file_str}/{f}.json')
        bbox = label.iloc[14, 1][0]
        disease = label.iloc[9, 1]
        crop = label.iloc[10, 1]
        print(f"crop_code:{crop}, disease_code:{disease}")

        try:
            srcBGR = cv2.imread(f"./{TorV}/{img_file_str}/{f}")
            srcRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
            dstRGB = srcRGB[bbox.get('ytl'):bbox.get('ybr'), bbox.get('xtl'):bbox.get('xbr')].copy()
         
            dstBGR = cv2.cvtColor(dstRGB, cv2.COLOR_RGB2BGR)
            if disease == 0:
                disease_num = "00"
            elif disease == 1:
                disease_num = '01'
            else:
                disease_num = '02'
        
            cv2.imwrite(f'./{TorV}/cropped/pear/{disease_num}/{f}.jpg', dstBGR)
            print("saved")

        except:
            pass


def cropApple(TorV, filename, label_file_str, img_file_str):
    for idx, f in tqdm(enumerate(filename)):
        # sleep(0.1)
        # if idx % 32 ==0:
        print(idx, f)
        label = pd.read_json(f'./{TorV}/{label_file_str}/{f}.json')
        disease = label.iloc[8, 1]
        # print(idx, f)
        bbox = label.iloc[-1, -1][0]
        crop = label.iloc[9, 1]
        print(f"crop_code:{crop}, disease_code:{disease}")

        try:
            srcBGR = cv2.imread(f"./{TorV}/{img_file_str}/{f}")
            srcRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
            dstRGB = srcRGB[bbox.get('ytl'):bbox.get('ybr'), bbox.get('xtl'):bbox.get('xbr')].copy()
            dstBGR = cv2.cvtColor(dstRGB, cv2.COLOR_RGB2BGR)
            if disease == 0:
                disease_num = "00"
            elif disease == 3:
                disease_num = '03'
            elif disease == 4:
                disease_num = '04'
            elif disease == 5:
                disease_num = '05'
            elif disease == 6:
                disease_num = '06'
            else:
                disease_num = '07'
        
            cv2.imwrite(f'/media/visbic/MGTEC/fireblight/fire_blight/Training/apple_dataset/train/{disease_num}/{f}.jpg', dstBGR)
            # cv2.imwrite(f'./Training/trainset/apple/{disease_num}/{f}.jpg', dstBGR)
            # cv2.imwrite(f'./{TorV}/cropped/apple/{disease_num}/{f}.jpg', dstBGR)
            print("saved")

        except:
            pass




def cropApple_aug(TorV, filename, label_file_str, img_file_str):
    for idx, f in tqdm(enumerate(filename)):
        # sleep(0.1)
        print(idx, f)
        label = pd.read_json(f'./{TorV}/{label_file_str}/{f}.json')
        bbox = label.iloc[14, 1][0]
        disease = label.iloc[9, 1]
        crop = label.iloc[10, 1]
        print(f"crop_code:{crop}, disease_code:{disease}")

        try:
            srcBGR = cv2.imread(f"./{TorV}/{img_file_str}/{f}")
            srcRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
            dstRGB = srcRGB[bbox.get('ytl'):bbox.get('ybr'), bbox.get('xtl'):bbox.get('xbr')].copy()
        
            dstBGR = cv2.cvtColor(dstRGB, cv2.COLOR_RGB2BGR)
            if disease == 0:
                disease_num = "00"
            elif disease == 3:
                disease_num = '03'
            elif disease == 4:
                disease_num = '04'
            elif disease == 5:
                disease_num = '05'
            elif disease == 6:
                disease_num = '06'
            else:
                disease_num = '07'

            cv2.imwrite(f'/media/visbic/MGTEC/fireblight/fire_blight/Training/trainset/apple/{disease_num}/{f}.jpg', dstBGR)
            # cv2.imwrite(f'./{TorV}/cropped/apple/{disease_num}/{f}.jpg', dstBGR)
            print("saved")

        except:
            pass