from tqdm import tqdm
import cv2
import pandas as pd
from glob import glob

def getImgName(TorV, img_file_str):
    imgname = []
    for f in glob(f'./{TorV}/{img_file_str}/*.jpg'):
        # ./Training/img_pear_normal1/V006_80_0_00_01_01_25_0_b07_20201005_0016_S01_1.jpg
        f = f.split('/')
        f = f[-1]
        imgname.append(f)
    return imgname

# def cropApple(TorV, filename, label_file_str, img_file_str):
#     for idx, f in tqdm(enumerate(filename)):
#         # sleep(0.1)
#     # if idx % 32 ==0:
#         # print(idx, f)
#         label = pd.read_json(f'./{TorV}/{label_file_str}/{f}.json')
#         disease = label.iloc[8, 1]
#         if disease == 6:
#             print(idx, f)
#             bbox = label.iloc[-1, -1][0]
#             crop = label.iloc[9, 1]
#             print(f"crop_code:{crop}, disease_code:{disease}")

#             try:
#                 srcBGR = cv2.imread(f"./{TorV}/{img_file_str}/{f}")
#                 srcRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
#                 dstRGB = srcRGB[bbox.get('ytl'):bbox.get('ybr'), bbox.get('xtl'):bbox.get('xbr')].copy()
            
#                 dstBGR = cv2.cvtColor(dstRGB, cv2.COLOR_RGB2BGR)
#                 if disease == 0:
#                     disease_num = "00"
#                 elif disease == 3:
#                     disease_num = '03'
#                 elif disease == 4:
#                     disease_num = '04'
#                 elif disease == 5:
#                     disease_num = '05'
#                 elif disease == 6:
#                     disease_num = '06'
#                 else:
#                     disease_num = '07'
            
#                 cv2.imwrite(f'./Training/trainset/apple/{disease_num}/{f}.jpg', dstBGR)
#                 # cv2.imwrite(f'./{TorV}/cropped/apple/{disease_num}/{f}.jpg', dstBGR)
#                 print("saved")

#             except:
#                 pass
        
#         else:
#             pass
    
#     # else:
#     #     pass

def countApple(TorV, filename, label_file_str, img_file_str):
    zero = 0
    three = 0
    four = 0
    five = 0
    six = 0
    seven = 0 
    for idx, f in enumerate(filename):
        # sleep(0.1)
    # if idx % 32 ==0:
        # print(idx, f)
        label = pd.read_json(f'./{TorV}/{label_file_str}/{f}.json')
        disease = label.iloc[8, 1]
        if disease == 0:
            zero += 1
        elif disease == 3:
            three += 1
        elif disease == 4:
            four += 1
        elif disease == 5:
            five += 1
        elif disease == 6:
            six += 1
        else:
            seven += 1
    print("zero:",zero,"/three:",three, "/four:",four,"/five:", five, "/six:", six, "/seven:", seven)
            

# apple_normal1_name_T = getImgName("Training", 'img_apple_normal1')
# countApple("Training", apple_normal1_name_T, 'label_apple_normal', 'img_apple_normal1')

# apple_normal2_name_T = getImgName("Training", 'img_apple_normal2')
# countApple("Training",apple_normal2_name_T, 'label_apple_normal', 'img_apple_normal2')

# apple_infected_name_T = getImgName("Training", 'img_apple_infected')
# countApple("Training", apple_infected_name_T, 'label_apple_infected', 'img_apple_infected')

# apple_normal_name_V = getImgName("Validation", 'img_apple_normal')
# countApple("Validation", apple_normal_name_V, 'label_apple_normal', 'img_apple_normal')

apple_infected_name_V = getImgName("Validation", 'img_apple_infected')
countApple("Validation",apple_infected_name_V, 'label_apple_infected', 'img_apple_infected')