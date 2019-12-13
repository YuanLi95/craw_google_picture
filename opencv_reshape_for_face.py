import cv2
import os

image_size = 224  #所需要的图片尺寸
source_path = "./data/train_data/angry"  # 文件来源
target_path = "./data/change_train/angry"  # 文件输出

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)   #改为文件下的所有

i = 0
for file in image_list:
    try:
        image_source = cv2.imread(source_path + file)
        image = cv2.resize(image_source, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
        cv2.imwrite(target_path + file, image)
    except Exception as e:
        print(e)

# image_source = cv2.imread(source_path +"giorgio_ancient-aliens-guy-memes-giorgio-a-tsoukalos-ancient-aliens-E268e727189307c9298ae4cfff109caa7.jpg")
# image = cv2.resize(image_source, (224, 224), 0, 0, cv2.INTER_LINEAR)
# cv2.imwrite(target_path +"giorgio_ancient-aliens-guy-memes-giorgio-a-tsoukalos-ancient-aliens-E268e727189307c9298ae4cfff109caa7.jpg", image)