from yolov3.utils import image_preprocess, postprocess_boxes, nms, load_yolo_weights, detect_image
import cv2
import numpy as np
from yolov3.yolov4 import Create_Yolo
from yolov3.configs import *
import matplotlib.pyplot as plt
import PIL
from PIL import ImageDraw
import easyocr
import string

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom_lp") # use keras weights
reader = easyocr.Reader(['en'])
ALLOW_LIST = [s for s in string.ascii_uppercase+string.digits]

def image_crop_2_array(img_path, debug = True):
  pil_image = PIL.Image.open(img_path).convert('RGB') 
  print(pil_image)
  original_image  = np.array(pil_image) #cv2.imread(img_path)
  if debug:
    plt.figure(figsize = (30, 15))
    plt.imshow(original_image)
    pil_image.show()
  image_data = image_preprocess(np.copy(original_image), [YOLO_INPUT_SIZE, YOLO_INPUT_SIZE])
  image_data = image_data[np.newaxis, ...].astype(np.float32)


  pred_bbox = yolo.predict(image_data)
  image = detect_image(yolo, image_path, "", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if debug:
    plt.figure(figsize=(30,15))
    plt.imshow(image)
    #image.show()
  pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
  pred_bbox = tf.concat(pred_bbox, axis=0)

  bboxes = postprocess_boxes(pred_bbox, original_image, YOLO_INPUT_SIZE, TEST_SCORE_THRESHOLD)
  bboxes = nms(bboxes, TEST_IOU_THRESHOLD, method='nms')
  if len(bboxes) != 0:
    return original_image[int(bboxes[0][1]):int(bboxes[0][3]), int(bboxes[0][0]):int(bboxes[0][2])]

def ocr(img, reader, debug = False, color='yellow', width=2):
  bounds = reader.readtext(img, allowlist=ALLOW_LIST)
  if debug:
    img_copy = PIL.Image.fromarray(img)
    draw = ImageDraw.Draw(img_copy)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        # draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    plt.figure(figsize=(30,15))
    plt.imshow(img_copy)
  return bounds

def licensePlateCleanUp(lp_bounds):
  return lp_bounds

def licensePlateReader(img_path, reader, debug = False):
  lp_cropped = image_crop_2_array(img_path, debug = debug)
  lp_string = ocr(lp_cropped, reader, debug = debug)
  return licensePlateCleanUp(lp_string)


TEST_IMAGE_NAME = "IMAGE_NAME.jpg"
TEST_IMAGE_PATH = "~/Desktop/LicensePlate/training_data/" + TEST_IMAGE_NAME

if __name__ == "__main__" and TEST_IMAGE_NAME:
  print("Testing LP reader")
  print(licensePlateReader(TEST_IMAGE_PATH, reader, True))
