from statistics import mode
# from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('cnn_model_training1.h5')

# image = cv2.imread('Untitled.png', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('Untitled.png', cv2.IMREAD_GRAYSCALE)

pred = model.predict(image.reshape(1, 28, 28, 1))
pred = np.argmax(pred)
dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'0',27:'1',28:'2', 29:'3', 30:'4',31:'5',32:'6', 33:'7', 34:'8', 35:'9'} 

print(dict[pred])