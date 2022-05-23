import cv2
import numpy as np
import time
import tensorflow as tf

np.set_printoptions(suppress=True)


# Load the model
model = tf.keras.models.load_model('saved_models/model_inceptionv3.hdf5')


# Load the labels
with open('labels.txt', 'r') as f:
   class_names = f.read().split('\n')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

size = (224, 224)


cap = cv2.VideoCapture(0)

while cap.isOpened():

   start = time.time()
   ret, img = cap.read()
   imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   height, width, channels = img.shape
   img_resized = cv2.resize(imgRGB, size)
   img_array = np.asarray(img_resized)

   prediction =  model(tf.expand_dims(img_array, 0)/255)
   if prediction > 0.5:
        score = 'non_fire'
   else:
        score = 'fire'
   

   end = time.time()
   totalTime = end - start

   fps = 1 / totalTime
   cv2.putText(img, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

   cv2.putText(img, score, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
   cv2.imshow('Classification Resized', img_resized)
   cv2.imshow('Classification Original', img)


   if cv2.waitKey(5) & 0xFF == 27:
      break


cv2.destroyAllWindows()
cap.release()