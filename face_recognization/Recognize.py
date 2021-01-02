from imageai.Prediction.Custom import CustomImagePrediction
import os
import cv2
execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet() #设置ResNet模型
prediction.setModelPath(os.path.join(execution_path, "data/models/model_ex-022_acc-0.975000.h5"))
prediction.setJsonPath(os.path.join(execution_path, "data/json/model_class.json"))
 
prediction.loadModel(num_objects=4)

cascade = cv2.CascadeClassifier('cascade.xml')
img = cv2.imread('test12.jpg')
#img = cv2.resize(img, (300, 300))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
faces = cascade.detectMultiScale(gray,1.1,3,cv2.CASCADE_SCALE_IMAGE,(24,24))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    face = img[y: y+h, x:x+w, :]
    face = cv2.resize(face,(96,96))
    cv2.imwrite('test0.jpg', face)
    predictions, probabilities = prediction.predictImage('test0.jpg', result_count=1)
    a = str(predictions[0])
    b = str(probabilities[0])[0:5]
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, a+':'+b, (x, y), font, 0.4, (255, 0, 0), 1)
    print(a + " : " + b)
    cv2.imshow('test', img)
 
