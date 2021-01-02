import glob
import cv2
import os.path

name = input('请输入人物名:')
file_path = "role//" + name + '//'  # 文件夹路径
images_path = glob.glob(os.path.join(file_path, '*.jpg'))  # 所有图片路径
i = 1
cascade = cv2.CascadeClassifier('cascade.xml')
for image_path in images_path:
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray,1.1,3,cv2.CASCADE_SCALE_IMAGE,(24,24)) 
        filename = 'role//' + name + '_face//' + str(i) + '.jpg'
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            face = img[y: y+h, x:x+w, :]
            face = cv2.resize(face,(96,96))
            cv2.imwrite(filename, face)
        i += 1
    except Exception as e:
        print(e)
        continue
 
