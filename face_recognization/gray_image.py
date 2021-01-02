import cv2

for i in range(200):
    img = cv2.imread('sample/pos1/pos'+str(i)+'.jpg', 0)
    try:
        img5050 = cv2.resize(img, (50, 50))#将图片裁剪成50*50大小
        cv2.imshow("img", img5050)
        cv2.waitKey(20)
        cv2.imwrite('sample/pos/pos1'+str(i)+'.jpg', img5050)
    except:
        continue

for i in range(120):
    img = cv2.imread('sample/pneg1/pneg'+str(i)+'.jpg', 0)
    try:
        cv2.imwrite('sample/pneg/pneg1'+str(i)+'.jpg', img)
    except:
        continue
