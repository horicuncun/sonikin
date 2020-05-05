from keras.models import load_model
import cv2
import numpy as np
from PIL import Image

model = load_model('my_model.h5')  # 事前に学習したモデルを読み込む
sonikin_img = cv2.imread('./sonikin.jpg')  # ソニ禁画像を事前に読み込む
cap = cv2.VideoCapture(0)  # Camera読み込み
q = []
before_frame = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = -1

    if before_frame is not None:

        # 前フレームからの差分
        cv2.accumulateWeighted(gray, before_frame, 0.7)
        mdframe = cv2.absdiff(gray, cv2.convertScaleAbs(before_frame))
        thresh = cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 最も大きい輪郭を求める
        max_area = 0
        target = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if max_area < area:
                max_area = area
                target = cnt

        # 長方形をトリミングし、右上の100 * 100ピクセルだけ抜き出す
        x, y, w, h = cv2.boundingRect(target)
        if y + 100 < frame.shape[1] and x + w - 100 > 0:
            trimmed = frame[y:y + 100, x + w - 100: x + w]
            cv2.imshow("Trimmed", trimmed)
            cv2.imwrite('./tmp.jpg', trimmed, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image = np.array(Image.open('./tmp.jpg').resize((25, 25)))
            image = image.transpose(2, 0, 1)
            image = image.reshape(
                1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            result = model.predict_classes(np.array([image / 255.]))[0]
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ソニ禁判定が直近20フレームで7フレーム以上あったらソニ禁画像を表示
    if len(q) > 20:
        q.pop(0)
    q.append(result)
    if(q.count(1) > 6):
        height, width = sonikin_img.shape[:2]
        frame[0:height + 0, 0:width + 0] = sonikin_img

    # 表示
    cv2.imshow('MotionDetected Area Frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    before_frame = gray.copy().astype('float')

cap.release()
cv2.destroyAllWindows()
