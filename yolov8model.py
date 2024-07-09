from ultralytics import YOLO
import cv2
import math
import cvzone

cap = cv2.VideoCapture("VID_20240103_143231.mp4")
cap.set(3,1280)
cap.set(4,720)

model = YOLO("best.pt")

classNames = ['2cm gap', 'Dominant aramid', 'Dyneema chunk', 'Dyneema twist', 'Frayed carbon', 
             'Gaps', 'Misplaced aramids', 'PC damage', 'Pinch mark', 'Stray threads', 'glue blob']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255))
            
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            conf = math.ceil(box.conf[0]*100)/100
            cvzone.putTextRect(img,f"{conf}",(max(0,x1),max(35,y1)))
            print (conf)
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f"{classNames[cls]} {conf}",(max(20,x1),max(35,y1)))
            
    cv2.imshow("Image",img)
    cv2.waitKey(1)