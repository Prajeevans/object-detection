import cv2

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


while True:
    ret, frame = cap.read()

    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
       (x,y,w,h)= bbox
       class_name = classes[class_id]
       cv2.putText(frame, str(class_name), (x, y - 5),cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2)
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)


    cv2.imshow("Face-detection", frame)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27: 
        break
cv2.destroyAllWindows()