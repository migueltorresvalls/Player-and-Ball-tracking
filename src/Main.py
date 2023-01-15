import cv2
import torch
import numpy as np

from Yolo import Yolo
from Minimapa import Minimapa

ruta = "El punto más viral del Estrella Damm Santander Open 2021 _ World Padel Tour (1)_Trim.mp4"
cap = cv2.VideoCapture(ruta)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
model = torch.hub.load('C:\\Users\mtorr\\Proyectos_python\\padel\\yolov_5\\yolov5-master', 'custom', path='yolov5s.pt', source='local', _verbose=True)
od = Yolo(model, cap)


im_src = cv2.imread("padel1.png")
im_dst = cv2.imread("planta2.png")

pts_dst = np.array([[4, 328], [2, 70], [259, 325], [259, 71]])
minimapa = Minimapa(im_src, im_dst, [], pts_dst)

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", od.get_coordinates)

while True: 

    frame = od.frame
            
    od.draw_bbox("person")
    od.draw_pt_medio()
    od.draw_labels()
    
     
    for i in od.labels: 
        minimapa.representa_punto(od.labels[i]["point"], od.labels[i]["color"])
    minimapa.im_dst = cv2.imread("planta2.png")
    
    
    key = cv2.waitKey(0)
    if key == ord("q"): break
        
    if len(od.coordinates_src) == 4:
        minimapa.pts_src = np.array(od.coordinates_src)
        
    cv2.imshow("frame", frame)
        
    if not od.is_finished():
        od.generate_next_frame()
        
cv2.destroyAllWindows()