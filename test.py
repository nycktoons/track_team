import cv2 as cv
from ultralytics import YOLO
import numpy as np
#from fpscheck import FPScheck

cap = cv.VideoCapture(1)

model = YOLO('best.pt', verbose=False, task="segment")
if not cap.isOpened():
    print("No input")
    exit()

# Initialize a list to store the path of segmented lines
path = []

cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
while True:
    ret, frame = cap.read()

     # Inside the main loop after reading the frame
    frame_height, frame_width, _ = frame.shape

    # Check if frame dimensions need to be swapped due to rotation
    if frame_width > frame_height:
    
    # Swap width and height 
        frame = cv.transpose(frame)
        frame = cv.flip(frame, flipCode=1)  # Flip around y-axis

    # Ensure all operations (drawing lines, calculating centroids) are based on the updated frame dimensions

    if not ret:
        break
    #frame = cv.resize(frame, (640, 640))
    mask_image = np.zeros_like(frame)
    results = model.predict(source=[frame], conf=0.55, save=False, imgsz=240, verbose=False, classes=None, stream=True)
    
    # Initialize Frame Check
    #fpc = FPScheck()

    # Predict on image
    for out in results:
        masks = out.masks
        objs_lst = ['qr_code', 'track']
        for index, box in enumerate(out.boxes):
            seg = masks.xy[index]
            obj_cls, conf, bb = (
                box.cls.numpy()[0],
                box.conf.numpy()[0],
                box.xyxy.numpy()[0],
            )
            seg_int = np.array(seg, dtype=np.int32)
                
            # Calculate the centroid of the mask
            if obj_cls==1:
                #only run if track is found    
                M = cv.moments(seg_int)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    center_coords = (cX, cY)
                    # Draw the mask on the mask_image
                    cv.fillPoly(mask_image, [seg_int], color=(0, 255, 0))  # Green color

                    # Draw the centroid and the path
                    cv.circle(frame, center_coords, 5, (255, 0, 0), -1)  # Red circle
                    path.append(center_coords)  # Add the centroid to the path list
                    
                    if len(seg_int) > 2:  # Ensure there are enough points to fit a line
                        [vx, vy, x0, y0] = cv.fitLine(seg_int, cv.DIST_L2, 0, 0.01, 0.01)
                        lefty = int((-x0 * vy / vx) + y0)
                        righty = int(((frame.shape[1] - x0) * vy / vx) + y0)
                        cv.line(frame, (frame.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 2)  # Yellow line

                    # Draw lines connecting the path points
                    if len(path) > 1:
                    #  cv.polylines(frame, [np.array(path, np.int32)], False, (0, 0, 255), 2)  # Red line
                        pass
                    label = f'{objs_lst[int(obj_cls)]}: {conf:.2f} @ ({center_coords[0]}, {center_coords[1]})'
                    cv.putText(frame, label, (center_coords[0] - 50, center_coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                 cv.fillPoly(mask_image, [seg_int], color=(255, 0, 0))  # Green color
                 label = f'{objs_lst[int(obj_cls)]}: {conf:.2f})'

                # Optionally, add text showing the class, confidence, and coordinates
            cv.putText(frame, label, (seg_int[0, 0], seg_int[0, 1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Blend the mask_image with the frame
            #frame = cv.addWeighted(frame, 0.7, mask_image, 0.3, 0)
            #f, _, _ = fpc.get_fps()
            #print('Frame rate is ', f)
            
    cv.imshow('Webcam Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
