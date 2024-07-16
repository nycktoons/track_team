import cv2 as cv
from ultralytics import YOLO
import numpy as np
import math
#from fpscheck import FPScheck

cap = cv.VideoCapture(1)

model = YOLO('best.pt', verbose=False, task="segment")
if not cap.isOpened():
    print("No input")
    exit()

# Initialize a list to store the path of segmented lines
path = []

desired_width = 640
desired_height = 480

cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height) 
cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Check current frame dimentions
    frame_height, frame_width, _ = frame.shape

    # Check if frame needs resizing to maintain 4:3 aspect ratio
    if frame_width / frame_height != 4 / 3:
        # Calculate new width and height
        new_height = frame_width * 3 // 4
        frame = frame[:new_height, :, :] #crop to maintain aspect ration
    
    # Draw a line in the middle of the screen parallel to with the y-axis with reduced opacity
    center_x = frame.shape[1] // 2
    line_color = (255, 255, 0) # Blue color
    line_thickness = 6
    line_opacity = 0.3 
    overlay = frame.copy()
    cv.line(overlay, (center_x, 1), (center_x, frame_height), line_color, line_thickness, 1) # Blue line
    cv.addWeighted(overlay, line_opacity, frame, 1 - line_opacity, 0, frame)  # Blend the line with the frame using addWeighted
    
    #frame = cv.resize(frame, (480, 640))
    mask_image = np.zeros_like(frame)
    results = model.predict(source=[frame], conf=0.55, save=False, imgsz=256, verbose=False, classes=None, stream=True)
    
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

                scale_factor = 5.0
                #only run if track is found    
                M = cv.moments(seg_int)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    center_coords = (cX, cY)

                    # Scale the centroid coordinates
                    cX_scaled = int(cX * scale_factor)
                    cY_scaled = int(cY * scale_factor)

                    # Rotate the centroid 90 degrees counter-clockwise around the center of the frame
                    center_x = frame.shape[1] // 2
                    center_y = frame.shape[0] // 2

                    angle_rad = math.radians(90)  # Convert angle to radians for math functions
                    cos_theta = math.cos(angle_rad)
                    sin_theta = math.sin(angle_rad)

                    rotated_cX = int((cX_scaled - center_x) * cos_theta - (cY_scaled - center_y) * sin_theta + center_x)
                    rotated_cY = int((cX_scaled - center_x) * sin_theta + (cY_scaled - center_y) * cos_theta + center_y)

                    #Now use rotated_cX and rotated_cY in your further processing or drawing
                    # Draw the mask on the mask_image
                    cv.fillPoly(mask_image, [seg_int], color=(0, 255, 0))  # Green color

                    # Draw the centroid and the path
                    cv.circle(frame, center_coords, 5, (255, 0, 0), -1)  # Blue circle
                    path.append(center_coords)  # Add the centroid to the path list
                    
                    # Calculate angle relative to center line
                    angle = math.atan2(cY - frame.shape[0] // 2, cX - center_x)
                    angle_deg = math.degrees(angle)
                    angle_deg_int = round(angle_deg)
            
                    label = f'{(angle_deg_int)}'
                    # # Determine label based on angle
                    # if -45 <= angle_deg <= 45:
                    #     label = '0' # parallel to center line
                    # elif angle_deg > 45:
                    #     label = f'{(angle_deg - 45) / 45:.2f}' # Perpendicular to right
                    # else:
                    #     label = f'{(angle_deg + 45) / 45:.2f}' # Perpendicular to left

                    # Rotate the centroid relative to the orientation of the mask
                    rotated_cX = cX
                    rotated_cY = cY

                    # Draw the line on the frame
                    if len(seg_int) > 2:  # Ensure there are enough points to fit a line
                        [vx, vy, x0, y0] = cv.fitLine(seg_int, cv.DIST_L2, 0, 0.01, 0.01)
                        if abs(vx) < 1e-6:
                            vx=1e-6
                        lefty = int((-x0 * vy / vx) + y0)
                        righty = int(((frame.shape[1] - x0) * vy / vx) + y0)
                        cv.line(frame, (frame.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 2)  # Yellow line

                    # Draw lines connecting the path points
                    if len(path) > 1:
                    #  cv.polylines(frame, [np.array(path, np.int32)], False, (0, 0, 255), 2)  # Red line
                        pass
                    #label = f'{objs_lst[int(obj_cls)]}: {conf:.2f} @ ({center_coords[0]}, {center_coords[1]})'
                    cv.putText(frame, label, (rotated_cX - 50, rotated_cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 51, 255), 1) # Draw label near centroid
            else:
                 cv.fillPoly(mask_image, [seg_int], color=(0, 255, 0))  # Green color
                 label = f'{objs_lst[int(obj_cls)]}: {conf:.2f})'
                 #cv.putText(frame, label, (seg_int[0, 0], seg_int[0, 1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 51, 255), 1) # Optionally, add text showing the class, confidence, and coordinates
                 cv.putText(frame, label, (cX - 50, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 51, 255), 1)
            #Blend the mask_image with the frame
            #frame = cv.addWeighted(frame, 0.7, mask_image, 0.3, 0)
            #f, _, _ = fpc.get_fps()
            #print('Frame rate is ', f)
            
    cv.imshow('Webcam Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
