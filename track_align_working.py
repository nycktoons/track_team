import cv2 as cv
from ultralytics import YOLO
import numpy as np
import math
#from fpscheck import FPScheck


# Function to calculate the angle between two vectors
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / (mag_v1 * mag_v2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Function to ensure the angle is acute
def ensure_acute_angle(angle):
    if angle > 90:
        return 180 - angle
    return angle

# Function to calculate the acute angle between two lines
def acute_angle_between_lines(line1, line2):
    # Line 1 endpoints
    x1, y1, x2, y2 = line1
    
    # Line 2 endpoints
    x3, y3, x4, y4 = line2
    
    # Calculate direction vectors
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])
    
    # Calculate the angle between the vectors
    angle = calculate_angle(v1, v2)
    
    # Ensure the angle is acute
    acute_angle = ensure_acute_angle(angle)
    
    return acute_angle

# Changes range from degrees to our custom scale

def map_value(value, min_original, max_original, min_target, max_target):
    # Calculate ranges
    range_original = max_original - min_original
    range_target = max_target - min_target

    # Map value from original range to  target range
    value_target = (value - min_original) * (range_target / range_original) + min_target

    return value_target
# Example usage
# Center line (vertical line)
#center_line = (320, 0, 320, 480)

# Another line
#other_line = (100, 100, 500, 400)

# Calculate the acute angle
#acute_angle = acute_angle_between_lines(center_line, other_line)
#print(f"Acute Angle: {acute_angle:.2f} degrees")



###############################################################################
cap = cv.VideoCapture(1)

model = YOLO('best.pt', verbose=False, task="segment")
if not cap.isOpened():
    print("No input")
    exit()

# Initialize a list to store the path of segmented lines
path = []

cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 240)
while True:
    ret, frame = cap.read()

     # Inside the main loop after reading the frame
    frame_height, frame_width, _ = frame.shape
    center_y=frame_height//2
    center_x=frame_width//2

    # Check if frame dimensions need to be swapped due to rotation
    if frame_width > frame_height:
        pass
    # Swap width and height 
        #frame = cv.transpose(frame)
        #frame = cv.flip(frame, flipCode=1)  # Flip around y-axis

    # Ensure all operations (drawing lines, calculating centroids) are based on the updated frame dimensions

    if not ret:
        break
    #frame = cv.resize(frame, (640, 640))
    mask_image = np.zeros_like(frame)
    results = model.predict(source=[frame], conf=0.65, save=False, imgsz=256, verbose=False, classes=None, stream=True)
    
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
                    print(seg_int)
                    if len(seg_int) > 2:  # Ensure there are enough points to fit a line
                        [vx, vy, x0, y0] = cv.fitLine(seg_int, cv.DIST_L2, 0, 0.01, 0.01)
                        #gradient= vy/vx if vx!=0 else float('inf')
                        print(vx, vy, x0, y0)
                        epsilon= 1e-6
                        if abs(vx) < epsilon:
                            vx = epsilon
                            gradient=0
                        else:
                            gradient= vy/vx
                            
                        lefty = int((-x0 * vy / vx) + y0)
                        righty = int(((frame.shape[1] - x0) * vy / vx) + y0)
                        print(lefty)
                        print(righty)
                        cv.line(frame, (frame.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 2)  # Yellow line
                        center_line = (center_x, 0, center_x, frame_height-1)
                        other_line = (frame.shape[1] - 1, righty, 0, lefty)
                        acute_angle = acute_angle_between_lines(center_line, other_line)
                        value_range = map_value(acute_angle, 0, 90, 0, 1)
                        
                        if gradient < 0:  # Check if value is negitive 
                            value_range = value_range * -1
                        
                        print(f"Acute Angle: {acute_angle:.2f} degrees")
                    # Draw lines connecting the path points
                    if len(path) > 1:
                    #  cv.polylines(frame, [np.array(path, np.int32)], False, (0, 0, 255), 2)  # Red line
                        pass
                    label = f'{objs_lst[int(obj_cls)]}: {conf:.2f}  {value_range:.2f}'
                    cv.putText(frame, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 51, 255), 2)
            else:
                 cv.fillPoly(mask_image, [seg_int], color=(255, 0, 0))  # Green color
                 label = f'{objs_lst[int(obj_cls)]}: {conf:.2f})'

                # Optionally, add text showing the class, confidence, and coordinates
            cv.putText(frame, label, (seg_int[0, 0], seg_int[0, 1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Blend the mask_image with the frame
            frame = cv.addWeighted(frame, 0.7, mask_image, 0.3, 0)
            cv.line(frame, (center_x, 0), (center_x, frame_height-1), (255,0,0), 5)
            #f, _, _ = fpc.get_fps()
            #print('Frame rate is ', f)
            
    cv.imshow('Webcam Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
