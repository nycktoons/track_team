import cv2 as cv
#import detect
from ultralytics import YOLO
import numpy as np
import math
import time

'''___________________class frame check_______________'''
class FPScheck:
    def __init__(self):
        self.start_time=time.time()
        self.frame_count=0
    
    def get_fps(self):
        self.frame_count+=1
        elapse_time=time.time()- self.start_time
        fps=self.frame_count/elapse_time
        return fps, elapse_time, self.frame_count
        
        
class Vision:
    '''____________________Initialize class______________________________'''
    def __init__(self, input:int, track_mode:bool): #0 for track and 1 for track and qr code
        self.model=YOLO('best.pt', verbose=False, task="segment")
        self.input=input
        self.start=False
        self.detector= cv.QRCodeDetector()
        self.track_mode=track_mode
        self.track_found=False # Initialize track found status
    
    '''_______________________________Open Camera and See surrounding________________________'''
    def look(self):
        #Open the Video capture device
        cap= cv.VideoCapture(self.input)
        
        #Check if capture device available
        if not cap.isOpened():
            return 'No input'
        
        #change frame height and width to correlate with model dimensions
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 240)
        
        #Initialise Frame Check
        fpc= FPScheck()
        #loop through each frame
        while True:
            ret, frame=cap.read()
            
            #break loop and exit function if frame is none
            if not ret:
                return 'None Frame'
            
            '''If frame true deside from this point how you want to use the frame'''
            
            frame= self.segment(frame)
            #data, dir = self.detect_qr(frame)
            #print(data, dir)
            
            if self.track_found:
                status_label = "Track Found"
                cv.putText(frame, status_label, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                status_label = "Track Not Found"
                cv.putText(frame, status_label, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv.imshow('Web Cam', frame)
            f, _, _ = fpc.get_fps()
            print('Frame rate is ', f)
           
            if cv.waitKey(1) & 0xFF==ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    '''_____________________________________Segment method___________________________________________'''
    def segment(self, frame):
        '''_____________________________helping function for track deviation_____________'''
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
        '''______________________________________Segmentation begins here_____________________________'''
        
        #get dimensions for frame center line
        frame_height, frame_width, _ = frame.shape
        center_x=frame_width//2
        
        #Predict on frame
        mask_image=np.zeros_like(frame)
        results = self.model.predict(source=[frame], conf=0.45, save=False, imgsz=256, verbose=False, classes=None, stream=True)
        
        self.track_found = False # Reset track found status for each frame
        # Anaylize results
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
            
            #Run only if track is detected    
                if obj_cls==1:   
                    # Draw the mask on the mask_image
                    cv.fillPoly(mask_image, [seg_int], color=(0, 255, 0))  # Green color
                    self.track_found = True # Track Found indicator
                    
                    #print(seg_int)
                    
                    #get the points and gradient of the line of best fit for the mask
                    if len(seg_int) > 2:  
                        [vx, vy, x0, y0] = cv.fitLine(seg_int, cv.DIST_L2, 0, 0.01, 0.01)
                        #print(vx, vy, x0, y0)
                        
                        #ensure gradient even for small values of vx
                        epsilon= 1e-6
                        if abs(vx) < epsilon:
                            #vx = epsilon
                            vx = 0.1
                            gradient=0 #gradient would be undefined; but we use zero
                            '''we are just checking whether gradient is positive, 0 won't affect anything'''
                        else:
                            gradient= vy/vx
                        
                        #draw line of best fit   
                        lefty = int((-x0 * vy / vx) + y0)
                        righty = int(((frame.shape[1] - x0) * vy / vx) + y0)
                        #print(lefty)
                        #print(righty)
                        cv.line(frame, (frame.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 2)  # Yellow line
                        
                        #get the angle between line-best-fit and center-line, using angle between vector principle
                        center_line = (center_x, 0, center_x, frame_height-1)
                        other_line = (frame.shape[1] - 1, righty, 0, lefty)
                        acute_angle = acute_angle_between_lines(center_line, other_line)
                        
                        #change the angle to range -1 to 0 to 1
                        value_range = map_value(acute_angle, 0, 90, 0, 1)
                        
                        if gradient < 0:  # Check if value is negitive 
                            value_range = value_range * -1
                        
                        #print(f"Acute Angle: {acute_angle:.2f} degrees")
                        label = f'{objs_lst[int(obj_cls)]}: {conf:.2f}  {value_range:.2f}'
                        cv.putText(frame, label, (10, 230), cv.FONT_HERSHEY_SIMPLEX, .5, (211, 229, 250), 2)
                        #if track only mode is off
                elif not self.track_mode: #
                    cv.fillPoly(mask_image, [seg_int], color=(255, 0, 0))  # Blue color
                    label = f'{objs_lst[int(obj_cls)]}: {conf:.2f})'
                    # Optionally, add text showing the class, confidence, and coordinates
                    cv.putText(frame, label, (10, 210), cv.FONT_HERSHEY_SIMPLEX, 0.5, (211, 229, 250), 2)
                else:
                    #if track only mode is on
                    cv.fillPoly(mask_image, [seg_int], color=(0, 255, 0))  # Green color
                    label = f'{objs_lst[1]}: {conf:.2f})'
                    # Optionally, add text showing the class, confidence, and coordinates
                    #cv.putText(frame, label, (seg_int[0, 0], seg_int[0, 1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    self.track_found = True
                    
            # Blend the mask_image with the frame
            
            frame = cv.addWeighted(frame, 0.7, mask_image, 0.3, 0)
            
                #draw center line
            #cv.line(frame, (center_x, 0), (center_x, frame_height-1), (255,0,0), 5)
        return frame
    
    '''_________________________Get QR code data_______________________'''
    def detect_qr(self, frame):
    #convert the frame to grey scale
        gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred=cv.GaussianBlur(gray, (5,5), 0)
        
        #detect qr code
        data, vertices, _ = self.detector.detectAndDecode(blurred)
        if vertices is not None and data:
            vertices = vertices.reshape(-1, 2)
            if len(vertices)==4:
                top_left, top_right, bottom_right, bottom_left= vertices
                angle = np.degrees(np.arctan2(bottom_right[1] - top_right[1], bottom_right[0] - top_right[0]))

                    # Determine the facing direction based on the angle
                if -45 <= angle < 45:
                    direction = "East"
                elif 45 <= angle < 135:
                        direction = "North"
                elif angle >= 135 or angle < -135:
                        direction = "West"
                elif -135 <= angle < -45:
                        direction = "South"
            else:
                #print("Warning: QR code detected but vertices are not 4 points.")
                return False, False
            return data, direction
        return False, False

if __name__ == '__main__':
      vs=Vision(1, 0) # 0 = detects the QR code and the track, 1 = Detects the track alone
      vs.look()  