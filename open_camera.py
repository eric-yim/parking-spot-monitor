import cv2
import numpy as np
from iou import match_boxes
import time
from emailer import send_email
CAMERA_NAME = '/dev/video0'
WINDOW_NAME = 'erics_cam'
EMAIL_PARAMS = {
    'sender_email': "XX-SENDER-XX@gmail.com",
    'receiver_email': "XX-RECEIVER-XX@gmail.com",
    'subject': "Hello from Python!",
    'message': "This is a test email sent from a Python script.",
    'image': 'parking.jpg',
    'region': 'us-east-1'
}
def visualize_boxes(im,boxes,color=[0,255,0],size=2):
    for box in boxes:
        # if is_wh:
        #     x,y,w,h = box
        #     box = [x,y,x+w,y+h]
        box = [int(round(j)) for j in box]
        cv2.rectangle(im,box[:2],box[2:],color,size)

class Aggregator:
    def __init__(self,n_seconds=10):
        self.n_seconds = n_seconds
        self.reset()
    def reset(self):
        self.t= time.time()
        self.totals = []
        self.n_count = 0
    def check(self):
        t = time.time()
        if (t - self.t) >= self.n_seconds:
            return True
        return False
    def update(self,matches):
        if len(self.totals)!= len(matches):
            self.totals = matches
        else:
            self.totals = [t+m for t,m in zip(self.totals,matches)]
        self.n_count+=1
    def get_totals(self,threshold=0.5):
        """
        smooths noise
        """
        totals =[(t/self.n_count)>threshold for t in self.totals]
        self.reset()
        return totals
class Gui:
    def __init__(self,frame=None):
        self._frame=frame
        self._p1,self._p2 = None,None
        self._monitors = []

    def update(self,frame):
        self._frame=frame
    def reset(self):
        self._monitors = []

    @property
    def frame(self):
        return self._frame
    def mouse_callback(self, event: int, x: int, y: int, flags, param):
        
        if event==cv2.EVENT_LBUTTONUP:
            self._p2 = [x,y]
            self._append_monitor()
        elif event==cv2.EVENT_LBUTTONDOWN:
            self._p1 = [x,y]
        elif event==cv2.EVENT_MOUSEMOVE and (self._p1 is not None):
            self._p2 = [x,y]
    def get_monitors(self):
        return self._monitors
    def get_active_box(self):
        if self._p1 is None:
            return None
        if self._p2 is None:
            return None
        return self._p1 + self._p2
    def _append_monitor(self):
        if self._p1 is None:
            return None
        if self._p2 is None:
            return None
        self._monitors.append(self._p1 + self._p2)
        self._p1, self._p2 = None,None
    def load_monitors(self,monitors):
        self._monitors = monitors
class ObjectDetector:
    def __init__(self,cfg_path='yolov3.cfg',weights_path='yolov3.weights'):
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    def inference(self,blob):
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

        return outputs
    def post_process(self,img, outputs, conf=0.5):
        H, W = img.shape[:2]

        boxes = []
        confidences = []
        classIDs = []
        outputs = np.vstack(outputs)
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                # cv.rectangle(img, p0, p1, WHITE, 1)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        if len(indices)==0:
            return {"boxes":[],"classes":[],"scores":[]}
        return {
            "boxes": [[boxes[i][0], boxes[i][1], boxes[i][0]+boxes[i][2],boxes[i][1]+boxes[i][3]] for i in indices.flatten()],
            "classes": [classIDs[i] for i in indices.flatten()],
            "scores": [confidences[i] for i in indices.flatten()]
        }
class StateStore:
    def __init__(self):
        self.last_state = []
        self.has_changed = False
    def check_last_state(self,spaces):
        if len(self.last_state)!=len(spaces):
            self.last_state = spaces
            self.has_changed=True
            return
        for last,space in zip(self.last_state,spaces):
            if last != space:
                self.last_state = spaces
                self.has_changed=True
                return
        self.has_changed=False
    def get_has_changed(self):
        return self.has_changed
    def check_has_changed(self):
        has_changed = self.has_changed
        self.has_changed=False
        return has_changed
    def get_last_state(self):
        return self.last_state
def main():
    #Open camera
    cam = cv2.VideoCapture(CAMERA_NAME)

    cv2.namedWindow(WINDOW_NAME, 0)
    gui = Gui()
    cv2.setMouseCallback(WINDOW_NAME, gui.mouse_callback)
    detector = ObjectDetector()
    aggregator=Aggregator(n_seconds=10)
    state_store = StateStore()
    while True:
        # Read camera
        ret,frame = cam.read()
        if not ret:
            break

        gui.update(frame)

        # Construct blob for object detector
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        r = blob[0, 0, :, :]

        # Run Yolo
        raw = detector.inference(blob)
        outputs = detector.post_process(frame,raw)

        # Monitors
        monitors = gui.get_monitors()
        # Detections matching against monitors
        matches = match_boxes(monitors,outputs["boxes"])
        aggregator.update(matches)

        # Check Monitors
        if aggregator.check():
            occuppied_spaces = aggregator.get_totals()
            state_store.check_last_state(occuppied_spaces)

        # Visualize
        active_box = gui.get_active_box()
        if active_box is not None:
            visualize_boxes(frame,[active_box],color=[255,0,0],size=3)
        #visualize_boxes(frame,outputs["boxes"],color=[0,255,0],size=2)
        last_state = state_store.get_last_state()
        # Unoccupied spaces in green
        # Occupied spaces in red
        if len(last_state)==len(monitors):
            occupieds = [monitor for last,monitor in zip(last_state,monitors) if last]
            visualize_boxes(frame,occupieds,color=[0,0,255],size=2)
            unoccupieds = [monitor for last,monitor in zip(last_state,monitors) if not last]
            visualize_boxes(frame,unoccupieds,color=[0,255,0],size=2)
        
        # Check if state has changed
        # If state has changed, we want to save the image
        # then we want to send an email with image
        if state_store.check_has_changed():
            cv2.imwrite(EMAIL_PARAMS['image'],frame)
            print()
            print("="*40)
            print("State has changed!")
            print(f"New state: {last_state}")
            print(f"Image saved to {EMAIL_PARAMS['image']}")
            print("="*40)
            time.sleep(1)
            send_email(**EMAIL_PARAMS)
            

            
            

        # Display image
        cv2.imshow(WINDOW_NAME,frame)
        chd = cv2.waitKey(1)
        if chd == ord('q'):
            break



if __name__=='__main__':
    main()