import cv2
from email_util import send_email
from mmdet.apis import init_detector, inference_detector
from mmcv.transforms import Compose
from visualize import visualize_boxes
from postprocessor import Postprocessor
import json, os
from iou import match_boxes
import time
# Set up the email parameters
EMAIL_PARAMS = {
    'sender_email': "XX-SENDER-XX@gmail.com",
    'receiver_email': "XX-RECEIVER-XX@gmail.com",
    'subject': "Hello from Python!",
    'message': "This is a test email sent from a Python script.",
    'image': 'parking.jpg',
    'region': 'us-east-1'
}

CONFIG = 'faster-rcnn_r50_fpn_1x_coco/faster-rcnn_r50_fpn_1x_coco.py'
CHECKPOINT = 'faster-rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
VIDEO = "Webcam Feed"
MONITOR_COORDS = 'monitor.json'
def save_monitors(json_path,monitors):
    json.dump(monitors,open(json_path,'w'),indent=2)
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
def check_last_state(last_state,spaces):
    if len(last_state)!=len(spaces):
        return False
    for last,space in zip(last_state,spaces):
        if last != space:
            return False
    return True
def main():
    # Build the model from a config file and a checkpoint file
    model = init_detector(CONFIG, CHECKPOINT, device='cuda:0')

    # Test a video and show the results
    # Build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    # Filter
    postprocessor = Postprocessor(definition={2:'car'},
        thresholds={'car':0.2},
        checkpoint_type='model_zoo')

    
    cam = cv2.VideoCapture('/dev/video0')
    cv2.namedWindow(VIDEO, 0)
    gui = Gui()
    cv2.setMouseCallback(VIDEO, gui.mouse_callback)

    # Load Existing Coords if they exist
    if os.path.exists(MONITOR_COORDS):
        monitors = json.load(open(MONITOR_COORDS,'r'))
        gui.load_monitors(monitors)
    aggregator=Aggregator(n_seconds=10)
    last_state = []
    email_system_on=False

    while True:
        ret,frame = cam.read()
        if not ret:
            break
        gui.update(frame)
        predictions = inference_detector(model, frame, test_pipeline=test_pipeline)
        # t.toc('inference')
        detections = postprocessor.get_detections_from_predictions(predictions)
        # Det boxes
        det_boxes = [det.box for det in detections]
        # Monitors
        monitors = gui.get_monitors()
        

        # Matches
        matches = match_boxes(monitors,det_boxes)
        aggregator.update(matches)

        # Visualize
        # visualize_boxes(frame, det_boxes,color=[0,255,0],size=1)
        active_box = gui.get_active_box()
        if active_box is not None:
            visualize_boxes(frame,[active_box],color=[255,0,0],size=3)
        # visualize_boxes(frame,monitors,color=[255,0,0],size=2)
        if len(last_state)==len(monitors):
            occupieds = [monitor for last,monitor in zip(last_state,monitors) if last]
            visualize_boxes(frame,occupieds,color=[0,0,255],size=2)
            unoccupieds = [monitor for last,monitor in zip(last_state,monitors) if not last]
            visualize_boxes(frame,unoccupieds,color=[0,255,0],size=2)
        # Display the frame in the named window
        cv2.imshow(VIDEO,gui.frame)
        cv2.imwrite(EMAIL_PARAMS['image'],frame)

        if aggregator.check():
            occuppied_spaces = aggregator.get_totals()
            is_same = check_last_state(last_state,occuppied_spaces)
            last_state = occuppied_spaces
        
            if not is_same:
                if len(last_state)==len(monitors):
                    occupieds = [monitor for last,monitor in zip(last_state,monitors) if last]
                    visualize_boxes(frame,occupieds,color=[0,0,255],size=2)
                    unoccupieds = [monitor for last,monitor in zip(last_state,monitors) if not last]
                    visualize_boxes(frame,unoccupieds,color=[0,255,0],size=2)
                    cv2.imwrite(EMAIL_PARAMS['image'],frame)
                    print("saving image...")
                    time.sleep(1)
                print("")
                print("====================")
                print("Change Detected!")
                print("These spaces are occuppied:")
                print(last_state)
                if email_system_on:
                    print("Sending Alert!")
                    send_email(**EMAIL_PARAMS)
                else:
                    print("Emailing is off.")
                print("====================")
            
        # Exit the loop if 'q' is pressed
        chd=cv2.waitKey(1)
        if chd==ord('q'):
            break
        elif chd==ord('r'):
            gui.reset()
        elif chd==ord('t'):
            email_system_on = not email_system_on



    save_monitors(MONITOR_COORDS,gui.get_monitors())
    # Release the webcam and close the window
    cam.release()
    cv2.destroyAllWindows()
        
    #Send an email
    #send_email(**EMAIL_PARAMS)


if __name__=='__main__':
    main()