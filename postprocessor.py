class Detection:
  def __init__(self,box,label,score):
    self.box=box
    self.label=label
    self.score=score
  def __str__(self):
    return(f"{self.label} | {self.score} | {self.box}")

class Postprocessor:
    def __init__(self,definition={0:'person',2:'car'},thresholds={'person':0.4,'car':0.5},checkpoint_type='train'):
        """
        definition turns numerical class to word class.
        thresholds should be in format "word-class": float  where float is [0,1]
        """
        assert checkpoint_type in {'model_zoo','train'}
        self.checkpoint_type=checkpoint_type
        self.definition=definition
        self.thresholds=thresholds
        self._class_names = set([v for v in self.definition.values()])
        self._class_i = len(self._class_names)-1
        self.toggle_class()
    def _get_detections_from_predictions(self,result):
        instances = result.pred_instances.numpy().cpu()
        if self.checkpoint_type=='model_zoo':
            [Detection(box,self.definition.get(label,'None'),score) for box,label,score in zip(instances.bboxes,instances.labels,instances.scores)]    
        return [Detection(box,self.definition.get(label,'None'),score) for box,label,score in zip(instances['bboxes'],instances['labels'],instances['scores'])]
    def _filter_detections(self,detections):
        dets = [det for det in detections if det.label in self._filter_class]
        return [det for det in dets if det.score>self.thresholds.get(det.label,0)]
    def toggle_class(self):
        self._class_i+=1
        if self._class_i == len(self._class_names):
            self._filter_class = self._class_names
            return
        elif self._class_i > len(self._class_names):
            self._class_i=0
        self._filter_class = list(self._class_names)[self._class_i]
        print(self._filter_class)
    def get_detections_from_predictions(self,result):
        dets = self._get_detections_from_predictions(result)
        return self._filter_detections(dets)