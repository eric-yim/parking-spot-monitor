# Parking Spot Monitor

This is the repository for the video demonstration here:

- [Tiktok]()

This is a demonstration, and no support will be provided.

## Pre-reqs

It's strongly encouraged that you install with a virtual environment, such as [venv](https://docs.python.org/3/library/venv.html).

- Install [MMdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html)

- Boto, OpenCV
```
pip install boto3 
pip install opencv-python
```

## Set up

#### Change the ```EMAIL_PARAMS``` in ```camera.py```. 

- ```sender_email``` is the "FROM" address. The sender can be any email account, but you have to connect it through AWS SES.
- ```receiver_email``` is the "TO" address.
- ```region``` should match what you set up in AWS.

```
EMAIL_PARAMS = {
    'sender_email': "codingai.alert@gmail.com",
    'receiver_email': "codingaitiktok@gmail.com",
    'subject': "Hello from Python!",
    'message': "This is a test email sent from a Python script.",
    'image': 'parking.jpg',
    'region': 'us-east-1'
}
```

#### Download an object detector from MMdetection Model Zoo

[Link](https://mmdetection.readthedocs.io/en/latest/model_zoo.html)

You will need 2 things:

- The ```.pth``` file
- The ```config.py``` file

Set these in the camera.py file:

```
CONFIG = 'faster-rcnn_r50_fpn_1x_coco/faster-rcnn_r50_fpn_1x_coco.py'
CHECKPOINT = 'faster-rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
```

## Run

Run ```python camera.py```.

To draw bounding box using left click. Your box should align with where you expect a car to appear on screen.

To reset the boxes, press "R".

To turn on emailing, press "T".
