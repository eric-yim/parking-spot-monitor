# Parking Spot Monitor

This is the repository for the video demonstration here:

- [Tiktok](https://www.tiktok.com/@codingai/video/7239046318646168875)
- [YouTube](https://youtu.be/G2wkCqTgyx8)

This is a demonstration, and no support will be provided.

## Pre-reqs

It's strongly encouraged that you install with a virtual environment, such as [venv](https://docs.python.org/3/library/venv.html).

- Boto, OpenCV
```
pip install boto3 
pip install opencv-python
```

Download [YOLO cfg and weights](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html) from opencv.


## Set up

#### Change the ```EMAIL_PARAMS``` in ```camera.py```. 

- ```sender_email``` is the "FROM" address. The sender can be any email account, but you have to connect it through AWS SES.
- ```receiver_email``` is the "TO" address.
- ```region``` should match what you set up in AWS.

```
EMAIL_PARAMS = {
    'sender_email': "YOUR_SENDER_EMAIL@gmail.com",
    'receiver_email': "YOUR_RECEIVER_EMAIL@gmail.com",
    'subject': "Hello from Python!",
    'message': "This is a test email sent from a Python script.",
    'image': 'parking.jpg',
    'region': 'us-east-1'
}
```

## Run

Run ```python open_camera.py```.

To draw bounding box using left click. Your box should align with where you expect a car to appear on screen.
