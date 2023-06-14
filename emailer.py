import boto3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_email(sender_email, receiver_email, subject, message, region, image):
    # Create a new SES client
    ses_client = boto3.client('ses', region_name=region)

    # Create a multipart/mixed parent container
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    # Create a multipart/alternative child container
    msg_body = MIMEMultipart('alternative')

    # Attach the message text
    text_part = MIMEText(message, 'plain')
    msg_body.attach(text_part)

    # Attach the image
    with open(image, 'rb') as file:
        img = MIMEImage(file.read(), name="image.png")
    msg.attach(img)

    # Attach the multipart/alternative child container to the multipart/mixed parent container
    msg.attach(msg_body)

    # Send the email
    response = ses_client.send_raw_email(
        Source=sender_email,
        Destinations=[receiver_email],
        RawMessage={'Data': msg.as_string()}
    )

    print("Email sent! Message ID:", response['MessageId'])

# Send the email
if __name__=='__main__':
    # Set up the email parameters
    EMAIL_PARAMS = {
        'sender_email': "XX-SENDER-XX@gmail.com",
        'receiver_email': "XX-RECEIVER-XX@gmail.com",
        'subject': "Hello from Python!",
        'message': "This is a test email sent from a Python script.",
        'image': 'sample.jpg',
        'region': 'us-east-1'
    }
    send_email(**EMAIL_PARAMS)