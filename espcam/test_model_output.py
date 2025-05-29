from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load your model
model = YOLO('best.pt')

# HTML template with video stream
HTML = '''
<!doctype html>
<html>
<head>
    <title>Live Baby Detection</title>
</head>
<body>
    <h2>Live Baby Detection from Webcam</h2>
    <img src="{{ url_for('video_feed') }}" style="border:2px solid black;">
</body>
</html>
'''

# Video generator function
def generate_frames()'''
from flask import Flask, request, render_template_string
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO('best.pt')

HTML = '''
<!doctype html>
<title>Baby Pose Detection</title>
<h2>Upload an image</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file><br><br>
  <input type=submit value=Upload>
</form>
{% if label %}
<h3>Prediction: {{ label }}</h3>
<img src="{{ img_data }}" style="max-width:500px;">
{% endif %}
'''

def read_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    label = None
    img_data = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        img = read_image(file.stream)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = model(img_rgb)[0]  # first results object
        
        if len(results.boxes) > 0:
            cls_id = int(results.boxes.cls[0])
            label = results.names.get(cls_id, 'Unknown')
        else:
            label = 'No prediction'

        img_data = encode_image_to_base64(img)

    return render_template_string(HTML, label=label, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
'''

'''
from flask import Flask, request, render_template_string
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load your trained YOLO model (make sure best.pt is in the same folder)
model = YOLO('best.pt')

# HTML template
HTML = '''
<!doctype html>
<title>Baby Presence Detection</title>
<h2>Upload an image to detect if baby is present</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file><br><br>
  <input type=submit value=Upload>
</form>
{% if label %}
  <h3>Prediction: {{ label }}</h3>
  <img src="{{ img_data }}" style="max-width:500px; border: 2px solid black;">
{% endif %}
'''

# Convert uploaded file to OpenCV image
def read_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

# Encode image to base64 for displaying in HTML
def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    label = None
    img_data = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        # Read and preprocess image
        img = read_image(file.stream)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run model prediction
        results = model(img_rgb)[0]  # first Results object

        # Check if any boxes were detected
        if len(results.boxes) > 0:
            label = 'Baby Present'
        else:
            label = 'No Baby Detected'

        # Convert image to display
        img_data = encode_image_to_base64(img)

    return render_template_string(HTML, label=label, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
'''

