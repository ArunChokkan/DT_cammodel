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
def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(img_rgb)[0]

        # Draw label
        if len(results.boxes) > 0:
            label = 'Baby Present'
            color = (0, 255, 0)
        else:
            label = 'No Baby Detected'
            color = (0, 0, 255)

        # Draw text on frame
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
