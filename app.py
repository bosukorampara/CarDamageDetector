from flask import Flask, request, render_template, jsonify, send_file
import cv2
import math
import cvzone
from ultralytics import YOLO
import numpy as np
import os


app = Flask(__name__)


yolo_model = YOLO("Weights/best.pt")


class_labels = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage', 
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent', 
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent', 
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]


output_folder = "temp_images"
os.makedirs(output_folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check-damages', methods=['POST'])
def check_damages():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  
    results = yolo_model(img)

    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

   
    output_path = os.path.join(output_folder, "processed_image.jpg")
    cv2.imwrite(output_path, img)

    return jsonify({'download_url': '/download-image'})

@app.route('/download-image', methods=['GET'])
def download_image():
    output_path = os.path.join(output_folder, "processed_image.jpg")
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
