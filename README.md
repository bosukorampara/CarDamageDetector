🚗 Car Damage Detection Using YOLOv8, OpenCV & Flask

An AI-powered system for detecting and marking car dents or damages from both images and live video feeds. This solution can serve as a game-changer for the **automotive** and **insurance** industries by enabling automated damage inspection using deep learning.

✨ Features

* Detects multiple types of car damages and dents using a YOLOv8 model
* Works on both **images** and **live video feeds**
* Uses **OpenCV** for image/video processing
* Flask-based **web interface** for uploading images and viewing results
* Outputs labeled and marked damaged areas with confidence scores

## 🧠 Technologies Used

| Technology                                           | Purpose                                         |
| ---------------------------------------------------- | ----------------------------------------------- |
| [YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection and training on custom dataset |
| [OpenCV](https://opencv.org/)                        | Image and video processing                      |
| [cvzone](https://github.com/cvzone/cvzone)           | Drawing styled bounding boxes                   |
| [Flask](https://flask.palletsprojects.com/)          | Lightweight web framework for frontend          |
| Google Colab                                         | Model training with GPU support                 |


🔧 Installation

✅ Prerequisites

* Python >= 3.6 <= 3.11
* Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install gitpython matplotlib numpy opencv-python pillow psutil PyYAML requests scipy thop torch torchvision tqdm ultralytics pandas seaborn setuptools filterpy scikit-image lap
```

🏋️‍♂️ How to Train the YOLOv8 Model (Generate `best.pt`)

The `best.pt` model file is **not included** in this repository due to its large size.

To train the model yourself and generate `best.pt`, follow these steps:

1. 🔗 Download the Dataset

Get the custom car damage dataset from [Roboflow](https://roboflow.com/) or your own labeled dataset in YOLO format.

2. 📁 Upload to Google Drive

Upload your dataset folder (e.g., `CarDent`) to your Google Drive, typically under:

```
/MyDrive/Datasets/CarDent/
```

3. 📓 Open Google Colab

Create a new [Google Colab](https://colab.research.google.com/) notebook, and use the following code:

```python
!pip install ultralytics
from ultralytics import YOLO

# Train the YOLOv8 model
model = YOLO('yolov8l.pt')  # or yolov8n.pt for a lighter model
model.train(data='/content/drive/MyDrive/Datasets/CarDent/data.yaml', epochs=50, imgsz=640)
```

Make sure to:

* Mount your Google Drive (`from google.colab import drive; drive.mount('/content/drive')`)
* Adjust paths as needed

4. ✅ Get the Trained Model

After training completes, the trained model will be saved as:

```
runs/detect/train/weights/best.pt
```

Download this file and place it in the `Weights/` folder of this project.

---

💡 Tips for Better Training Results

* **Use a larger model** like `yolov8m.pt` or `yolov8l.pt` for higher accuracy (requires more compute).
* **Train for more epochs** (e.g., 100–200) if you're not getting good performance.
* **Ensure class names match** in your `data.yaml`.
* Use **data augmentation** and balanced datasets.


## 📂 Directory Structure

```
CarDentDetector/
├── Weights/
│   └── best.pt                  # Trained YOLOv8 model
├── Media/
│   └── dent_1.jpg               # Sample images
│   └── CarDent.mp4             # Sample video
├── templates/
│   └── index.html              # Flask frontend
├── static/
│   └── styles.css              # Optional styling
├── temp_images/
│   └── processed_image.jpg     # Output image
├── CarDentDetector.py          # Image detection script
├── CarDentDetectorLive.py      # Live video feed detection
├── app.py                      # Flask web server
├── requirements.txt            # Python dependencies
└── README.md
```

---

🧪 Training the YOLOv8 Model

1. Download the **custom dataset** (car dents and damage) from [Roboflow](https://roboflow.com/).
2. Use **Google Colab** for training with GPU acceleration.
3. Upload dataset to `My Drive/Datasets/CarDent`.
4. Modify `data.yaml` path:

   ```yaml
   path: ../drive/MyDrive/Datasets/CarDent
   ```
5. Run the following in Colab:

   ```python
   !pip install ultralytics
   !yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/Datasets/CarDent/data.yaml epochs=50 imgsz=640
   ```
6. Download `best.pt` from the `runs/detect/train/weights/` folder and place it in the `Weights/` directory.

---

🖼️ Detect Car Damage in an Image

Run the standalone image detection script:

```bash
python CarDentDetector.py
```

It will load `dent_1.jpg` from the `Media/` folder, detect damages, and display bounding boxes and labels.

---

🌐 Web Interface Using Flask

Step 1: Start Flask Server

```bash
python app.py
```

Step 2: Access Web App

Open browser and visit: [http://127.0.0.1:5000](http://127.0.0.1:5000) (may differ in your system)

Step 3: Upload Image

Upload a car image via the UI. The app will process and display bounding boxes for detected damages. Click the **Download** button to save the processed image.

---

📽️ Live Video Detection (Optional)

To detect damages from a video feed (e.g., webcam or video file):

```bash
python CarDentDetectorLive.py
```

---

🧾 Flask Route Overview

| Route             | Method | Description                                 |
| ----------------- | ------ | ------------------------------------------- |
| `/`               | GET    | Home page with upload form                  |
| `/check-damages`  | POST   | Accepts uploaded image, runs YOLO detection |
| `/download-image` | GET    | Allows downloading processed image          |

---

📸 Supported Classes

The model can detect the following damage types:

* Bodypanel-Dent
* Front-Windscreen-Damage
* Headlight-Damage
* Rear-windscreen-Damage
* RunningBoard-Dent
* Sidemirror-Damage
* Signlight-Damage
* Taillight-Damage
* Bonnet-dent
* Boot-dent
* Doorouter-dent
* Fender-dent
* Front-bumper-dent
* Pillar-dent
* Quaterpanel-dent
* Rear-bumper-dent
* Roof-dent

---

💡 Future Enhancements

* Add support for bounding box editing in frontend
* Include severity estimation (minor, major dent)
* Deploy as a mobile app using Flask REST APIs
* Add report generation (PDF or HTML summary)

---

📚 References

* [YOLOv8 Documentation](https://docs.ultralytics.com/)
* [OpenCV Official Site](https://opencv.org/)
* [cvzone GitHub](https://github.com/cvzone/cvzone)
* [Flask Documentation](https://flask.palletsprojects.com/)
* [Roboflow for Datasets](https://roboflow.com/)




