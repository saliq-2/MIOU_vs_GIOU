# MIOU vs GIOU



Model Preds Vs Ground truth Preds




Comparison of Ground-truth (left), Prediction-only (center), and after setting a threshold.... to bee updated (right).



![Prokto_7_clip_0645_0](https://github.com/user-attachments/assets/e6cbe031-99eb-4052-bb50-58efd4090f70)
Figure: Left = GT only (blue boxes), Center = Pred only (red boxes), Right = Analysis (green = TP, red = FP, blue = FN).


Detection Result Analysis & Visualization

A Python-based tool for analyzing object detection results by comparing model predictions against ground truth annotations using IoU-based evaluation. The script generates a detailed analysis CSV and produces side-by-side visualizations highlighting True Positives, False Positives, and False Negatives.

🚀 Features

IoU-based comparison between predicted and ground-truth bounding boxes

Automatic classification into:

True Positive (TP)

False Positive (FP)

False Negative (FN)

Background

Generates a filtered analysis CSV

Visualizes results using three-pane comparison images

Supports visualization over a custom image range

Simple configuration and easy integration into existing pipelines

📂 Project Structure
.
├── analysis_script.py
├── images/
│   └── *.jpg
├── csv/
│   └── detection_results.csv
└── output/
    ├── filtered_analysis_results.csv
    └── *.jpg

📥 Input
Image Directory

Set the path containing the original images:

IMAGE_DIR = "/path/to/image_directory"

CSV File

The CSV must contain the following required columns:

Column Name	Description
Image_Name	Image filename
Pred_Box	Predicted bounding box [x1, y1, x2, y2]
GT_Box	Ground truth bounding box [x1, y1, x2, y2]

Optional columns:

IoU

Pred_Class

GT_Class

Confidence

Example:

Image_Name,Pred_Box,GT_Box,IoU,Confidence
img_001.jpg,"[50,60,200,220]","[55,65,195,215]",0.78,0.92

⚙️ Configuration
IoU Threshold
IOU_THRESHOLD = 0.5

Image Range for Visualization
START_IMG = "start_image.jpg"
END_IMG   = "end_image.jpg"


Only images between these filenames (lexicographically) will be visualized.

🧠 Evaluation Logic
Condition	Result
IoU ≥ threshold and GT & Prediction exist	True Positive
Prediction exists but IoU < threshold or GT missing	False Positive
GT exists but Prediction missing	False Negative
No GT and no Prediction	Background
🖼 Visualization Output

For each image, a three-panel comparison is generated:

Ground Truth Only (Blue)

Prediction Only (Red)

Analysis View

🟢 Green → True Positive

🔴 Red → False Positive

🔵 Blue → False Negative

All images are saved in:

/output/triplecomparios_labels/

📊 Output Files
Analysis CSV
filtered_analysis_results.csv


Includes:

Image_Name

Result_Type

Pred_Class

GT_Class

Confidence

IoU

Pred_Box

GT_Box

Visualization Images

Side-by-side comparison images saved with the original image names.

▶️ Usage
python analysis_script.py

📈 Console Summary

After execution, the script prints:

Summary for Visualized Range
True_Positive: X
False_Positive: X
False_Negative: X

🛠 Dependencies

Install required packages:

pip install opencv-python pandas

🎯 Use Cases

Object detection error analysis

Model evaluation and debugging

Dataset quality inspection

Research and benchmarking visualization

📄 License

This project is intended for research and development purposes.
Feel free to modify and extend it for your use case.
