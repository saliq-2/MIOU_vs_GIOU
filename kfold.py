import os
import yaml
import csv
import torch
import torchvision
from ultralytics import YOLO
from tqdm import tqdm


MODEL_PATH = "path_to_model"
DATA_YAML  = "path_yaml_file" #yaml file points to txts which has image list
print(MODEL_PATH)
print(DATA_YAML)
SAVE_DIR_ROOT = "output_directory" 
EXPERIMENT_NAME = "fold_1_experiment"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

def xywhn2xyxy(x, w, h):
    """Converts normalized xywh to pixel xyxy format."""
    labels_xyxy = []
    if len(x) == 0:
        return torch.Tensor([])
    for box in x:
        cls, xc, yc, bw, bh = box
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        labels_xyxy.append([int(cls), x1, y1, x2, y2])
    return torch.tensor(labels_xyxy)

def get_image_list(yaml_path):
    """Parses the dataset YAML to find the list of validation/test images."""
    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    root_path = data_cfg.get('path', '')
    test_file_rel = data_cfg.get('test')
    
    # Resolve path to the text file listing images
    if not os.path.isabs(test_file_rel):
        test_list_path = os.path.join(root_path, test_file_rel)
    else:
        test_list_path = test_file_rel

    print(f"Reading image list from: {test_list_path}")
    with open(test_list_path, 'r') as f:
        img_lines = [x.strip() for x in f.readlines() if x.strip()]

    full_img_paths = []
    for line in img_lines:
        if line.startswith('/'):
            full_img_paths.append(line)
        else:
            full_img_paths.append(os.path.join(root_path, line))
            
    return full_img_paths, data_cfg.get('names', {})

def main():
    model = YOLO(MODEL_PATH)
    

    print("\nRunning Standard YOLO Validation")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.5,
        verbose=True,
        save=True,              
        project=SAVE_DIR_ROOT,  
        name=EXPERIMENT_NAME,   
        exist_ok=True           
    )
    
    save_dir = metrics.save_dir 
    print(f"\nStandard metrics saved to: {save_dir}")


    print("\n Generating Wrong Predictions CSV ")
    
    image_files, class_names = get_image_list(DATA_YAML)
    csv_path = os.path.join(save_dir, "wrong_predictions.csv")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Image_Name', 'Error_Type', 'Class_ID', 'Class_Name', 'Confidence', 'IoU', 'Details', 'Pred_Box_xyxy'])

        for img_path in tqdm(image_files, desc="Analyzing Errors"):
            if not os.path.exists(img_path):
                continue

            file_name = os.path.basename(img_path)
            
            # Independent inference for error analysis
            results = model.predict(img_path, conf=CONF_THRESHOLD, iou=0.5, verbose=False)[0]
            
            pred_boxes = results.boxes.xyxy.cpu()
            pred_clss = results.boxes.cls.cpu()
            pred_confs = results.boxes.conf.cpu()
            h, w = results.orig_shape

            
            label_path = img_path.rsplit('.', 1)[0] + '.txt'
            label_path = label_path.replace('/images/', '/labels/')
            
            gt_boxes = torch.Tensor([])
            gt_clss = torch.Tensor([])
            
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as lf:
                        lines = [list(map(float, line.strip().split())) for line in lf.readlines() if line.strip()]
                    if lines:
                        t_lines = torch.tensor(lines)
                        gt_clss = t_lines[:, 0]
                        gt_boxes = xywhn2xyxy(t_lines, w, h)[:, 1:]
                except:
                    pass

           
            matched_gt_indices = set()
            
        
            if len(pred_boxes) > 0:
                if len(gt_boxes) > 0:
                    iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)
                    for i, p_box in enumerate(pred_boxes):
                        max_iou, max_idx = torch.max(iou_matrix[i], dim=0)
                        max_iou = max_iou.item()
                        max_idx = max_idx.item()
                        
                        p_cls = int(pred_clss[i])
                        p_name = class_names[p_cls] if isinstance(class_names, dict) else str(p_cls)
                        
                        
                        p_box_list = [round(x, 2) for x in p_box.tolist()]
                        
                        if max_iou > IOU_THRESHOLD:
                            gt_cls = int(gt_clss[max_idx])
                            gt_name = class_names[gt_cls] if isinstance(class_names, dict) else str(gt_cls)
                            
                            if p_cls == gt_cls:
                                matched_gt_indices.add(max_idx)
                            else:
                                
                                writer.writerow([file_name, 'FP_Wrong_Class', p_cls, p_name, f"{pred_confs[i]:.4f}", f"{max_iou:.4f}", f"GT: {gt_name}", p_box_list])
                        else:
                            
                            writer.writerow([file_name, 'FP_Background', p_cls, p_name, f"{pred_confs[i]:.4f}", f"{max_iou:.4f}", "Low IoU", p_box_list])
                else:
                    #
                    for i, p_cls in enumerate(pred_clss):
                        p_name = class_names[int(p_cls)] if isinstance(class_names, dict) else str(int(p_cls))
                        p_box_list = [round(x, 2) for x in pred_boxes[i].tolist()]
                        writer.writerow([file_name, 'FP_Ghost', int(p_cls), p_name, f"{pred_confs[i]:.4f}", "0.0", "Image empty", p_box_list])

            # Check False Negatives (Missed Objects) 
            if len(gt_boxes) > 0:
                for idx in range(len(gt_boxes)):
                    if idx not in matched_gt_indices:
                        gt_cls = int(gt_clss[idx])
                        gt_name = class_names[gt_cls] if isinstance(class_names, dict) else str(gt_cls)
                        
                        # For False Negatives, there is NO predicted box.
                        # You might optionally want to log the GT box here instead, but strictly speaking
                        # the "Predicted BBox" is N/A.
                        writer.writerow([file_name, 'False_Negative', gt_cls, gt_name, "N/A", "0.0", "Missed", "N/A"])

    print(f"\nAll results (Metrics + CSV) saved to:\n{save_dir}")

if __name__ == "__main__":
    main()
