import cv2
import pandas as pd
import os
import ast


IMAGE_DIR = "/path/to/image_directory"
CSV_PATH = "/path/csv_file/containing_both_ground_truth_model_prediction" 
OUTPUT_DIR = "/output/triplecomparios_labels"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "filtered_analysis_results.csv")

"""
The start image from where you want to start processing
"""
START_IMG = "start_image.jpg"
"""
The end image where you want to stop processing in the folder
"""
END_IMG = "Prokto_7_clip_600_3.jpg"

# threshold setting
IOU_THRESHOLD = 0.5

def parse_box(box_str):
    if pd.isna(box_str) or box_str == "N/A":
        return None
    if isinstance(box_str, (list, tuple)):
        return box_str
    try:
        return ast.literal_eval(box_str)
    except (ValueError, SyntaxError):
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        df = pd.read_csv(CSV_PATH)
    
        df.columns = df.columns.str.strip() 
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    
    required_cols = ['Image_Name', 'Pred_Box', 'GT_Box']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        print(f"\nCRITICAL ERROR: Missing columns: {missing_cols}")
        print(f"Found in CSV: {list(df.columns)}")
        return

    
    results_list = []
    print("Generating full analysis CSV...")
    
    for _, row in df.iterrows():
        iou_val = row.get('IoU', 0.0)
        iou = 0.0 if (pd.isna(iou_val) or iou_val == "N/A") else float(iou_val)
        
        pred_box = parse_box(row['Pred_Box'])
        gt_box = parse_box(row['GT_Box'])
        
        # result type
        if iou >= IOU_THRESHOLD and gt_box and pred_box:
            res_type = "True_Positive"
        elif pred_box and (not gt_box or iou < IOU_THRESHOLD):
            res_type = "False_Positive"
        elif gt_box and not pred_box:
            res_type = "False_Negative"
        else:
            res_type = "Background"

        results_list.append({
            "Image_Name": row['Image_Name'],
            "Result_Type": res_type,
            "Pred_Class": row.get('Pred_Class', 'N/A'),
            "GT_Class": row.get('GT_Class', 'N/A'),
            "Confidence": row.get('Confidence', 0.0),
            "IoU": iou,
            "Pred_Box": pred_box,
            "GT_Box": gt_box
        })

    output_df = pd.DataFrame(results_list)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Full CSV saved: {OUTPUT_CSV}")

    # --- 3. VISUALIZE SPECIFIC RANGE ---
    all_images = sorted(df['Image_Name'].astype(str).str.strip().unique())
    
    try:
        if START_IMG not in all_images or END_IMG not in all_images:
            print(f"Warning: Range {START_IMG} to {END_IMG} not fully found. Using available subset.")
            #
            target_images = [img for img in all_images if START_IMG <= img <= END_IMG]
        else:
            start_idx = all_images.index(START_IMG)
            end_idx = all_images.index(END_IMG)
            target_images = all_images[start_idx : end_idx + 1]
            
        print(f"Visualizing {len(target_images)} images...")
    except ValueError:
        print("Range error. Check if filenames match exactly.")
        return

    # Counter for summary
    stats = {"True_Positive": 0, "False_Positive": 0, "False_Negative": 0}

    for img_name in target_images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        original = cv2.imread(img_path)
        if original is None: continue

        pane_gt = original.copy()
        pane_pred = original.copy()
        pane_logic = original.copy()
        
        img_rows = output_df[output_df['Image_Name'] == img_name]

        for _, row in img_rows.iterrows():
            
            if row['Result_Type'] in stats:
                stats[row['Result_Type']] += 1

            # 1. Ground Truth Pane (Blue)
            if row['GT_Box']:
                gx1, gy1, gx2, gy2 = map(int, row['GT_Box'])
                cv2.rectangle(pane_gt, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
            
            # 2. Prediction Pane (Red)
            if row['Pred_Box']:
                px1, py1, px2, py2 = map(int, row['Pred_Box'])
                cv2.rectangle(pane_pred, (px1, py1), (px2, py2), (0, 0, 255), 2)

            # 3. Logic Analysis Pane
            if row['Result_Type'] == "True_Positive":
                x1, y1, x2, y2 = map(int, row['Pred_Box'])
                cv2.rectangle(pane_logic, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green
            elif row['Result_Type'] == "False_Positive":
                x1, y1, x2, y2 = map(int, row['Pred_Box'])
                cv2.rectangle(pane_logic, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red
            elif row['Result_Type'] == "False_Negative":
                x1, y1, x2, y2 = map(int, row['GT_Box'])
                cv2.rectangle(pane_logic, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pane_gt, "GT ONLY", (20, 40), font, 1.2, (255, 0, 0), 3)
        cv2.putText(pane_pred, "PRED ONLY", (20, 40), font, 1.2, (0, 0, 255), 3)
        cv2.putText(pane_logic, "ANALYSIS (Green=TP, Red=FP, Blue=FN)", (20, 40), font, 0.8, (255, 255, 255), 2)

        combined = cv2.hconcat([pane_gt, pane_pred, pane_logic])
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), combined)

    print("\nSummary for Visualized Range")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
