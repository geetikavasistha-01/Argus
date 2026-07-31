import os
import argparse
import json
from ultralytics import YOLO

def main(weights_path, data_config):
    print(f"Loading model weights from {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
    model = YOLO(weights_path)
    
    print("Running validation on validation split...")
    results = model.val(data=data_config, split="val")
    
    # Extract results
    # (B) represents bounding box metrics, (M) represents mask/segmentation metrics
    results_dict = results.results_dict
    
    box_precision = results_dict.get("metrics/precision(B)", 0.0)
    box_recall = results_dict.get("metrics/recall(B)", 0.0)
    box_map50 = results_dict.get("metrics/mAP50(B)", 0.0)
    box_map50_95 = results_dict.get("metrics/mAP50-95(B)", 0.0)
    
    seg_precision = results_dict.get("metrics/precision(M)", 0.0)
    seg_recall = results_dict.get("metrics/recall(M)", 0.0)
    seg_map50 = results_dict.get("metrics/mAP50(M)", 0.0)
    seg_map50_95 = results_dict.get("metrics/mAP50-95(M)", 0.0)
    
    # We prioritize segmentation metrics since we are doing pixel-level defect segmentation
    overall_p = seg_precision
    overall_r = seg_recall
    
    # Class names map
    class_names = ["defect_class_1", "defect_class_2", "defect_class_3", "defect_class_4"]
    
    # Get actual class indices evaluated
    ap_classes = results.ap_class_index.tolist() if hasattr(results, 'ap_class_index') else []
    
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        # i is the class_id (0 to 3)
        if i in ap_classes:
            idx = ap_classes.index(i)
            p_c = float(results.seg.p[idx]) if hasattr(results, 'seg') and len(results.seg.p) > idx else 0.0
            r_c = float(results.seg.r[idx]) if hasattr(results, 'seg') and len(results.seg.r) > idx else 0.0
            m_c = float(results.seg.all_ap[idx, 0]) if hasattr(results, 'seg') and hasattr(results.seg, 'all_ap') and len(results.seg.all_ap) > idx else 0.0
        else:
            p_c = 0.0
            r_c = 0.0
            m_c = 0.0
            
        per_class_metrics[name] = {
            "precision": p_c,
            "recall": r_c,
            "mAP50": m_c
        }
        
    # Check targets (NFR: Precision >= 90% (0.90), Recall >= 85% (0.85))
    precision_met = overall_p >= 0.90
    recall_met = overall_r >= 0.85
    
    # Generate gap analysis statement
    gap_comments = []
    if not precision_met:
        gap_comments.append(f"Precision ({overall_p*100:.1f}%) is below the NFR target of 90%.")
    if not recall_met:
        gap_comments.append(f"Recall ({overall_r*100:.1f}%) is below the NFR target of 85%.")
    if precision_met and recall_met:
        gap_comments.append("All NFR targets successfully met.")
    else:
        gap_comments.append("Remediation strategies: 1) Train for 100+ epochs, 2) Apply Focal/Dice loss adjustments, 3) Incorporate CBAM/Attention backbone blocks.")
        
    gap_analysis = " ".join(gap_comments)
    
    report = {
        "model": "yolov8-seg",
        "weights": weights_path,
        "overall": {
            "precision": overall_p,
            "recall": overall_r,
            "mAP50": seg_map50,
            "mAP50-95": seg_map50_95,
            "box_precision": box_precision,
            "box_recall": box_recall,
            "box_mAP50": box_map50,
            "box_mAP50-95": box_map50_95
        },
        "per_class": per_class_metrics,
        "nfr_target_met": {
            "precision_gte_90": precision_met,
            "recall_gte_85": recall_met
        },
        "gap_analysis": gap_analysis
    }
    
    # Save report
    os.makedirs("cv/runs", exist_ok=True)
    report_path = "cv/runs/eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Evaluation report successfully written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-seg model and emit report JSON")
    parser.add_argument("--weights", type=str, default="models/severstal_yolov8m_seg_best.pt", help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="cv/data_prep/dataset.yaml", help="Path to dataset.yaml")
    args = parser.parse_args()
    
    main(args.weights, args.data)
