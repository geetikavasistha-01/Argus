import os
import argparse
import json
import torch
from ultralytics import YOLO

def main(epochs, imgsz, batch, model_size, save_period, patience, fraction):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Select pre-trained weights
    model_name = f"yolov{model_size}-seg.pt"
    print(f"Loading pre-trained YOLO model: {model_name}...")
    model = YOLO(model_name)
    
    # Check device availability
    if torch.cuda.is_available():
        device = "0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using training device: {device}")
    
    # Run training
    print(f"Starting fine-tuning on Severstal dataset for {epochs} epochs at resolution {imgsz}...")
    project_path = os.path.abspath("cv/runs")
    results = model.train(
        data="cv/data_prep/dataset.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_path,
        name="severstal_train",
        exist_ok=True,
        save_period=save_period,
        patience=patience,
        fraction=fraction
    )
    
    # Save the best model to standard models directory
    best_weights_src = os.path.join(project_path, "severstal_train", "weights", "best.pt")
    best_weights_dst = os.path.join("models", f"severstal_yolov{model_size}_seg_best.pt")
    
    if os.path.exists(best_weights_src):
        import shutil
        shutil.copy(best_weights_src, best_weights_dst)
        print(f"Best weights successfully copied to {best_weights_dst}")
    else:
        print(f"Warning: Best weights not found at {best_weights_src}")
        
    # Extract training metrics and write to runs directory
    metrics_path = os.path.join(project_path, "train_metrics.json")
    try:
        metrics = {
            "model_size": model_size,
            "epochs": epochs,
            "imgsz": imgsz,
            "metrics": {
                "box_map50": results.box.map50 if hasattr(results, 'box') else None,
                "box_map50-95": results.box.map if hasattr(results, 'box') else None,
                "seg_map50": results.seg.map50 if hasattr(results, 'seg') else None,
                "seg_map50-95": results.seg.map if hasattr(results, 'seg') else None,
            }
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Training metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Could not extract or save metrics: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg model on Severstal dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model-size", type=str, default="8m", choices=["8n", "8s", "8m", "8l", "8x"], 
                        help="YOLO model size (8n, 8s, 8m, 8l, 8x)")
    parser.add_argument("--save-period", type=int, default=5, help="Save checkpoint every X epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience epochs")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to train on")
    args = parser.parse_args()
    
    main(args.epochs, args.imgsz, args.batch, args.model_size, args.save_period, args.patience, args.fraction)
