"""
Wound Detection - REAL-TIME VIDEO MODE (with Inference Benchmarking)
Raspberry Pi 5 + Camera Module v2 + YOLOv8-seg (ONNX/NCNN)
"""

from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np
import time

# ---------------- CONFIG ----------------
# MODEL_PATH = "yoloreferpaperlatest.onnx"
MODEL_PATH = "yoloreferpaperlatest_ncnn_model"
CONF_THRESHOLD = 0.7        # try and error
AREA_CONVERSION_FACTOR = 0.00005125  # cm² per pixel — your calibrated ratio
CAPTURE_SIZE = (1280, 720)

# Confirmed from data.yaml — index position = class ID
CUSTOM_NAMES = {
    0: 'mild',
    1: 'normal',
    2: 'severe'
}
# Color per class for the overlay box/text (BGR)
CLASS_COLORS = {
    0: (0, 165, 255),   # mild - orange
    1: (0, 255, 0),      # normal - green
    2: (0, 0, 255)        # severe - red
}
# -----------------------------------------

print("Loading model...")
model = YOLO(MODEL_PATH, task='segment')
print("Model loaded.")

print("Starting camera...")
picam2 = Picamera2()
# NOTE: picamera2's "RGB888" format actually returns data in BGR byte order.
# Feed it straight into OpenCV with no cvtColor conversion — this matches
# the confirmed-working camera test and gives true colors.
config = picam2.create_preview_configuration(
    main={"size": CAPTURE_SIZE, "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
print("Camera ready. Press 'q' to quit.")

# Benchmarking tracking variables
all_inference_times = []
rolling_inference_times = []

try:
    while True:
        frame_bgr = picam2.capture_array()  # already correct order, no conversion needed

        # --- Inference Timing ---
        t0 = time.perf_counter()
        results = model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)
        t1 = time.perf_counter()
        
        inference_ms = (t1 - t0) * 1000
        all_inference_times.append(inference_ms)
        
        # Maintain a rolling window for live display smoothness
        rolling_inference_times.append(inference_ms)
        if len(rolling_inference_times) > 30:
            rolling_inference_times.pop(0)
        avg_rolling_ms = sum(rolling_inference_times) / len(rolling_inference_times)
        # ------------------------

        r = results[0]

        # Draw masks/boxes without ultralytics' built-in text labels — we draw our own below,
        # using our confirmed class-name mapping, so we never depend on the model's internal names.
        plotted_img = r.plot(labels=False)

        orig_shape = r.orig_shape
        pixel_count = 0
        wound_area_cm2 = 0.0
        num_wounds = 0
        detected_classes = []

        if r.masks is not None:
            masks_data = r.masks.data.cpu().numpy() if hasattr(r.masks.data, "cpu") else r.masks.data
            num_wounds = len(masks_data)
            binary_mask = np.zeros(orig_shape, dtype=np.uint8)
            for mask in masks_data:
                resized_mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
                binary_mask = np.logical_or(binary_mask, resized_mask).astype(np.uint8) * 255
            pixel_count = int(np.sum(binary_mask == 255))
            wound_area_cm2 = pixel_count * AREA_CONVERSION_FACTOR

        # Per-detection severity label, drawn at each box's top-left corner
        if r.boxes is not None and len(r.boxes) > 0:
            classes_raw = r.boxes.cls.tolist()
            confs = r.boxes.conf.tolist()
            xyxy = r.boxes.xyxy.tolist()

            for cls_id, conf, box in zip(classes_raw, confs, xyxy):
                cls_id = int(cls_id)
                x1, y1 = int(box[0]), int(box[1])

                # Safety guard: if class id is ever out of range
                cls_name = CUSTOM_NAMES.get(cls_id, f"unknown_{cls_id}")
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                detected_classes.append(cls_name)

                text = f"{cls_name} {conf:.2f}"
                cv2.putText(plotted_img, text, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Overall summary overlay (top-left of frame) including Inference Speed
        severity_summary = ", ".join(detected_classes) if detected_classes else "none"
        label = f"Wounds: {num_wounds} ({severity_summary}) | Area: {wound_area_cm2:.2f} cm2 | Inf: {avg_rolling_ms:.1f}ms"
        cv2.putText(plotted_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Wound Detection - Live", plotted_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    
    # Print final summary matching your benchmarking snippet format
    if len(all_inference_times) > 0:
        avg_inference_ms = sum(all_inference_times) / len(all_inference_times)
        print(f"\nAvg INFERENCE: {avg_inference_ms:.2f} ms | Frames Processed: {len(all_inference_times)}")
    
    print("Stopped.")