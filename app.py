import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

print("Loading YOLO11-Seg model...")
model = YOLO("yolo11n-seg.pt")
print("Model loaded successfully!")

CLASS_WEIGHTS = {
    "cable": 2.5,
    "wire": 2.5,
    "book": 1.6,
    "cup": 1.0,
    "bottle": 1.0,
    "cell phone": 1.2,
    "scissors": 1.3,
    "laptop": 0.3,
    "keyboard": 0.2,
    "mouse": 0.2,
    "tv": 0.2,
    "default": 0.8
}

def generate_ergonomic_mask(width, height, alpha=1.2):
    """Focuses cognitive friction on the center-bottom workspace."""
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xx, yy = np.meshgrid(x, y)
    
    mu_x = width * 0.5
    mu_y = height * 0.85
    sigma_x = width * 0.35
    sigma_y = height * 0.25
    
    gaussian_focus = np.exp(-0.5 * (((xx - mu_x) / sigma_x) ** 2 + ((yy - mu_y) / sigma_y) ** 2))
    return 1.0 + alpha * gaussian_focus

def compute_cognitive_field(image, detections, alpha_overlay=0.55):
    h, w = image.shape[:2]
    accumulator = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det.get("label", "default")
        conf = det.get("confidence", 1.0)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        sigma_x = max((x2 - x1) / 2.2, 1.0)
        sigma_y = max((y2 - y1) / 2.2, 1.0)

        weight = CLASS_WEIGHTS.get(label, CLASS_WEIGHTS["default"]) * conf

        rx1, rx2 = max(0, int(cx - 3 * sigma_x)), min(w, int(cx + 3 * sigma_x))
        ry1, ry2 = max(0, int(cy - 3 * sigma_y)), min(h, int(cy + 3 * sigma_y))

        if rx2 <= rx1 or ry2 <= ry1:
            continue

        gx, gy = np.meshgrid(np.arange(rx1, rx2), np.arange(ry1, ry2))
        patch = np.exp(-0.5 * (((gx - cx) / sigma_x) ** 2 + ((gy - cy) / sigma_y) ** 2))
        accumulator[ry1:ry2, rx1:rx2] += weight * patch

    ergo_mask = generate_ergonomic_mask(w, h)
    accumulator *= ergo_mask

    max_val = np.max(accumulator)
    norm_map = ((accumulator / max_val) * 255.0).astype(np.uint8) if max_val > 0 else accumulator.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_TURBO)
    bg_mask = (norm_map < 25)[:, :, np.newaxis]
    heatmap_color = np.where(bg_mask, image, heatmap_color)

    blended = cv2.addWeighted(image, 1.0 - alpha_overlay, heatmap_color, alpha_overlay, 0)
    return blended

def calculate_zen_score(detections, free_surface_ratio, img_height):
    """
    Precision Zen Score:
    - Anchors directly to free surface ratio (91% -> ~82 base pts).
    - Tolerates 1-2 work accessories (like a notepad or planner).
    - Penalizes real clutter (cables, trash, or 3+ loose books/papers).
    """
    # 1. Base points directly from free usable space
    score = float(free_surface_ratio * 90.0)

    # 2. Desk reach zone detections (y > 45% of image height)
    desk_items = [d for d in detections if d["bbox"][3] > (img_height * 0.45)]
    
    # Critical clutter (cables, trash)
    harsh_clutter = [d for d in desk_items if d["label"] in ["cable", "wire", "trash"]]
    score -= len(harsh_clutter) * 8.0

    # Loose documents/books: allow up to 2 items (e.g. planner/binder) without penalty
    books_papers = [d for d in desk_items if d["label"] in ["book", "paper"]]
    if len(books_papers) > 2:
        score -= (len(books_papers) - 2) * 7.0

    # Mild single penalty for beverage near input gear
    has_cup = any(d["label"] in ["cup", "bottle"] for d in desk_items)
    if has_cup:
        score -= 3.0

    # Bonus if surface is wide open and free of messy clutter
    if free_surface_ratio >= 0.85 and len(harsh_clutter) == 0 and len(books_papers) <= 2:
        score += 8.0

    return int(np.clip(score, 18, 95))

def render_detections_and_masks(image, yolo_result):
    annotated = image.copy()
    h, w = image.shape[:2]

    if yolo_result.masks is not None:
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        colors = [(56, 189, 248), (167, 139, 250), (74, 222, 128), (250, 204, 21), (248, 113, 113)]
        raw_masks = yolo_result.masks.data.cpu().numpy()
        for idx, m in enumerate(raw_masks):
            color = colors[idx % len(colors)]
            resized_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_overlay[resized_mask > 0.5] = color
        annotated = cv2.addWeighted(annotated, 0.75, mask_overlay, 0.25, 0)

    if yolo_result.boxes is not None:
        boxes = yolo_result.boxes.xyxy.cpu().numpy()
        confs = yolo_result.boxes.conf.cpu().numpy()
        classes = yolo_result.boxes.cls.cpu().numpy()
        names = yolo_result.names

        for box, conf, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls_id)]} {int(conf * 100)}%"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (56, 189, 248), 2)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 8, y1), (15, 23, 42), -1)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 8, y1), (56, 189, 248), 1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 245, 249), 1, cv2.LINE_AA)

    return annotated

def generate_action_items(detections, free_ratio, img_w, img_h):
    items = []
    desk_detections = [d for d in detections if d["bbox"][3] > (img_h * 0.45)]

    paper_items = [d for d in desk_detections if d["label"] in ["book", "paper"]]
    if len(paper_items) >= 2:
        items.append({
            "priority": len(items) + 1,
            "title": "Loose Paper & Document Clusters",
            "desc": f"Detected {len(paper_items)} document/paper stacks on the desktop. File or align them into a tray.",
            "impact": "High"
        })

    cables = [d for d in desk_detections if d["label"] in ["cable", "wire"]]
    if len(cables) >= 1:
        items.append({
            "priority": len(items) + 1,
            "title": "Exposed Cable Sprawl",
            "desc": f"Detected {len(cables)} unmanaged cords in the primary work zone. Route them behind monitors or risers.",
            "impact": "High"
        })

    if free_ratio < 0.65:
        items.append({
            "priority": len(items) + 1,
            "title": "Restricted Work Envelope",
            "desc": f"Only {int(free_ratio * 100)}% usable surface area remaining. Clear secondary items off the central desktop.",
            "impact": "Medium"
        })

    left_items = [d for d in desk_detections if (d["bbox"][0] + d["bbox"][2]) / 2 < img_w * 0.45]
    right_items = [d for d in desk_detections if (d["bbox"][0] + d["bbox"][2]) / 2 > img_w * 0.55]

    if len(right_items) >= 3 and len(right_items) > len(left_items) * 1.5:
        items.append({
            "priority": len(items) + 1,
            "title": "Right-Side Clutter Density",
            "desc": "Heavy pile accumulation on your right side. Shift non-essential items away from your primary reach zone.",
            "impact": "Medium"
        })
    elif len(left_items) >= 3 and len(left_items) > len(right_items) * 1.5:
        items.append({
            "priority": len(items) + 1,
            "title": "Left-Side Clutter Density",
            "desc": "Clutter concentrated on the left. Relocate stacked items to open up visual breathing room.",
            "impact": "Medium"
        })

    cups = [d for d in desk_detections if d["label"] in ["cup", "bottle"]]
    electronics = [d for d in desk_detections if d["label"] in ["keyboard", "laptop"]]
    if len(cups) >= 1 and len(electronics) >= 1:
        items.append({
            "priority": len(items) + 1,
            "title": "Beverage Proximity Risk",
            "desc": "Drink containers detected close to input devices. Place on a coaster outside the primary reach zone.",
            "impact": "Low"
        })

    if not items:
        items.append({
            "priority": 1,
            "title": "Workspace Clear & Focused",
            "desc": "No major clutter hotspots detected. Ready for deep work!",
            "impact": "Low"
        })

    return items

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    h, w = image.shape[:2]

    results = model(image, conf=0.20)[0]

    detections = []
    total_mask_area = 0

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        names = results.names

        for box, conf, cls_id in zip(boxes, confs, classes):
            detections.append({
                "bbox": [float(b) for b in box],
                "label": names[int(cls_id)],
                "confidence": float(conf)
            })

    if results.masks is not None:
        raw_masks = results.masks.data.cpu().numpy()
        combined_mask = np.any(raw_masks > 0.5, axis=0)
        total_mask_area = np.sum(combined_mask)

    total_pixels = h * w
    occupied_ratio = total_mask_area / max(total_pixels, 1)
    free_surface_ratio = max(0.0, 1.0 - occupied_ratio)

    blended_heatmap = compute_cognitive_field(image, detections)
    _, buffer_heatmap = cv2.imencode(".jpg", blended_heatmap)
    heatmap_b64 = base64.b64encode(buffer_heatmap).decode("utf-8")

    zen_score = calculate_zen_score(detections, free_surface_ratio, h)

    detected_annotated = render_detections_and_masks(image, results)
    _, buffer_boxes = cv2.imencode(".jpg", detected_annotated)
    boxes_b64 = base64.b64encode(buffer_boxes).decode("utf-8")

    action_items = generate_action_items(detections, free_surface_ratio, w, h)

    return jsonify({
        "zen_score": zen_score,
        "free_surface_ratio": round(float(free_surface_ratio), 2),
        "total_objects": len(detections),
        "action_items": action_items,
        "heatmap": f"data:image/jpeg;base64,{heatmap_b64}",
        "detections_view": f"data:image/jpeg;base64,{boxes_b64}"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)