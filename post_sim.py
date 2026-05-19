import cv2
import numpy as np

# ==========================================
# 1. POSTPROCESS FUNCTIONS (YOLOv8 Logic)
# ==========================================
REG_MAX = 16
STRIDES = [8, 16, 32]
CONF_THRES = 0.25
IOU_THRES = 0.7

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dfl_decode(bbox):
    B, C, H, W = bbox.shape
    bbox = bbox.reshape(B, 4, REG_MAX, H, W)

    bbox = np.exp(bbox - bbox.max(axis=2, keepdims=True))
    bbox = bbox / bbox.sum(axis=2, keepdims=True)

    proj = np.arange(REG_MAX, dtype=np.float32)
    bbox = (bbox * proj.reshape(1,1,REG_MAX,1,1)).sum(axis=2)

    return bbox

def make_grid(H, W):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return x, y

def decode_scale(output, stride):
    B, C, H, W = output.shape

    bbox = output[:, :64, :, :]
    cls  = output[:, 64:, :, :]

    bbox = dfl_decode(bbox)

    grid_x, grid_y = make_grid(H, W)
    grid_x = grid_x.reshape(1, 1, H, W)
    grid_y = grid_y.reshape(1, 1, H, W)

    l = bbox[:, 0]
    t = bbox[:, 1]
    r = bbox[:, 2]
    b = bbox[:, 3]

    x1 = (grid_x - l) * stride
    y1 = (grid_y - t) * stride
    x2 = (grid_x + r) * stride
    y2 = (grid_y + b) * stride

    boxes = np.stack([x1, y1, x2, y2], axis=-1)
    scores = sigmoid(cls)

    return boxes.reshape(-1,4), scores.reshape(-1)

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])

    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)

    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

    return inter / (area1 + area2 - inter + 1e-6)

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]

    return keep

def postprocess(outputs, debug_mode=False): # Add a debug_mode flag
    all_boxes, all_scores = [], []

    for out, stride in zip(outputs, STRIDES):
        boxes, scores = decode_scale(out, stride)

        mask = scores > CONF_THRES
        all_boxes.append(boxes[mask])
        all_scores.append(scores[mask])

    if len(all_boxes) == 0:
        return np.array([]), np.array([])

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    
    # --- DEBUGGING CHANGE ---
    if debug_mode:
        print(f"DEBUG: Found {len(boxes)} raw boxes before NMS.")
        return boxes, scores # Return EVERYTHING before NMS

    # --- Original Logic ---
    keep = nms(boxes, scores, IOU_THRES)
    return boxes[keep], scores[keep]

# ==========================================
# 2. IMAGE PREPARATION
# ==========================================
img_path = "real2.jpg"  # <-- Update this to your image name
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Re-calculate the exact same scale used in your preprocessing
input_size = 640
scale = min(input_size / h, input_size / w)

# ==========================================
# 3. LOAD INT8 .NPZ & FORMAT
# ==========================================
npz_path = "selected_outputs_real2.npz"  # <-- Update this to your NPZ name
loaded_data = np.load(npz_path)

# Extract and sort (largest to smallest)
raw_outputs = [loaded_data[key] for key in loaded_data.files]
outputs_sorted = sorted(raw_outputs, key=lambda x: x.size, reverse=True)

# ---> UPDATE THESE WITH YOUR MODEL'S ACTUAL OUTPUT SCALES <---
DEQUANT_SCALES = [0.25, 0.25, 0.25] 

formatted_outputs = []
CHANNELS = 65

print("\n=== FORMATTING INT8 TENSORS ===")
for out, dequant_scale in zip(outputs_sorted, DEQUANT_SCALES):
    
    # Cast to float and dequantize
    out_float = out.astype(np.float32) * dequant_scale
    
    # Reshape from Flat -> HWC -> CHW -> BCHW
    spatial_dim = int(np.sqrt(out_float.size / CHANNELS))
    out_hwc = out_float.reshape((spatial_dim, spatial_dim, CHANNELS))
    out_chw = out_hwc.transpose(2, 0, 1)
    out_bchw = np.expand_dims(out_chw, axis=0)
    
    formatted_outputs.append(out_bchw)
    print(f"Flat int8 {out.shape} -> Float32 {out_bchw.shape}")

# ==========================================
# 4. RUN INFERENCE & RESCALE
# ==========================================
boxes, scores = postprocess(formatted_outputs, debug_mode=False)
num_boxes = len(boxes)

print(f"\nTotal boxes detected: {num_boxes}")

if num_boxes > 0:
    # This is now the correct way to scale back to the original image
    # because the input image's aspect ratio was preserved!
    boxes /= scale

# ==========================================
# 5. DRAW ON IMAGE
# ==========================================
for box, score in zip(boxes, scores):
    x1, y1, x2, y2 = map(int, box)

    # Dynamic thickness/font scale for very large high-res images
    text_scale = max(0.5, w / 1500.0)
    thickness = max(1, int(w / 1000.0))
    box_thick = max(2, int(w / 800.0))

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), box_thick)
    cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), thickness)

# Draw Total Count
cv2.putText(img, f"Total Boxes: {num_boxes}", (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, max(1.0, w / 1000.0), (0, 0, 255), max(2, int(w / 800.0)))

# # ==========================================
# # 6. DISPLAY IMAGE NICELY
# # ==========================================
# cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

# # Calculate display window size so it doesn't overflow your monitor
# disp_scale = min(800 / h, 1000 / w)
# if disp_scale < 1.0:
#     cv2.resizeWindow("Result", int(w * disp_scale), int(h * disp_scale))
# else:
#     cv2.resizeWindow("Result", w, h)

# cv2.imshow("Result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("result.jpg", img)
# ==========================================
# 6. START LOCAL ETHERNET WEB SERVER
# ==========================================
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
import threading

PORT = 8080

# Get local Ethernet IP
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

print("\n===================================")
print("IMAGE AVAILABLE OVER ETHERNET")
print(f"http://{local_ip}:{PORT}/result.jpg")
print("===================================\n")

# Create server
server = HTTPServer(("0.0.0.0", PORT), SimpleHTTPRequestHandler)

# Run server in background thread
server_thread = threading.Thread(
    target=server.serve_forever
)

server_thread.start()

print("Server running in background.")
print("Script finished normally.")
# Keep server alive until you press Enter
input("Press Enter to stop server...\n")

server.shutdown()
server.server_close()
