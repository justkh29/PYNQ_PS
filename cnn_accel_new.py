import numpy as np
import time
import cv2
import struct
from pynq import Overlay, allocate
from numpy.lib.stride_tricks import sliding_window_view
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
import threading
import sys
# ==============================================================================
# YOLO Post-Processing Constants 
# ==============================================================================
REG_MAX = 16
STRIDES = [8, 16, 32]
CONF_THRES = 0.25
IOU_THRES = 0.7
CHANNELS = 65
DEQUANT_SCALES = [0.25, 0.25, 0.25]
# ==============================================================================
# HLS Constants 
# ==============================================================================
LEAKY_RELU_MULT = 13    # Example value
LEAKY_RELU_SHIFT = 7    # Example value

# ==============================================================================
# Utility Functions
# ==============================================================================


class Logger(object):
    def __init__(self, filename="inference_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes to the file immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()
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

def postprocess(outputs, debug_mode=False):
    all_boxes, all_scores = [], []
    for out, stride in zip(outputs, STRIDES):
        boxes, scores = decode_scale(out, stride)
        mask = scores > CONF_THRES
        all_boxes.append(boxes[mask])
        all_scores.append(scores[mask])

    if len(all_boxes) == 0 or len(np.concatenate(all_boxes)) == 0:
        return np.array([]), np.array([])

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    
    if debug_mode: return boxes, scores 

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    cv_boxes = np.column_stack((boxes[:, 0], boxes[:, 1], widths, heights)).tolist()
    cv_scores = scores.tolist()

    indices = cv2.dnn.NMSBoxes(cv_boxes, cv_scores, CONF_THRES, IOU_THRES)
    if len(indices) > 0:
        keep = indices.flatten()
        return boxes[keep], scores[keep]
    else:
        return np.array([]), np.array([])
def preprocess_quantized(img, size=640):
    h, w = img.shape[:2]

    # 1. Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Calculate scale and resize while KEEPING aspect ratio
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))

    # 3. Create a black square canvas and paste image in top-left
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized

    # 4. Quantize to INT8
    y_scale = 0.015625
    y_zero_point = 0

    img_normalized = canvas.astype(np.float32) / 255.0
    img_quantized = np.round(img_normalized / y_scale) + y_zero_point
    
    img_int8 = np.clip(img_quantized, -128, 127).astype(np.int8)

    # Return the image exactly as [H, W, C] (Shape: 640, 640, 3)
    return img_int8

def slice_feature_map(src_buf, H, W, total_channels):
    cpu_start_time = time.perf_counter()
    tensor_3d = src_buf.reshape((H, W, total_channels))
    half_ch = total_channels // 2
    part1 = tensor_3d[:, :, :half_ch]
    part2 = tensor_3d[:, :, half_ch:]
    cpu_end_time = time.perf_counter()
    print(f"    [CPU] Slice Done in {(cpu_end_time - cpu_start_time):.6f} sec")
    return part1.ravel(), part2.ravel()

def concat_feature_maps(buf1, buf2, H, W, ch1, ch2):
    cpu_start_time = time.perf_counter()
    t1 = buf1.reshape((H, W, ch1))
    t2 = buf2.reshape((H, W, ch2))
    t_concat = np.concatenate((t1, t2), axis=2)
    cpu_end_time = time.perf_counter()
    print(f"    [CPU] Concat Done in {(cpu_end_time - cpu_start_time):.6f} sec")
    return t_concat.ravel() # Also changed flatten() to ravel()

def resize_nearest_cpu(input_flat, H, W, C, scale=2):
    print(f"    [CPU] Running Resize (Nearest, scale={scale}) (H={H}, W={W}, C={C})...")
    cpu_start_time = time.perf_counter()

    x = input_flat.reshape(H, W, C)

    out = np.repeat(x, scale, axis=0)
    out = np.repeat(out, scale, axis=1)

    cpu_end_time = time.perf_counter()
    print(f"    [CPU] Resize Done in {(cpu_end_time - cpu_start_time):.6f} sec")

    # 3. Flatten and return
    return out.ravel()

def print_tensor_stats(name, tensor):
    """
    Print useful debug statistics for a tensor.
    """

    flat = tensor.flatten()

    print("\n=================================================")
    print(f"TENSOR STATS: {name}")
    print("=================================================")

    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")

    print(f"\nMin value:  {flat.min()}")
    print(f"Max value:  {flat.max()}")

    print(f"\nMean value: {flat.mean():.6f}")

    # Count specific values
    num_zero = np.sum(flat == 0)
    num_pos1 = np.sum(flat == 1)
    num_neg1 = np.sum(flat == -1)

    print("\nValue counts:")
    print(f"0  count : {num_zero}")
    print(f"1  count : {num_pos1}")
    print(f"-1 count : {num_neg1}")

    # Optional saturation debug
    num_pos_sat = np.sum(flat == 127)
    num_neg_sat = np.sum(flat == -128)

    print("\nSaturation counts:")
    print(f"+127 count : {num_pos_sat}")
    print(f"-128 count : {num_neg_sat}")
def cpu_conv2d_layer(
    input_flat, weights, bias, params, 
    residual_flat=None, apply_activation=True, is_output=False, bias_left_shift=0
):
    W, H, Cin, Cout, K, stride, shift_val = params

    pad = 1 if K == 3 else 0
    out_w = (W - K + 2 * pad) // stride + 1
    out_h = (H - K + 2 * pad) // stride + 1

    print(f"    [CPU] Running NumPy Conv2D (Cin={Cin}, Cout={Cout}, K={K}x{K})...")
    cpu_start_time = time.perf_counter()

# =========================================================
    # OPTIMIZATION: 1x1 Convolution Fast-Path
    # =========================================================
    if K == 1 and stride == 1:
        # Cast to float32 to trigger optimized OpenBLAS/NEON matmul
        X_col = input_flat.reshape(H * W, Cin).astype(np.float32)
        W_reshaped = weights.reshape(Cout, Cin).T.astype(np.float32)
        
        # Matrix Multiply and instantly cast back to int32
        Y_col = np.dot(X_col, W_reshaped).astype(np.int32)
    
    # =========================================================
    # Standard 3x3 (or stride 2) path using im2col
    # =========================================================
    else:
        x = input_flat.reshape(H, W, Cin) 
        if pad > 0:
            x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
        else:
            x_pad = x

        windows = sliding_window_view(x_pad, (K, K, Cin))
        windows = windows[::stride, ::stride, 0, :, :, :]
        
        # Cast to float32 here as well
        X_col = windows.reshape(out_h * out_w, -1).astype(np.float32)
        W_reshaped = weights.reshape(Cout, Cin, K, K).transpose(0, 2, 3, 1).reshape(Cout, -1).T.astype(np.float32)

        Y_col = np.dot(X_col, W_reshaped).astype(np.int32)

    # =========================================================
    # Post-Processing (In-Place)
    # =========================================================
    bias_shifted = bias.astype(np.int32) << bias_left_shift
    Y_col += bias_shifted

    if apply_activation:
        neg_mask = Y_col < 0
        np.multiply(Y_col, LEAKY_RELU_MULT, out=Y_col, where=neg_mask)
        np.right_shift(Y_col, LEAKY_RELU_SHIFT, out=Y_col, where=neg_mask)

    np.right_shift(Y_col, shift_val, out=Y_col)

    if residual_flat is not None:
        res_col = residual_flat.reshape(out_h * out_w, Cout).astype(np.int32)
        Y_col += res_col

    if is_output:
        out_final = Y_col.astype(np.int16)
    else:
        np.clip(Y_col, -128, 127, out=Y_col)
        out_final = Y_col.astype(np.int8)

    out_final = out_final.reshape(out_h, out_w, Cout)

    cpu_end_time = time.perf_counter()
    print(f"    [CPU] Done in {(cpu_end_time - cpu_start_time):.6f} sec")

    return out_final.flatten()
def maxpool_2d_cpu(input_flat, H, W, C, K=5, stride=1, pad=2):
    print(f"    [CPU] Running MaxPool2D (H={H}, W={W}, C={C}, K={K})...")
    cpu_start_time = time.perf_counter()

    x = input_flat.reshape(H, W, C)

    if pad > 0:
        x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=-128)
    else:
        x_pad = x

    windows = sliding_window_view(x_pad, (K, K, C))
    windows = windows[::stride, ::stride, 0, :, :, :]

    out = np.max(windows, axis=(2, 3))

    cpu_end_time = time.perf_counter()
    print(f"    [CPU] MaxPool Done in {(cpu_end_time - cpu_start_time):.6f} sec")
    return out.ravel()

def run_fused_sppf_cpu(input_flat, H, W, C, K=5):
    """
    Fuses 3 sequential 5x5 MaxPools and 3 Channel-Concatenations into a single,
    highly optimized operation leveraging the KV260's ARM NEON cache locality.
    """
    # 1. Reshape the 1D flat input back to 3D (H, W, C)
    cpu_start_time = time.perf_counter()
    x = input_flat.reshape(H, W, C)
    
    # 2. Local helper for separable MaxPool to maximize NEON vectorization
    def fast_maxpool_5x5(tensor):
        # Pad spatial dimensions with -128 (standard for int8, change to 0 if float32)
        padded = np.pad(tensor, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=-128)
        
        # Pool across Height (Separable Part 1)
        max_h = padded[0:H, :, :].copy()
        for i in range(1, K):
            np.maximum(max_h, padded[i:i+H, :, :], out=max_h)
            
        # Pool across Width (Separable Part 2)
        out = max_h[:, 0:W, :].copy()
        for j in range(1, K):
            np.maximum(out, max_h[:, j:j+W, :], out=out)
            
        return out

    # 3. Chain the 3 pooling steps (Keeping memory hot in L1/L2 ARM Cache)
    pool1 = fast_maxpool_5x5(x)
    pool2 = fast_maxpool_5x5(pool1)
    pool3 = fast_maxpool_5x5(pool2)
    
    # 4. Perform a 4-way Concatenation in a single C-level memory copy
    # Original Input + Pool 1 + Pool 2 + Pool 3
    out = np.concatenate([x, pool1, pool2, pool3], axis=2)
    cpu_end_time = time.perf_counter()
    print(f"    [CPU] SPPF Done in {(cpu_end_time - cpu_start_time):.6f} sec")
    # Return as 1D array to match your engine's expectations
    return out.ravel()

# ==============================================================================
# YOLOv8 Engine
# ==============================================================================
class YOLOv8Engine:

    # Giao diện điều khiển (s_axi_CTRL)
    REG_CTRL_AP_CTRL             = 0x00
    REG_CTRL_DESCRIPTOR_1        = 0x10
    REG_CTRL_DESCRIPTOR_2        = 0x14
    REG_CTRL_DESCRIPTOR_3        = 0x18
    REG_CTRL_DESCRIPTOR_4        = 0x1C
    REG_CTRL_FMAP_OFFSET         = 0x24
    REG_CTRL_RESIDUAL_OFFSET     = 0x2C
    REG_CTRL_WEIGHT_OFFSET       = 0x34
    REG_CTRL_PACKET_OFFSET       = 0x3C
    REG_CTRL_BIAS_OFFSET         = 0x44
    REG_CTRL_START_ACCEL         = 0x4C

    # Giao diện cấp phát con trỏ bộ nhớ (s_axi_control)
    REG_MEM_FMAP_IN              = 0x10
    REG_MEM_RESIDUAL_IN          = 0x1C
    REG_MEM_WEIGHT_IN            = 0x28
    REG_MEM_DDR_OUT              = 0x34
    REG_MEM_BIAS_IN              = 0x40

    def __init__(self, bitstream_path, target_mhz=None):
        print("Loading Overlay...")
        from pynq import Clocks
        self.overlay = Overlay(bitstream_path)
        
        if target_mhz is not None:
            Clocks.fclk0_mhz = target_mhz
            print(f"PL Clock 0 set to: {Clocks.fclk0_mhz:.2f} MHz")

        self.cnn_ip = self.overlay.cnn_accelerator_top_0
        
        # Vitis HLS sẽ ánh xạ các bundle thành các attribute phân biệt
        self.ctrl_bus = self.cnn_ip.CTRL if hasattr(self.cnn_ip, 'CTRL') else self.cnn_ip
        self.mem_bus = self.cnn_ip.control if hasattr(self.cnn_ip, 'control') else self.cnn_ip
        
        MAX_FMAP_BYTES = 16 * 1024 * 1024  
        MAX_WEIGHT_BYTES = 8 * 1024 * 1024

        print("Pre-allocating CMA buffers...")
        self.cma_in_pixels = allocate(shape=(MAX_FMAP_BYTES,), dtype=np.int8)
        self.cma_out_pixels = allocate(shape=(MAX_FMAP_BYTES,), dtype=np.int8)
        self.cma_in_weights = allocate(shape=(MAX_WEIGHT_BYTES,), dtype=np.int8)
        self.cma_in_residual = allocate(shape=(MAX_FMAP_BYTES,), dtype=np.int8)
        
        # Buffer mới dành riêng cho giao thức Burst Read Bias
        self.cma_in_bias = allocate(shape=(4096,), dtype=np.int8)
        
        # --- CACHES FOR EXTREME SPEED ---
        self.weight_cache = {}
        self.bias_cache = {}
        self.config_cache = {}

    def _write_64bit_ptr(self, bus, reg_addr, ptr_val):
        bus.write(reg_addr, ptr_val & 0xFFFFFFFF)
        bus.write(reg_addr + 4, (ptr_val >> 32) & 0xFFFFFFFF)

    def run_layer(self, layer_name, input_fmap, weights, bias, params, residual_fmap=None, bias_shift=0):
        t_start = time.perf_counter()

        input_prep_time = 0.0
        weight_time_total = 0.0
        residual_time_total = 0.0
        hw_time_total = 0.0
        unpack_time_total = 0.0

        def print_profiling():
            total_time = time.perf_counter() - t_start
            print(f"\n[{layer_name}] Profiling:")
            print(f"  Input Prep     : {input_prep_time*1000:.3f} ms")
            print(f"  Weight Process : {weight_time_total*1000:.3f} ms")
            print(f"  Residual Pack  : {residual_time_total*1000:.3f} ms")
            print(f"  HW Execution   : {hw_time_total*1000:.3f} ms")
            print(f"  Output Unpack  : {unpack_time_total*1000:.3f} ms")
            print(f"  TOTAL          : {total_time*1000:.3f} ms\n")
        
        W, H, Cin, Cout, K, stride, requant = params
        has_residual = 1 if residual_fmap is not None else 0

        if K == 3 and stride == 1: mode = 0          
        elif K == 3 and stride == 2: mode = 1        
        else: mode = 2                               

        out_w = (W + stride - 1) // stride if stride == 2 else W
        out_h = (H + stride - 1) // stride if stride == 2 else H
        pad = 1 if K == 3 else 0

        max_bram_addr = 8192
        if mode == 0:
            max_cout_step = max_bram_addr // Cin
        else:
            k_max = 9 if mode == 1 else 1
            max_cout_step = (max_bram_addr // (k_max * Cin)) * 16

        max_cout_step = (max_cout_step // 16) * 16
        if max_cout_step == 0: max_cout_step = 16
        cout_step = min(Cout, max_cout_step)

        # =====================================================================
        # 1. INPUT FEATURE MAP 
        # =====================================================================
        t0_in = time.perf_counter()
        
        if Cin == 3:
            padded = np.pad(input_fmap.reshape(H, W, 3), ((0,0), (0,0), (0, 13)), mode='constant')
            in_size = W * H * 16
            np.copyto(self.cma_in_pixels[:in_size], padded.ravel())
        else:
            in_size = W * H * Cin
            np.copyto(self.cma_in_pixels[:in_size], input_fmap) 
            
        self.cma_in_pixels[:in_size].flush()

        needs_chunking = cout_step < Cout
        if needs_chunking:
            final_output = np.empty((out_h, out_w, Cout), dtype=np.int8)

        input_prep_time = time.perf_counter() - t0_in

        # =====================================================================
        # 2. CHUNK PROCESSING LOOP
        # =====================================================================
        for cout_start in range(0, Cout, cout_step):
            current_cout = min(cout_step, Cout - cout_start)
            padded_current_cout = ((current_cout + 15) // 16) * 16
            cache_key = f"{layer_name}_{cout_start}"

            # --- WEIGHTS PREP ---
            t0_wt = time.perf_counter()
            if cache_key in self.weight_cache:
                weights_chunk = self.weight_cache[cache_key]
            else:
                if mode == 0:
                    w_4d = weights.reshape((Cin, ((Cout + 3) // 4) * 4, 4, 4))
                    padded_wino_cout = ((current_cout + 3) // 4) * 4
                    weights_chunk = w_4d[:, cout_start : cout_start + padded_wino_cout, :, :].flatten()
                else:
                    total_cout_blocks = (Cout + 15) // 16
                    w_5d = weights.reshape((total_cout_blocks, Cin, K, K, 16))
                    start_block = cout_start // 16
                    num_blocks = (current_cout + 15) // 16
                    weights_chunk = w_5d[start_block : start_block + num_blocks, :, :, :, :].flatten()
                self.weight_cache[cache_key] = weights_chunk

            np.copyto(self.cma_in_weights[:len(weights_chunk)], weights_chunk)
            self.cma_in_weights[:len(weights_chunk)].flush()
            out_size = out_h * out_w * padded_current_cout
            weight_time_total += (time.perf_counter() - t0_wt)

            # --- RESIDUAL PREP ---
            t0_res = time.perf_counter()
            residual_size = 0
            if has_residual:
                res_3d = residual_fmap.reshape(out_h, out_w, Cout)[:, :, cout_start:cout_start + current_cout]
                
                if mode == 0:
                    tiles_x = max(1, (out_w + 1) // 2)
                    tiles_y = max(1, (out_h + 1) // 2)
                    pad_h = (tiles_y * 2) - out_h
                    pad_w = (tiles_x * 2) - out_w
                    if pad_h > 0 or pad_w > 0:
                        res_3d = np.pad(res_3d, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                    if padded_current_cout > current_cout:
                        res_3d = np.pad(res_3d, ((0, 0), (0, 0), (0, padded_current_cout - current_cout)), mode='constant')
                    
                    res_chunk = res_3d.reshape(tiles_y, 2, tiles_x, 2, padded_current_cout).transpose(0, 2, 1, 3, 4).flatten()
                else:
                    if padded_current_cout > current_cout:
                        res_3d = np.pad(res_3d, ((0,0), (0,0), (0, padded_current_cout - current_cout)), mode='constant')
                    res_chunk = res_3d.flatten()

                residual_size = len(res_chunk)
                np.copyto(self.cma_in_residual[:residual_size], res_chunk)
                self.cma_in_residual[:residual_size].flush()
            residual_time_total += (time.perf_counter() - t0_res)

            # --- BIAS PREP ---
            if cache_key not in self.bias_cache:
                b_arr = bias[cout_start:cout_start + current_cout]
                self.bias_cache[cache_key] = b_arr.astype(np.int8)

            bias_chunk = self.bias_cache[cache_key]
            bias_size = len(bias_chunk)
            np.copyto(self.cma_in_bias[:bias_size], bias_chunk)
            self.cma_in_bias[:bias_size].flush()

            # =====================================================================
            # 3. HARDWARE EXECUTION
            # =====================================================================
            t0_hw = time.perf_counter()
            
            # Khởi tạo con trỏ Memory (s_axi_control)
            self._write_64bit_ptr(self.mem_bus, self.REG_MEM_FMAP_IN, self.cma_in_pixels.physical_address)
            self._write_64bit_ptr(self.mem_bus, self.REG_MEM_WEIGHT_IN, self.cma_in_weights.physical_address)
            self._write_64bit_ptr(self.mem_bus, self.REG_MEM_RESIDUAL_IN, self.cma_in_residual.physical_address)
            self._write_64bit_ptr(self.mem_bus, self.REG_MEM_DDR_OUT, self.cma_out_pixels.physical_address)
            self._write_64bit_ptr(self.mem_bus, self.REG_MEM_BIAS_IN, self.cma_in_bias.physical_address)

            # Đóng gói và ghi LayerDescriptor (s_axi_CTRL)
            if cache_key not in self.config_cache:
                _type = 1 if mode == 3 else 0 
                packed_bytes = struct.pack('<BxHHHHBBBBBB', _type, W, H, Cin, current_cout, K, stride, pad, has_residual, requant, bias_shift)
                packed_bytes += b'\x00' * (16 - len(packed_bytes))
                self.config_cache[cache_key] = struct.unpack('<IIII', packed_bytes)

            w0, w1, w2, w3 = self.config_cache[cache_key]
            self.ctrl_bus.write(self.REG_CTRL_DESCRIPTOR_1, w0)
            self.ctrl_bus.write(self.REG_CTRL_DESCRIPTOR_2, w1)
            self.ctrl_bus.write(self.REG_CTRL_DESCRIPTOR_3, w2)
            self.ctrl_bus.write(self.REG_CTRL_DESCRIPTOR_4, w3)

            # Đặt Offset bằng 0 do chúng ta sử dụng Base Physical Address
            self.ctrl_bus.write(self.REG_CTRL_FMAP_OFFSET, 0)
            self.ctrl_bus.write(self.REG_CTRL_WEIGHT_OFFSET, 0)
            self.ctrl_bus.write(self.REG_CTRL_RESIDUAL_OFFSET, 0)
            self.ctrl_bus.write(self.REG_CTRL_PACKET_OFFSET, 0)
            self.ctrl_bus.write(self.REG_CTRL_BIAS_OFFSET, 0)
            
            # Gửi Start Tín Hiệu tới Controller & AXI-Lite
            self.ctrl_bus.write(self.REG_CTRL_START_ACCEL, 1)
            self.ctrl_bus.write(self.REG_CTRL_AP_CTRL, 1)

            # Đợi AP_DONE
            ctrl_reg = self.REG_CTRL_AP_CTRL
            while (self.ctrl_bus.read(ctrl_reg) & 0x2) == 0:
                pass 

            self.cma_out_pixels[:out_size].invalidate()
            hw_time_total += (time.perf_counter() - t0_hw)

            # =====================================================================
            # 4. FAST OUTPUT UNPACKING
            # =====================================================================
            t0_unpack = time.perf_counter()
            raw_out_3d = self.cma_out_pixels[:out_size].reshape(out_h, out_w, padded_current_cout)
            
            if not needs_chunking:
                out = np.empty((out_h, out_w, current_cout), dtype=np.int8)
                np.copyto(out, raw_out_3d[:, :, :current_cout])
                unpack_time_total += (time.perf_counter() - t0_unpack)
                
                print_profiling()
                return out.ravel()
            else:
                final_output[:, :, cout_start:cout_start + current_cout] = raw_out_3d[:, :, :current_cout]
                unpack_time_total += (time.perf_counter() - t0_unpack)
            
        print_profiling()
        return final_output.ravel()

    def clean_up(self):
        self.cma_in_pixels.freebuffer()
        self.cma_out_pixels.freebuffer()
        self.cma_in_weights.freebuffer()
        self.cma_in_residual.freebuffer()
        self.cma_in_bias.freebuffer()
# ==============================================================================
# Graph Runner
# ==============================================================================
class YOLOv8GraphRunner:
    def __init__(self, engine, npz_data):
        self.engine = engine
        self.npz_data = npz_data
        self.tensor_store = {}

    def delete_tensor(self, tensor_name):
        if tensor_name in self.tensor_store:
            del self.tensor_store[tensor_name]

    def run(self, network_layers, input_tensor):
        self.tensor_store["input"] = input_tensor.ravel()
        prev_output_name = "input"

        for layer in network_layers:
            op = layer["op"]

            if layer.get("print_debug"):
                print(f"\n====================================================")
                print(f"Running: {layer.get('name', op)} ({op})")
                print(f"====================================================")
            else:
                print(f"\n========== {op.upper()}: {layer.get('name', op)} ==========")

            # ------------------------------------------------------------------
            # CONV
            # ------------------------------------------------------------------
            if op == "conv":
                input_name = layer.get("input_from", prev_output_name)
                input_tensor_to_use = self.tensor_store[input_name]
                
                residual_tensor = None
                if "residual_from" in layer:
                    residual_tensor = self.tensor_store[layer["residual_from"]]
                    
                weights = self.npz_data[layer["w_key"]]
                bias = self.npz_data[layer["b_key"]]
                
                params = (
                    layer["W"], layer["H"], layer["Cin"], layer["Cout"],
                    layer["K"], layer["stride"], layer["requant"]
                )

                bias_shift = layer.get("bias_shift", 0)
                apply_activation = layer.get("apply_activation", True)
                is_output = layer.get("is_output", False)
                device = layer.get("device", "fpga")
                
                if device == "cpu":
                    output = cpu_conv2d_layer(
                        input_flat=input_tensor_to_use, weights=weights, bias=bias,
                        params=params, residual_flat=residual_tensor,
                        apply_activation=apply_activation, is_output=is_output,
                        bias_left_shift=bias_shift
                    )
                else:
                    output = self.engine.run_layer(
                        layer_name=layer["name"], input_fmap=input_tensor_to_use,
                        weights=weights, bias=bias, params=params,
                        residual_fmap=residual_tensor, bias_shift=bias_shift
                    )

                save_name = layer.get("save_as", layer["name"])
                self.tensor_store[save_name] = output
                prev_output_name = save_name

            # ------------------------------------------------------------------
            # SLICE
            # ------------------------------------------------------------------
            elif op == "slice":
                src_tensor = self.tensor_store[layer["src"]]
                out1, out2 = slice_feature_map(src_tensor, layer["H"], layer["W"], layer["C"])
                self.tensor_store[layer["save_as_1"]] = out1
                self.tensor_store[layer["save_as_2"]] = out2

            # ------------------------------------------------------------------
            # RESIZE (UPSAMPLE)
            # ------------------------------------------------------------------
            elif op == "resize":
                input_name = layer.get("input_from", prev_output_name)
                input_tensor_to_use = self.tensor_store[input_name]
                output = resize_nearest_cpu(
                    input_flat=input_tensor_to_use, H=layer["H"], W=layer["W"], C=layer["C"], scale=layer.get("scale", 2)
                )
                save_name = layer.get("save_as", layer.get("name", "resize_out"))
                self.tensor_store[save_name] = output
                prev_output_name = save_name

            # ------------------------------------------------------------------
            # CONCAT
            # ------------------------------------------------------------------
            elif op == "concat":
                buf1 = self.tensor_store[layer["src1"]]
                buf2 = self.tensor_store[layer["src2"]]
                output = concat_feature_maps(buf1, buf2, layer["H"], layer["W"], layer["C1"], layer["C2"])
                save_name = layer["save_as"]
                self.tensor_store[save_name] = output
                prev_output_name = save_name

            # ------------------------------------------------------------------
            # MAXPOOL (CPU)
            # ------------------------------------------------------------------
            elif op == "maxpool":
                input_name = layer.get("input_from", prev_output_name)
                input_tensor_to_use = self.tensor_store[input_name]
                output = maxpool_2d_cpu(
                    input_flat=input_tensor_to_use, H=layer["H"], W=layer["W"], C=layer["C"],
                    K=layer.get("K", 5), stride=layer.get("stride", 1), pad=layer.get("pad", 2)
                )
                save_name = layer.get("save_as", layer.get("name", "maxpool_out"))
                self.tensor_store[save_name] = output
                prev_output_name = save_name
            
            elif op == "sppf":
                input_name = layer.get("input_from", prev_output_name)
                input_tensor_to_use = self.tensor_store[input_name]
                
                output = run_fused_sppf_cpu(
                    input_flat=input_tensor_to_use, 
                    H=layer["H"], 
                    W=layer["W"], 
                    C=layer["C"],
                    K=layer.get("K", 5)
                )
                save_name = layer.get("save_as", layer.get("name", "sppf_out"))
                self.tensor_store[save_name] = output
                prev_output_name = save_name

            # ------------------------------------------------------------------
            # EXPLICIT DELETE OP
            # ------------------------------------------------------------------
            elif op == "delete":
                for t_name in layer.get("tensors", []): self.delete_tensor(t_name)
                continue

            else:
                raise ValueError(f"Unsupported op: {op}")
            
            # ------------------------------------------------------------------
            # INLINE TENSOR CLEANUP 
            # ------------------------------------------------------------------
            if "delete_tensors" in layer:
                for t_name in layer["delete_tensors"]: self.delete_tensor(t_name)

        return self.tensor_store[prev_output_name]


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import glob
    import os
    sys.stdout = Logger("inference_log.txt")
    # Find up to 10 JPG images in the current directory (change this path if needed)
    image_paths = glob.glob("*.jpg")
    image_paths = [p for p in image_paths if not p.startswith("result")] # exclude old results
    image_paths = image_paths[:10] 
    
    if len(image_paths) == 0:
        print("No images found! Please add some .jpg files to the directory.")
        exit()

    print(f"Found {len(image_paths)} images to process: {image_paths}")

    # 1. Initialize Engine and load Weights ONCE
    engine = YOLOv8Engine("design_2.bit", 125)
    npz_data = np.load("hardware_ready_512.npz")

    network_layers = [
        # You can paste your entire graph array here
        {
            "op": "conv",
            "name": "Conv_19",
            "w_key": "layer1_w",
            "b_key": "layer1_b",
            "W": 640,
            "H": 640,
            "Cin": 3,
            "Cout": 16,
            "K": 3,
            "stride": 2,
            "requant": 6,
            "device": "fpga",
            "bias_shift": 4
        },
        {
            "op": "conv",
            "name": "Conv_39",
            "w_key": "layer2_w",
            "b_key": "layer2_b",
            "W": 320,
            "H": 320,
            "Cin": 16,
            "Cout": 32,
            "K": 3,
            "stride": 2,
            "requant": 7,
            "device": "fpga",
            "bias_shift": 5
        },
        {
            "op": "conv",
            "name": "Conv_59",
            "save_as": "Conv_59",
            "w_key": "layer3_w",
            "b_key": "layer3_b",
            "W": 160,
            "H": 160,
            "Cin": 32,
            "Cout": 32,
            "K": 1,
            "stride": 1,
            "requant": 6,
            "device": "fpga",
            "bias_shift": 5
        },
        {
            "op": "slice",
            "name": "Slice_71",
            "src": "Conv_59",
            "H": 160,
            "W": 160,
            "C": 32,
            "save_as_1": "Concat_197",
            "save_as_2": "Conv_202"
        },
        {
            "op": "conv",
            "name": "Conv_89",
            "input_from": "Conv_202",
            "w_key": "layer4_w",
            "b_key": "layer4_b",
            "W": 160,
            "H": 160,
            "Cin": 16,
            "Cout": 16,
            "K": 3,
            "stride": 1,
            "requant": 7,
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "conv",
            "name": "Conv_109",
            "save_as": "Concat_243",
            "w_key": "layer5_w",
            "b_key": "layer5_b",
            "W": 160,
            "H": 160,
            "Cin": 16,
            "Cout": 16,
            "K": 3,
            "stride": 1,
            "requant": 8,
            "residual_from": "Conv_202",
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "concat",
            "name": "Concat_118",
            "src1": "Conv_59",
            "src2": "Concat_243",
            "H": 160,
            "W": 160,
            "C1": 32,
            "C2": 16,
            "save_as": "blob",
            "delete_tensors": [
                "Conv_59",
                "Concat_243",
                "Conv_202",
                "Concat_197"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_137",
            "input_from": "blob",
            "w_key": "layer6_w",
            "b_key": "layer6_b",
            "W": 160,
            "H": 160,
            "Cin": 48,
            "Cout": 32,
            "K": 1,
            "stride": 1,
            "requant": 6,
            "delete_tensors": [
                "blob"
            ],
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "conv",
            "name": "Conv_157",
            "w_key": "layer7_w",
            "b_key": "layer7_b",
            "W": 160,
            "H": 160,
            "Cin": 32,
            "Cout": 64,
            "K": 3,
            "stride": 2,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_177",
            "save_as": "Conv_177",
            "w_key": "layer8_w",
            "b_key": "layer8_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 64,
            "K": 1,
            "stride": 1,
            "requant": 7,
            "device": "fpga",
            "bias_shift": 5
        },
        {
            "op": "slice",
            "name": "Slice_189",
            "src": "Conv_177",
            "H": 80,
            "W": 80,
            "C": 64,
            "save_as_1": "Concat_315",
            "save_as_2": "Conv_320"
        },
        {
            "op": "conv",
            "name": "Conv_207",
            "input_from": "Conv_320",
            "w_key": "layer9_w",
            "b_key": "layer9_b",
            "W": 80,
            "H": 80,
            "Cin": 32,
            "Cout": 32,
            "K": 3,
            "stride": 1,
            "requant": 7,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "conv",
            "name": "Conv_227",
            "save_as": "Conv_367",
            "w_key": "layer10_w",
            "b_key": "layer10_b",
            "W": 80,
            "H": 80,
            "Cin": 32,
            "Cout": 32,
            "K": 3,
            "stride": 1,
            "requant": 8,
            "residual_from": "Conv_320",
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "conv",
            "name": "Conv_254",
            "input_from": "Conv_367",
            "w_key": "layer11_w",
            "b_key": "layer11_b",
            "W": 80,
            "H": 80,
            "Cin": 32,
            "Cout": 32,
            "K": 3,
            "stride": 1,
            "requant": 8,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_274",
            "save_as": "Concat_408",
            "w_key": "layer12_w",
            "b_key": "layer12_b",
            "W": 80,
            "H": 80,
            "Cin": 32,
            "Cout": 32,
            "K": 3,
            "stride": 1,
            "requant": 8,
            "residual_from": "Conv_367",
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "concat",
            "name": "Concat_283_1",
            "src1": "Conv_177",
            "src2": "Conv_367",
            "H": 80,
            "W": 80,
            "C1": 64,
            "C2": 32,
            "save_as": "Concat_283_1"
        },
        {
            "op": "concat",
            "name": "Concat_283_2",
            "src1": "Concat_283_1",
            "src2": "Concat_408",
            "H": 80,
            "W": 80,
            "C1": 96,
            "C2": 32,
            "save_as": "blob.8",
            "delete_tensors": [
                "Concat_283_1",
                "Conv_367",
                "Concat_408",
                "Concat_315",
                "Conv_177",
                "Conv_320"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_302",
            "input_from": "blob.8",
            "save_as": "Conv_435",
            "w_key": "layer13_w",
            "b_key": "layer13_b",
            "W": 80,
            "H": 80,
            "Cin": 128,
            "Cout": 64,
            "K": 1,
            "stride": 1,
            "requant": 8,
            "delete_tensors": [
                "blob.8"
            ],
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "conv",
            "name": "Conv_322",
            "input_from": "Conv_435",
            "w_key": "layer14_w",
            "b_key": "layer14_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 128,
            "K": 3,
            "stride": 2,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_342",
            "save_as": "Conv_342",
            "w_key": "layer15_w",
            "b_key": "layer15_b",
            "W": 40,
            "H": 40,
            "Cin": 128,
            "Cout": 128,
            "K": 1,
            "stride": 1,
            "requant": 8,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "slice",
            "name": "Slice_354",
            "src": "Conv_342",
            "H": 40,
            "W": 40,
            "C": 128,
            "save_as_1": "Concat_480",
            "save_as_2": "Conv_485"
        },
        {
            "op": "conv",
            "name": "Conv_372",
            "input_from": "Conv_485",
            "w_key": "layer16_w",
            "b_key": "layer16_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_392",
            "save_as": "Conv_532",
            "w_key": "layer17_w",
            "b_key": "layer17_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "residual_from": "Conv_485",
            "device": "fpga",
            "bias_shift": 9
        },
        {
            "op": "conv",
            "name": "Conv_419",
            "input_from": "Conv_532",
            "w_key": "layer18_w",
            "b_key": "layer18_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_439",
            "save_as": "Concat_573",
            "w_key": "layer19_w",
            "b_key": "layer19_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 10,
            "residual_from": "Conv_532",
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "concat",
            "name": "Concat_448_1",
            "src1": "Conv_342",
            "src2": "Conv_532",
            "H": 40,
            "W": 40,
            "C1": 128,
            "C2": 64,
            "save_as": "Concat_448_1"
        },
        {
            "op": "concat",
            "name": "Concat_448_2",
            "src1": "Concat_448_1",
            "src2": "Concat_573",
            "H": 40,
            "W": 40,
            "C1": 192,
            "C2": 64,
            "save_as": "blob.16",
            "delete_tensors": [
                "Concat_573",
                "Concat_448_1",
                "Conv_532",
                "Conv_342",
                "Conv_485",
                "Concat_480"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_467",
            "input_from": "blob.16",
            "save_as": "Conv_600",
            "w_key": "layer20_w",
            "b_key": "layer20_b",
            "W": 40,
            "H": 40,
            "Cin": 256,
            "Cout": 128,
            "K": 1,
            "stride": 1,
            "requant": 9,
            "delete_tensors": [
                "blob.16"
            ],
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "conv",
            "name": "Conv_487",
            "input_from": "Conv_600",
            "w_key": "layer21_w",
            "b_key": "layer21_b",
            "W": 40,
            "H": 40,
            "Cin": 128,
            "Cout": 256,
            "K": 3,
            "stride": 2,
            "requant": 10,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_507",
            "save_as": "Conv_507",
            "w_key": "layer22_w",
            "b_key": "layer22_b",
            "W": 20,
            "H": 20,
            "Cin": 256,
            "Cout": 256,
            "K": 1,
            "stride": 1,
            "requant": 8,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "slice",
            "name": "Slice_524",
            "src": "Conv_507",
            "H": 20,
            "W": 20,
            "C": 256,
            "save_as_1": "Concat_645",
            "save_as_2": "Conv_650"
        },
        {
            "op": "conv",
            "name": "Conv_537",
            "input_from": "Conv_650",
            "w_key": "layer23_w",
            "b_key": "layer23_b",
            "W": 20,
            "H": 20,
            "Cin": 128,
            "Cout": 128,
            "K": 3,
            "stride": 1,
            "requant": 10,
            "device": "fpga",
            "bias_shift": 9
        },
        {
            "op": "conv",
            "name": "Conv_557",
            "save_as": "Concat_691",
            "w_key": "layer24_w",
            "b_key": "layer24_b",
            "W": 20,
            "H": 20,
            "Cin": 128,
            "Cout": 128,
            "K": 3,
            "stride": 1,
            "requant": 11,
            "residual_from": "Conv_650",
            "device": "fpga",
            "bias_shift": 10
        },
        {
            "op": "concat",
            "name": "Concat_566",
            "src1": "Conv_507",
            "src2": "Concat_691",
            "H": 20,
            "W": 20,
            "C1": 256,
            "C2": 128,
            "save_as": "blob.20",
            "delete_tensors": [
                "Concat_645",
                "Concat_691",
                "Conv_650",
                "Conv_507"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_585",
            "w_key": "layer25_w",
            "b_key": "layer25_b",
            "W": 20,
            "H": 20,
            "Cin": 384,
            "Cout": 256,
            "K": 1,
            "stride": 1,
            "requant": 9,
            "delete_tensors": [
                "blob.20"
            ],
            "device": "fpga",
            "bias_shift": 9
        },
        {
            "op": "conv",
            "name": "Conv_605",
            "save_as": "MaxPool_738",
            "w_key": "layer26_w",
            "b_key": "layer26_b",
            "W": 20,
            "H": 20,
            "Cin": 256,
            "Cout": 128,
            "K": 1,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 9
        },
        {
            "op": "sppf",
            "name": "SPPF_Fused",
            "H": 20,
            "W": 20,
            "C": 128,
            "K": 5,
            "input_from": "MaxPool_738",
            "save_as": "blob.32",
            "delete_tensors": [
                "MaxPool_738"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_647",
            "input_from": "blob.32",
            "save_as": "Conv_647",
            "w_key": "layer27_w",
            "b_key": "layer27_b",
            "W": 20,
            "H": 20,
            "Cin": 512,
            "Cout": 256,
            "K": 1,
            "stride": 1,
            "requant": 10,
            "device": "fpga",
            "delete_tensors": [
                "blob.32"
            ],
            "bias_shift": 11
        },
        {
            "op": "resize",
            "name": "Resize_655",
            "input_from": "Conv_647",
            "H": 20,
            "W": 20,
            "C": 256,
            "scale": 2,
            "save_as": "Concat_785"
        },
        {
            "op": "concat",
            "name": "Concat_656",
            "src1": "Concat_785",
            "src2": "Conv_600",
            "H": 40,
            "W": 40,
            "C1": 256,
            "C2": 128,
            "save_as": "blob.36",
            "delete_tensors": [
                "Concat_785",
                "Conv_600"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_675",
            "input_from": "blob.36",
            "save_as": "Conv_675",
            "w_key": "layer28_w",
            "b_key": "layer28_b",
            "W": 40,
            "H": 40,
            "Cin": 384,
            "Cout": 128,
            "K": 1,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "delete_tensors": [
                "blob.36"
            ],
            "bias_shift": 9
        },
        {
            "op": "slice",
            "name": "Slice_692",
            "src": "Conv_675",
            "H": 40,
            "W": 40,
            "C": 128,
            "save_as_1": "Concat_817",
            "save_as_2": "Conv_822"
        },
        {
            "op": "conv",
            "name": "Conv_705",
            "input_from": "Conv_822",
            "w_key": "layer29_w",
            "b_key": "layer29_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_725",
            "save_as": "Concat_856",
            "w_key": "layer30_w",
            "b_key": "layer30_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 10,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "concat",
            "name": "Concat_727",
            "src1": "Conv_675",
            "src2": "Concat_856",
            "H": 40,
            "W": 40,
            "C1": 128,
            "C2": 64,
            "save_as": "blob.40",
            "delete_tensors": [
                "Conv_675",
                "Concat_856",
                "Conv_822",
                "Concat_817"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_746",
            "input_from": "blob.40",
            "save_as": "Conv_746",
            "w_key": "layer31_w",
            "b_key": "layer31_b",
            "W": 40,
            "H": 40,
            "Cin": 192,
            "Cout": 128,
            "K": 1,
            "stride": 1,
            "requant": 7,
            "delete_tensors": [
                "blob.40"
            ],
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "resize",
            "name": "Resize_754",
            "input_from": "Conv_746",
            "H": 40,
            "W": 40,
            "C": 128,
            "scale": 2,
            "save_as": "Concat_888"
        },
        {
            "op": "concat",
            "name": "Concat_755",
            "src1": "Concat_888",
            "src2": "Conv_435",
            "H": 80,
            "W": 80,
            "C1": 128,
            "C2": 64,
            "save_as": "blob.44",
            "delete_tensors": [
                "Concat_888",
                "Conv_435"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_774",
            "input_from": "blob.44",
            "save_as": "Conv_774",
            "w_key": "layer32_w",
            "b_key": "layer32_b",
            "W": 80,
            "H": 80,
            "Cin": 192,
            "Cout": 64,
            "K": 1,
            "stride": 1,
            "requant": 8,
            "delete_tensors": [
                "blob.44"
            ],
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "slice",
            "name": "Slice_791",
            "src": "Conv_774",
            "H": 80,
            "W": 80,
            "C": 64,
            "save_as_1": "Concat_920",
            "save_as_2": "Conv_925"
        },
        {
            "op": "conv",
            "name": "Conv_804",
            "input_from": "Conv_925",
            "w_key": "layer33_w",
            "b_key": "layer33_b",
            "W": 80,
            "H": 80,
            "Cin": 32,
            "Cout": 32,
            "K": 3,
            "stride": 1,
            "requant": 8,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "conv",
            "name": "Conv_824",
            "save_as": "Concat_959",
            "w_key": "layer34_w",
            "b_key": "layer34_b",
            "W": 80,
            "H": 80,
            "Cin": 32,
            "Cout": 32,
            "K": 3,
            "stride": 1,
            "requant": 8,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "concat",
            "name": "Concat_826",
            "src1": "Conv_774",
            "src2": "Concat_959",
            "H": 80,
            "W": 80,
            "C1": 64,
            "C2": 32,
            "save_as": "blob.48",
            "delete_tensors": [
                "Conv_774",
                "Concat_959",
                "Conv_925",
                "Concat_920"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_845",
            "input_from": "blob.48",
            "save_as": "Conv_986",
            "w_key": "layer35_w",
            "b_key": "layer35_b",
            "W": 80,
            "H": 80,
            "Cin": 96,
            "Cout": 64,
            "K": 1,
            "stride": 1,
            "requant": 8,
            "delete_tensors": [
                "blob.48"
            ],
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "conv",
            "name": "Conv_865",
            "input_from": "Conv_986",
            "save_as": "Concat_1000",
            "w_key": "layer36_w",
            "b_key": "layer36_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 2,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "concat",
            "name": "Concat_867",
            "src1": "Concat_1000",
            "src2": "Conv_746",
            "H": 40,
            "W": 40,
            "C1": 64,
            "C2": 128,
            "save_as": "blob.52",
            "delete_tensors": [
                "Concat_1000",
                "Conv_746"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_886",
            "input_from": "blob.52",
            "save_as": "Conv_886",
            "w_key": "layer37_w",
            "b_key": "layer37_b",
            "W": 40,
            "H": 40,
            "Cin": 192,
            "Cout": 128,
            "K": 1,
            "stride": 1,
            "requant": 8,
            "delete_tensors": [
                "blob.52"
            ],
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "slice",
            "name": "Slice_903",
            "src": "Conv_886",
            "H": 40,
            "W": 40,
            "C": 128,
            "save_as_1": "Concat_1032",
            "save_as_2": "Conv_1037"
        },
        {
            "op": "conv",
            "name": "Conv_916",
            "input_from": "Conv_1037",
            "w_key": "layer38_w",
            "b_key": "layer38_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "conv",
            "name": "Conv_936",
            "save_as": "Concat_1071",
            "w_key": "layer39_w",
            "b_key": "layer39_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "concat",
            "name": "Concat_938",
            "src1": "Conv_886",
            "src2": "Concat_1071",
            "H": 40,
            "W": 40,
            "C1": 128,
            "C2": 64,
            "save_as": "blob.56",
            "delete_tensors": [
                "Conv_886",
                "Concat_1071",
                "Conv_1037",
                "Concat_1032"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_957",
            "input_from": "blob.56",
            "save_as": "Conv_1098",
            "w_key": "layer40_w",
            "b_key": "layer40_b",
            "W": 40,
            "H": 40,
            "Cin": 192,
            "Cout": 128,
            "K": 1,
            "stride": 1,
            "requant": 8,
            "delete_tensors": [
                "blob.56"
            ],
            "device": "fpga",
            "bias_shift": 6
        },
        {
            "op": "conv",
            "name": "Conv_977",
            "input_from": "Conv_1098",
            "save_as": "Concat_1112",
            "w_key": "layer41_w",
            "b_key": "layer41_b",
            "W": 40,
            "H": 40,
            "Cin": 128,
            "Cout": 128,
            "K": 3,
            "stride": 2,
            "requant": 11,
            "device": "fpga",
            "bias_shift": 9
        },
        {
            "op": "concat",
            "name": "Concat_979",
            "src1": "Concat_1112",
            "src2": "Conv_647",
            "H": 20,
            "W": 20,
            "C1": 128,
            "C2": 256,
            "save_as": "blob.60",
            "delete_tensors": [
                "Concat_1112",
                "Conv_647"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_998",
            "input_from": "blob.60",
            "save_as": "Conv_998",
            "w_key": "layer42_w",
            "b_key": "layer42_b",
            "W": 20,
            "H": 20,
            "Cin": 384,
            "Cout": 256,
            "K": 1,
            "stride": 1,
            "requant": 9,
            "delete_tensors": [
                "blob.60"
            ],
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "slice",
            "name": "Slice_1015",
            "src": "Conv_998",
            "H": 20,
            "W": 20,
            "C": 256,
            "save_as_1": "Concat_1144",
            "save_as_2": "Conv_1149"
        },
        {
            "op": "conv",
            "name": "Conv_1028",
            "input_from": "Conv_1149",
            "w_key": "layer43_w",
            "b_key": "layer43_b",
            "W": 20,
            "H": 20,
            "Cin": 128,
            "Cout": 128,
            "K": 3,
            "stride": 1,
            "requant": 12,
            "device": "fpga",
            "bias_shift": 9
        },
        {
            "op": "conv",
            "name": "Conv_1048",
            "save_as": "Concat_1183",
            "w_key": "layer44_w",
            "b_key": "layer44_b",
            "W": 20,
            "H": 20,
            "Cin": 128,
            "Cout": 128,
            "K": 3,
            "stride": 1,
            "requant": 10,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "concat",
            "name": "Concat_1050",
            "src1": "Conv_998",
            "src2": "Concat_1183",
            "H": 20,
            "W": 20,
            "C1": 256,
            "C2": 128,
            "save_as": "blob.64",
            "delete_tensors": [
                "Conv_998",
                "Concat_1183",
                "Conv_1149",
                "Concat_1144"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_1069",
            "input_from": "blob.64",
            "save_as": "Conv_1210",
            "w_key": "layer45_w",
            "b_key": "layer45_b",
            "W": 20,
            "H": 20,
            "Cin": 384,
            "Cout": 256,
            "K": 1,
            "stride": 1,
            "requant": 10,
            "delete_tensors": [
                "blob.64"
            ],
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_1089",
            "input_from": "Conv_986",
            "w_key": "layer46_w",
            "b_key": "layer46_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_1109",
            "w_key": "layer47_w",
            "b_key": "layer47_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "conv",
            "name": "Conv_1129",
            "save_as": "Concat_1263",
            "w_key": "layer48_w",
            "b_key": "layer48_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 64,
            "K": 1,
            "stride": 1,
            "requant": 9,
            "apply_activation": False,
            "is_output": True,
            "device": "cpu",
            "bias_shift": 5
        },
        {
            "op": "conv",
            "name": "Conv_1142",
            "input_from": "Conv_986",
            "w_key": "layer49_w",
            "b_key": "layer49_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 8,
            "delete_tensors": [
                "Conv_986"
            ],
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_1162",
            "w_key": "layer50_w",
            "b_key": "layer50_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 9
        },
        {
            "op": "conv",
            "name": "Conv_1182",
            "save_as": "Concat_1316",
            "w_key": "layer51_w",
            "b_key": "layer51_b",
            "W": 80,
            "H": 80,
            "Cin": 64,
            "Cout": 1,
            "K": 1,
            "stride": 1,
            "requant": 9,
            "apply_activation": False,
            "is_output": True,
            "device": "cpu",
            "bias_shift": 7
        },
        {
            "op": "concat",
            "name": "Concat_1183",
            "src1": "Concat_1263",
            "src2": "Concat_1316",
            "H": 80,
            "W": 80,
            "C1": 64,
            "C2": 1,
            "save_as": "1323",
            "delete_tensors": [
                "Concat_1263",
                "Concat_1316"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_1202",
            "input_from": "Conv_1098",
            "w_key": "layer52_w",
            "b_key": "layer52_b",
            "W": 40,
            "H": 40,
            "Cin": 128,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 10,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_1222",
            "w_key": "layer53_w",
            "b_key": "layer53_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 10,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_1242",
            "save_as": "Concat_1376",
            "w_key": "layer54_w",
            "b_key": "layer54_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 1,
            "stride": 1,
            "requant": 9,
            "apply_activation": False,
            "is_output": True,
            "device": "cpu",
            "bias_shift": 5
        },
        {
            "op": "conv",
            "name": "Conv_1255",
            "input_from": "Conv_1098",
            "w_key": "layer55_w",
            "b_key": "layer55_b",
            "W": 40,
            "H": 40,
            "Cin": 128,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 10,
            "delete_tensors": [
                "Conv_1098"
            ],
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_1275",
            "w_key": "layer56_w",
            "b_key": "layer56_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_1295",
            "save_as": "Concat_1429",
            "w_key": "layer57_w",
            "b_key": "layer57_b",
            "W": 40,
            "H": 40,
            "Cin": 64,
            "Cout": 1,
            "K": 1,
            "stride": 1,
            "requant": 8,
            "apply_activation": False,
            "is_output": True,
            "device": "cpu",
            "bias_shift": 6
        },
        {
            "op": "concat",
            "name": "Concat_1296",
            "src1": "Concat_1376",
            "src2": "Concat_1429",
            "H": 40,
            "W": 40,
            "C1": 64,
            "C2": 1,
            "save_as": "1436",
            "delete_tensors": [
                "Concat_1376",
                "Concat_1429"
            ]
        },
        {
            "op": "conv",
            "name": "Conv_1315",
            "input_from": "Conv_1210",
            "w_key": "layer58_w",
            "b_key": "layer58_b",
            "W": 20,
            "H": 20,
            "Cin": 256,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 12,
            "device": "fpga",
            "bias_shift": 9
        },
        {
            "op": "conv",
            "name": "Conv_1335",
            "w_key": "layer59_w",
            "b_key": "layer59_b",
            "W": 20,
            "H": 20,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 7
        },
        {
            "op": "conv",
            "name": "Conv_1355",
            "save_as": "Concat_1489",
            "w_key": "layer60_w",
            "b_key": "layer60_b",
            "W": 20,
            "H": 20,
            "Cin": 64,
            "Cout": 64,
            "K": 1,
            "stride": 1,
            "requant": 10,
            "apply_activation": False,
            "is_output": True,
            "device": "cpu",
            "bias_shift": 6
        },
        {
            "op": "conv",
            "name": "Conv_1368",
            "input_from": "Conv_1210",
            "w_key": "layer61_w",
            "b_key": "layer61_b",
            "W": 20,
            "H": 20,
            "Cin": 256,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 12,
            "delete_tensors": [
                "Conv_1210"
            ],
            "device": "fpga",
            "bias_shift": 10
        },
        {
            "op": "conv",
            "name": "Conv_1388",
            "w_key": "layer62_w",
            "b_key": "layer62_b",
            "W": 20,
            "H": 20,
            "Cin": 64,
            "Cout": 64,
            "K": 3,
            "stride": 1,
            "requant": 9,
            "device": "fpga",
            "bias_shift": 8
        },
        {
            "op": "conv",
            "name": "Conv_1408",
            "save_as": "Concat_1542",
            "w_key": "layer63_w",
            "b_key": "layer63_b",
            "W": 20,
            "H": 20,
            "Cin": 64,
            "Cout": 1,
            "K": 1,
            "stride": 1,
            "requant": 10,
            "apply_activation": False,
            "is_output": True,
            "device": "cpu",
            "bias_shift": 8,
        },
        {
            "op": "concat",
            "name": "Concat_1409",
            "src1": "Concat_1489",
            "src2": "Concat_1542",
            "H": 20,
            "W": 20,
            "C1": 64,
            "C2": 1,
            "save_as": "1549",
            "delete_tensors": [
                "Concat_1489",
                "Concat_1542"
            ],
            
        }
    ]

    total_start_time = time.perf_counter()

    # 2. LOOP THROUGH IMAGES
    for idx, img_path in enumerate(image_paths):
        print(f"\n" + "="*50)
        print(f"PROCESSING IMAGE {idx+1}/{len(image_paths)}: {img_path}")
        print("="*50)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        h, w = img.shape[:2]
        img_int8 = preprocess_quantized(img, size=640)

        # Initialize a NEW graph runner for each image to clear memory!
        graph_runner = YOLOv8GraphRunner(engine=engine, npz_data=npz_data)
        
        # Run inference
        inf_start = time.perf_counter()
        final_output = graph_runner.run(network_layers=network_layers, input_tensor=img_int8)
        inf_end = time.perf_counter()
        
        print(f"\n[INFO] Inference alone took: {inf_end - inf_start:.4f} sec")

        try:
            # Grab outputs directly from memory
            raw_outputs = [
                graph_runner.tensor_store["1323"],
                graph_runner.tensor_store["1436"],
                graph_runner.tensor_store["1549"]
            ]
            
            # Sort (largest to smallest) and format
            outputs_sorted = sorted(raw_outputs, key=lambda x: x.size, reverse=True)
            formatted_outputs = []
            
            for out, dequant_scale in zip(outputs_sorted, DEQUANT_SCALES):
                out_float = out.astype(np.float32) * dequant_scale
                spatial_dim = int(np.sqrt(out_float.size / CHANNELS))
                out_hwc = out_float.reshape((spatial_dim, spatial_dim, CHANNELS))
                out_chw = out_hwc.transpose(2, 0, 1)
                out_bchw = np.expand_dims(out_chw, axis=0)
                formatted_outputs.append(out_bchw)

            # Postprocess
            boxes, scores = postprocess(formatted_outputs)
            num_boxes = len(boxes)
            print(f"Total boxes detected: {num_boxes}")

            # Draw on Image
            scale = min(640 / h, 640 / w)
            if num_boxes > 0:
                boxes /= scale
                
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                text_scale = max(0.5, w / 1500.0)
                thickness = max(1, int(w / 1000.0))
                box_thick = max(2, int(w / 800.0))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), box_thick)
                cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), thickness)

            cv2.putText(img, f"Total Boxes: {num_boxes}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, max(1.0, w / 1000.0), (0, 0, 255), max(2, int(w / 800.0)))

            # Save with a dynamic name
            out_filename = f"result_{idx}.jpg"
            cv2.imwrite(out_filename, img)
            print(f"Saved {out_filename}")

        except KeyError as e:
            print(f"\n[Warning] Could not process outputs: Tensor {e} not found.")

    # Loop finished
    total_end_time = time.perf_counter()
    print(f"\nProcessed {len(image_paths)} images in {total_end_time - total_start_time:.2f} seconds.")

    engine.clean_up()
    print("\nHardware Cleaned Up!")

    # Start Web Server to view results
    PORT = 8080
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n===================================")
    print("ALL RESULTS AVAILABLE OVER ETHERNET")
    print(f"Go to: http://{local_ip}:{PORT}/")
    print("Click on the 'result_X.jpg' files to view them.")
    print("===================================\n")

    server = HTTPServer(("0.0.0.0", PORT), SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True 
    server_thread.start()

    input("Press Enter to stop server and exit...\n")
    server.shutdown()
    server.server_close()
