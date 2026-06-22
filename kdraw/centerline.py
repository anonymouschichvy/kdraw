import os
import sys
import time
import numpy as np

from .graph import build_and_prune_graph
from .optimization import optimize_paths
from .smoothing import smooth_paths_chaikin, smooth_paths_laplacian

def convert_centerline(input_path, output_path, threshold_val=None, epsilon=0.3, no_sort=False,
                       invert_threshold=False, blur_size=9, use_adaptive=False, block_size=15, c_val=10,
                       min_spur_length=16, max_join=2.5, loop_gap=0.0, min_path_len=0.0,
                       smooth_type='chaikin', smooth_iters=3, smooth_weight=0.5, smooth_decimate=0.1,
                       upscale_factor=4, morph_close=5, morph_open=0, collapse_junc=8):
    """
    Skeletonize the image to a 1-pixel-wide centerline, trace it into single-line paths,
    prune short spurs using graph topology, and output a stroke-only SVG with minimized pen-up travel.
    """
    try:
        import cv2
    except ImportError:
        print("\n[ERROR] The 'opencv-python' package is required for Centerline Mode.")
        print("Please install it via:")
        print("    pip install opencv-python")
        return False
        
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        print("\n[ERROR] The 'scikit-image' package is required for Centerline Mode.")
        print("Please install it via:")
        print("    pip install scikit-image")
        return False
        
    print(f"Loading image {input_path} in grayscale...")
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image {input_path} with OpenCV.")
        sys.exit(1)
        
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Upscale if requested
    if upscale_factor > 1:
        print(f"Upscaling input image by {upscale_factor}x for smooth curve definition...")
        width_up, height_up = width * upscale_factor, height * upscale_factor
        gray = cv2.resize(gray, (width_up, height_up), interpolation=cv2.INTER_CUBIC)
        if blur_size > 0:
            blur_size = int(blur_size)
            if blur_size % 2 == 0:
                blur_size += 1
        
    # 2. Apply Gaussian Blur to smooth pixelated edges and JPEG compression wiggles
    if blur_size > 0:
        if blur_size % 2 == 0:
            blur_size += 1
        print(f"Applying Gaussian Blur (kernel={blur_size}x{blur_size}) to smooth wiggles...")
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
    # 3. Apply Thresholding
    if use_adaptive:
        print(f"Applying Adaptive Gaussian Thresholding (blockSize={block_size}, C={c_val})...")
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)
        
        thresh_type = cv2.THRESH_BINARY if invert_threshold else cv2.THRESH_BINARY_INV
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, c_val
        )
    else:
        thresh_type = cv2.THRESH_BINARY if invert_threshold else cv2.THRESH_BINARY_INV
        if threshold_val is None:
            print("Applying Otsu's adaptive thresholding...")
            _, processed = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        else:
            print(f"Applying binary thresholding at value {threshold_val}...")
            _, processed = cv2.threshold(gray, threshold_val, 255, thresh_type)
            
    # 4. Apply Morphological closing/opening if upscaled
    if upscale_factor > 1:
        if morph_close > 0:
            print(f"Applying morphological closing (kernel={morph_close}x{morph_close})...")
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_close)
        if morph_open > 0:
            print(f"Applying morphological opening (kernel={morph_open}x{morph_open})...")
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_open)
            
    # 5. Skeletonize (thinning to 1-pixel centerline)
    print("Skeletonizing image...")
    start_skel = time.time()
    binary_bool = (processed > 0)
    skel_bool = skeletonize(binary_bool)
    print(f"Skeletonization completed in {time.time() - start_skel:.3f} seconds.")
    
    # 6. Graph-based tracing and pruning
    print(f"Tracing centerline paths with graph topology (min_spur={min_spur_length}, collapse_junc={collapse_junc})...")
    start_trace = time.time()
    paths = build_and_prune_graph(skel_bool, min_spur_length=min_spur_length, collapse_dist=collapse_junc)
    print(f"Tracing finished in {time.time() - start_trace:.3f} seconds. Extracted {len(paths):,} paths.")
    
    # 7. Scale coordinates back and simplify paths
    processed_paths = []
    pruned_count = 0
    total_raw_points = sum(len(p) for p in paths)
    
    for p in paths:
        if len(p) < 2:
            continue
            
        # Scale back to original coordinates
        if upscale_factor > 1:
            p_scaled = p / upscale_factor
        else:
            p_scaled = p
            
        # Pruning short paths
        if min_path_len > 0.0:
            p_len = float(np.sum(np.sqrt(np.sum(np.diff(p_scaled, axis=0)**2, axis=1))))
            if p_len < min_path_len:
                pruned_count += 1
                continue
                
        # Simplify path using RDP
        if len(p_scaled) == 2:
            approx = p_scaled
        else:
            p_reshaped = p_scaled.reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(p_reshaped, epsilon, False).reshape(-1, 2)
            
        if len(approx) >= 2:
            processed_paths.append(approx)
            
    paths = processed_paths
    if min_path_len > 0.0:
        print(f"Pruned {pruned_count:,} short paths (length < {min_path_len}px).")
        
    # 8. Path Smoothing and Post-decimation
    if smooth_iters > 0:
        print(f"Applying {smooth_type} path smoothing ({smooth_iters} iterations)...")
        smoothed_paths = []
        for p in paths:
            if smooth_type.lower() == 'chaikin':
                sp = smooth_paths_chaikin(p, smooth_iters)
            else:
                sp = smooth_paths_laplacian(p, smooth_iters, smooth_weight)
                
            if smooth_decimate > 0.0 and len(sp) > 2:
                sp_reshaped = sp.reshape(-1, 1, 2)
                approx = cv2.approxPolyDP(sp_reshaped, smooth_decimate, False)
                sp = approx.reshape(-1, 2)
                
            if len(sp) >= 2:
                smoothed_paths.append(sp)
        paths = smoothed_paths
        
    total_simp_points = sum(len(p) for p in paths)
    print(f"Simplified to {len(paths):,} paths.")
    print(f"Reduced points from {total_raw_points:,} to {total_simp_points:,} ({(1 - total_simp_points/max(1, total_raw_points))*100:.1f}% reduction).")
    
    # 9. Optimize path sequence (TSP)
    if not no_sort:
        print("Optimizing path sequences to minimize pen travel (TSP)...")
        start_sort = time.time()
        optimized, unopt_travel, opt_travel = optimize_paths(paths, max_join)
        sort_time = time.time() - start_sort
        print(f"TSP optimization finished in {sort_time:.3f} seconds.")
        if unopt_travel > 0:
            saved = (1 - opt_travel / unopt_travel) * 100
            print(f"Pen-up travel distance reduced from {unopt_travel:.1f}px to {opt_travel:.1f}px ({saved:.1f}% travel saved!).")
    else:
        print("Skipping path sequence optimization.")
        optimized = paths
        
    print(f"Writing centerline SVG to {output_path}...")
    start_write = time.time()
    
    with open(output_path, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n')
        
        path_d_parts = []
        for path in optimized:
            if len(path) < 2:
                continue
            d_str = f"M{path[0][0]:.2f},{path[0][1]:.2f}"
            for pt in path[1:]:
                d_str += f"L{pt[0]:.2f},{pt[1]:.2f}"
            path_d_parts.append(d_str)
            
        path_d = " ".join(path_d_parts)
        f.write(f'  <path d="{path_d}" fill="none" stroke="black" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />\n')
        f.write('</svg>\n')
        
    print(f"SVG written in {time.time() - start_write:.3f} seconds.")
    print(f"Output file size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB" if os.path.getsize(output_path) > 1024*1024 else f"Output file size: {os.path.getsize(output_path) / 1024:.2f} KB")
    return True
