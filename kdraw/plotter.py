import os
import sys
import time
import numpy as np

from .optimization import optimize_paths
from .smoothing import smooth_paths_chaikin, smooth_paths_laplacian

def convert_plotter(input_path, output_path, use_canny=False, threshold_val=None, epsilon=0.5, no_sort=False, blur_size=0, max_join=2.0,
                    smooth_type='chaikin', smooth_iters=0, smooth_weight=0.5, smooth_decimate=0.0):
    """
    Extract contour paths from image, simplify them, sort them to minimize pen travel,
    and output a stroke-only SVG ideal for a CNC pen plotter.
    """
    try:
        import cv2
    except ImportError:
        print("\n[ERROR] The 'opencv-python' package is required for CNC Plotter Mode.")
        print("Please install it via:")
        print("    pip install opencv-python")
        return False
        
    print(f"Loading image {input_path} in grayscale...")
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image {input_path} with OpenCV.")
        sys.exit(1)
        
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if blur_size > 0:
        if blur_size % 2 == 0:
            blur_size += 1
        print(f"Applying Gaussian Blur (kernel={blur_size}x{blur_size}) to smooth outlines...")
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    if use_canny:
        print("Using Canny Edge Detection...")
        median_val = np.median(gray)
        lower = int(max(0, 0.66 * median_val))
        upper = int(min(255, 1.33 * median_val))
        print(f"Canny auto-thresholds: lower={lower}, upper={upper}")
        processed = cv2.Canny(gray, lower, upper)
    else:
        print("Using Thresholding...")
        if threshold_val is None:
            print("Applying Otsu's adaptive thresholding...")
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            print(f"Applying binary thresholding at value {threshold_val}...")
            _, processed = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
            
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Extracted {len(contours):,} raw contours.")
    
    simplified = []
    total_raw_points = sum(len(c) for c in contours)
    for c in contours:
        if len(c) < 2:
            continue
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) > 1:
            simplified.append(approx.reshape(-1, 2))
            
    # Apply Path Smoothing and Optional Decimation
    if smooth_iters > 0:
        print(f"Applying {smooth_type} path smoothing ({smooth_iters} iterations)...")
        smoothed_paths = []
        for p in simplified:
            if smooth_type.lower() == 'chaikin':
                sp = smooth_paths_chaikin(p, smooth_iters)
            else:
                sp = smooth_paths_laplacian(p, smooth_iters, smooth_weight)
                
            if smooth_decimate > 0.0 and len(sp) >= 2:
                sp_reshaped = sp.reshape(-1, 1, 2)
                approx = cv2.approxPolyDP(sp_reshaped, smooth_decimate, True)
                sp = approx.reshape(-1, 2)
                
            if len(sp) >= 2:
                smoothed_paths.append(sp)
        simplified = smoothed_paths
            
    total_simp_points = sum(len(c) for c in simplified)
    print(f"Simplified to {len(simplified):,} contours.")
    print(f"Reduced points from {total_raw_points:,} to {total_simp_points:,} ({(1 - total_simp_points/max(1, total_raw_points))*100:.1f}% reduction).")
    
    if not no_sort:
        print("Optimizing path sequences to minimize pen travel (TSP)...")
        start_sort = time.time()
        optimized, unopt_travel, opt_travel = optimize_paths(simplified, max_join)
        sort_time = time.time() - start_sort
        print(f"TSP optimization finished in {sort_time:.3f} seconds.")
        if unopt_travel > 0:
            saved = (1 - opt_travel / unopt_travel) * 100
            print(f"Pen-up travel distance reduced from {unopt_travel:.1f}px to {opt_travel:.1f}px ({saved:.1f}% travel saved!).")
    else:
        print("Skipping path sequence optimization.")
        optimized = [c.reshape(-1, 2) for c in simplified]
        
    print(f"Writing stroke-only SVG to {output_path}...")
    start_write = time.time()
    
    with open(output_path, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n')
        
        path_d_parts = []
        for contour in optimized:
            if len(contour) < 2:
                continue
            d_str = f"M{contour[0][0]:.2f},{contour[0][1]:.2f}"
            for pt in contour[1:]:
                d_str += f"L{pt[0]:.2f},{pt[1]:.2f}"
            d_str += "z"
            path_d_parts.append(d_str)
            
        path_d = " ".join(path_d_parts)
        f.write(f'  <path d="{path_d}" fill="none" stroke="black" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />\n')
        f.write('</svg>\n')
        
    print(f"SVG written in {time.time() - start_write:.3f} seconds.")
    print(f"Output file size: {os.path.getsize(output_path) / 1024:.2f} KB")
    return True
