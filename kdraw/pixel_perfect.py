import os
import time
import collections
import numpy as np
from PIL import Image

def get_hex_color(val, has_alpha=False):
    """Convert packed pixel integer to SVG-compatible hex or rgba color string."""
    if has_alpha:
        a = (val >> 24) & 0xFF
        r = (val >> 16) & 0xFF
        g = (val >> 8) & 0xFF
        b = val & 0xFF
        if a < 255:
            return f"rgba({r},{g},{b},{a/255.0:.3f})"
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        r = (val >> 16) & 0xFF
        g = (val >> 8) & 0xFF
        b = val & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"

def convert_pixel_perfect(input_path, output_path, num_colors=None):
    """
    Convert a raster image to SVG using an optimized Run-Length Box-Merging (RLBM) algorithm.
    This groups contiguous pixels of the same color into horizontal/vertical rectangular spans
    and combines them into a single path per color to minimize file size and render time.
    """
    print(f"Loading image {input_path}...")
    img = Image.open(input_path)
    width, height = img.size
    print(f"Dimensions: {width} x {height} ({width * height:,} pixels)")
    
    has_alpha = (img.mode == 'RGBA')
    if num_colors is not None and num_colors > 0:
        print(f"Quantizing image to {num_colors} colors...")
        start_q = time.time()
        if has_alpha:
            quantized = img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
        else:
            quantized = img.convert('RGB').quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
            
        palette = quantized.getpalette()
        data = np.array(quantized)
        
        def get_color_str(idx):
            r = palette[3 * idx]
            g = palette[3 * idx + 1]
            b = palette[3 * idx + 2]
            return f"#{r:02x}{g:02x}{b:02x}"
        print(f"Quantization finished in {time.time() - start_q:.3f} seconds.")
    else:
        print("Using lossless mode (no color quantization)...")
        if has_alpha:
            arr = np.array(img)
            data = (arr[:, :, 3].astype(np.uint32) << 24) | \
                   (arr[:, :, 0].astype(np.uint32) << 16) | \
                   (arr[:, :, 1].astype(np.uint32) << 8) | \
                   arr[:, :, 2].astype(np.uint32)
        else:
            arr = np.array(img.convert('RGB'))
            data = (arr[:, :, 0].astype(np.uint32) << 16) | \
                   (arr[:, :, 1].astype(np.uint32) << 8) | \
                   arr[:, :, 2].astype(np.uint32)
            
        def get_color_str(val):
            return get_hex_color(val, has_alpha)

    print("Running Run-Length Box-Merging algorithm...")
    start_merge = time.time()
    
    active_spans = {}
    rects_by_color = collections.defaultdict(list)
    
    for y in range(height):
        row = data[y]
        changes = np.where(row[:-1] != row[1:])[0]
        starts = np.concatenate(([0], changes + 1))
        ends = np.concatenate((changes + 1, [width]))
        colors = row[starts]
        
        next_active_spans = {}
        
        for start, end, color in zip(starts, ends, colors):
            key = (start, end, color)
            if key in active_spans:
                y_start = active_spans.pop(key)
                next_active_spans[key] = y_start
            else:
                next_active_spans[key] = y
                
        for key, y_start in active_spans.items():
            start, end, color = key
            color_str = get_color_str(color)
            rects_by_color[color_str].append((start, y_start, end - start, y - y_start))
            
        active_spans = next_active_spans

    for key, y_start in active_spans.items():
        start, end, color = key
        color_str = get_color_str(color)
        rects_by_color[color_str].append((start, y_start, end - start, height - y_start))
        
    print(f"Merging finished in {time.time() - start_merge:.3f} seconds.")
    
    total_rects = sum(len(rects) for rects in rects_by_color.values())
    unique_colors = len(rects_by_color)
    print(f"Found {total_rects:,} merged rectangles across {unique_colors:,} unique colors.")
    print(f"Average pixels per rectangle: {(width * height) / total_rects:.1f}")

    print(f"Writing SVG to {output_path}...")
    start_write = time.time()
    
    with open(output_path, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" shape-rendering="crispEdges">\n')
        
        for color_str, rects in rects_by_color.items():
            path_d_parts = []
            for x, y, w, h in rects:
                path_d_parts.append(f"M{x},{y}h{w}v{h}h{-w}z")
            
            path_d = "".join(path_d_parts)
            f.write(f'  <path d="{path_d}" fill="{color_str}" />\n')
            
        f.write('</svg>\n')
        
    print(f"SVG written in {time.time() - start_write:.3f} seconds.")
    print(f"Output file size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
