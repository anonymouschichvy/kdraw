import os
import time

def convert_smooth(input_path, output_path, filter_speckle=4, color_precision=6, corner_threshold=60, path_precision=3):
    """
    Convert a raster image to SVG using the Rust-backed `vtracer` library.
    This creates smooth Bezier paths, perfect for vector logos or drawings.
    """
    try:
        import vtracer
    except ImportError:
        print("\n[ERROR] The 'vtracer' package is not installed.")
        print("To use smooth vectorization, please install it via:")
        print("    pip install vtracer")
        return False
        
    print(f"Vectorizing {input_path} with vtracer (smooth curves)...")
    start = time.time()
    
    vtracer.convert_image_to_svg_py(
        input_path,
        output_path,
        colormode='color',
        hierarchical='stacked',
        mode='spline',
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        corner_threshold=corner_threshold,
        path_precision=path_precision
    )
    
    print(f"Smooth vectorization completed in {time.time() - start:.3f} seconds.")
    print(f"Output file size: {os.path.getsize(output_path) / 1024:.2f} KB")
    return True
