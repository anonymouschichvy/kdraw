#!/usr/bin/env python3
import os
import sys
import argparse

import kdraw

def main():
    parser = argparse.ArgumentParser(description="Convert an image (JPG/PNG) into an optimized SVG path document.")
    parser.add_argument("input_path", help="Path to the input image (e.g. a.jpg)")
    parser.add_argument("output_path", nargs="?", help="Path to the output SVG (defaults to <input_name>.svg)")
    parser.add_argument("-c", "--colors", type=int, default=None,
                        help="Reduce image to N colors using quantization before merging (highly recommended for JPEGs)")
    parser.add_argument("-l", "--lossless", action="store_true",
                        help="Convert exact pixel colors without quantization (can result in very large files for JPEGs)")
    parser.add_argument("-s", "--smooth", action="store_true",
                        help="Generate smooth vectorized curves instead of pixel-perfect blocks (requires vtracer)")
    
    # CNC Plotter arguments
    parser.add_argument("-p", "--plotter", action="store_true",
                        help="Generate stroke-only SVG paths optimized for a CNC pen plotter")
    parser.add_argument("-cl", "--centerline", action="store_true",
                        help="Generate single-line centerpaths using skeletonization to eliminate bubble letters")
    parser.add_argument("--invert", action="store_true",
                        help="Invert thresholding in centerline mode (for white lines on a dark background)")
    parser.add_argument("--canny", action="store_true",
                        help="Use Canny edge detection instead of thresholding for plotter mode")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Custom binary threshold value (0-255) for plotter/centerline mode")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="Path simplification distance tolerance in pixels (default: 0.3)")
    parser.add_argument("--no-sort", action="store_true",
                        help="Skip TSP path sorting/sequence optimization")
    parser.add_argument("--max-join", type=float, default=2.5,
                        help="Join path endpoints within this distance in pixels to reduce pen lifts (default: 2.5)")
    
    # Path smoothing options
    parser.add_argument("--smooth-type", choices=["laplacian", "chaikin"], default="chaikin",
                        help="Smoothing algorithm to use: laplacian or chaikin (default: chaikin)")
    parser.add_argument("--smooth-iters", type=int, default=3,
                        help="Number of smoothing passes/iterations (default: 3)")
    parser.add_argument("--smooth-weight", type=float, default=0.5,
                        help="Laplacian smoothing blend factor between 0.0 and 1.0 (default: 0.5)")
    parser.add_argument("--smooth-decimate", type=float, default=0.1,
                        help="Post-smoothing RDP decimation epsilon tolerance (default: 0.1)")
    
    # Image processing enhancements
    parser.add_argument("--blur", type=int, default=9,
                        help="Gaussian blur kernel size to smooth out pixelation wiggles (default: 9)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable Adaptive Gaussian thresholding and use global thresholding instead")
    parser.add_argument("--block-size", type=int, default=15,
                        help="Local neighborhood block size for adaptive thresholding (default: 15)")
    parser.add_argument("--c-val", type=int, default=10,
                        help="Constant subtracted from local mean for adaptive thresholding; higher makes lines thinner (default: 10)")
    parser.add_argument("--min-spur", type=int, default=16,
                        help="Minimum pixel length for a skeleton branch to not be pruned as a spur (default: 16)")
    parser.add_argument("--loop-gap", type=float, default=0.0,
                        help="Width of gap in pixels to open small closed loops (e.g. 5.0 to 8.0) (default: 0.0)")
    parser.add_argument("--min-path-len", type=float, default=0.0,
                        help="Minimum length of a path in pixels to keep; shorter paths are pruned (default: 0.0)")
    
    # Upscaling & Advanced Graph-based Truning arguments
    parser.add_argument("--upscale", type=int, default=4,
                        help="Upscale factor to smooth out pixelation wiggles during centerline mode (default: 4)")
    parser.add_argument("--morph-close", type=int, default=5,
                        help="Morphological closing kernel size on upscaled image to fill gaps (default: 5)")
    parser.add_argument("--morph-open", type=int, default=0,
                        help="Morphological opening kernel size on upscaled image to smooth contours (default: 0)")
    parser.add_argument("--collapse-junc", type=int, default=8,
                        help="Distance in pixels below which adjacent junctions will be collapsed (default: 8)")
    
    # Advanced vtracer parameters
    parser.add_argument("--filter-speckle", type=int, default=4, help="Speckle filter size for smooth vectorization")
    parser.add_argument("--color-precision", type=int, default=6, help="Color precision (significant bits) for smooth vectorization")
    parser.add_argument("--corner-threshold", type=int, default=60, help="Corner threshold angle for smooth vectorization")
    parser.add_argument("--path-precision", type=int, default=3, help="Decimal precision of path coordinates")

    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' not found.")
        sys.exit(1)
        
    output_path = args.output_path
    if not output_path:
        base, _ = os.path.splitext(args.input_path)
        if args.centerline:
            suffix = "_centerline"
        elif args.plotter:
            suffix = "_plotter"
        else:
            suffix = ""
        output_path = base + suffix + ".svg"
        
    if args.centerline:
        success = kdraw.convert_centerline(
            args.input_path, output_path,
            threshold_val=args.threshold,
            epsilon=args.epsilon,
            no_sort=args.no_sort,
            invert_threshold=args.invert,
            blur_size=args.blur,
            use_adaptive=not args.no_adaptive,
            block_size=args.block_size,
            c_val=args.c_val,
            min_spur_length=args.min_spur,
            max_join=args.max_join,
            loop_gap=args.loop_gap,
            min_path_len=args.min_path_len,
            smooth_type=args.smooth_type,
            smooth_iters=args.smooth_iters,
            smooth_weight=args.smooth_weight,
            smooth_decimate=args.smooth_decimate,
            upscale_factor=args.upscale,
            morph_close=args.morph_close,
            morph_open=args.morph_open,
            collapse_junc=args.collapse_junc
        )
        if not success:
            sys.exit(1)
    elif args.plotter:
        success = kdraw.convert_plotter(
            args.input_path, output_path,
            use_canny=args.canny,
            threshold_val=args.threshold,
            epsilon=args.epsilon,
            no_sort=args.no_sort,
            blur_size=args.blur,
            max_join=args.max_join,
            smooth_type=args.smooth_type,
            smooth_iters=args.smooth_iters,
            smooth_weight=args.smooth_weight,
            smooth_decimate=args.smooth_decimate
        )
        if not success:
            sys.exit(1)
    elif args.smooth:
        success = kdraw.convert_smooth(
            args.input_path, output_path,
            filter_speckle=args.filter_speckle,
            color_precision=args.color_precision,
            corner_threshold=args.corner_threshold,
            path_precision=args.path_precision
        )
        if not success:
            sys.exit(1)
    else:
        num_colors = args.colors
        if not args.lossless and num_colors is None:
            _, ext = os.path.splitext(args.input_path.lower())
            if ext in ('.jpg', '.jpeg'):
                print("WARNING: Input is a JPEG and no quantization is specified.")
                print("JPEG compression noise will prevent efficient pixel merging, producing a huge SVG.")
                print("Defaulting to 64 colors quantization for efficiency. Use --lossless to override.")
                num_colors = 64
        
        kdraw.convert_pixel_perfect(args.input_path, output_path, num_colors=num_colors)

if __name__ == "__main__":
    main()
