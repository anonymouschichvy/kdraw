import numpy as np

def optimize_paths(contours, max_join_dist=0.0):
    """
    Sort and orient contours to minimize pen-up travel distance (greedy TSP solver).
    Optionally merges close endpoints within max_join_dist to eliminate pen lifts.
    Returns list of optimized contours, unoptimized travel, and optimized travel distance.
    """
    if not contours or len(contours) == 0:
        return [], 0.0, 0.0
        
    formatted = [c.reshape(-1, 2) for c in contours if len(c) > 0]
    if not formatted:
        return [], 0.0, 0.0
        
    unopt_travel = 0.0
    for i in range(len(formatted) - 1):
        unopt_travel += np.linalg.norm(formatted[i][-1] - formatted[i+1][0])
        
    optimized = []
    remaining = list(formatted)
    
    current_path = list(remaining.pop(0))
    current_pos = current_path[-1]
    
    opt_travel = 0.0
    
    while remaining:
        starts = np.array([c[0] for c in remaining])
        ends = np.array([c[-1] for c in remaining])
        
        dist_starts = np.sum((starts - current_pos) ** 2, axis=1)
        dist_ends = np.sum((ends - current_pos) ** 2, axis=1)
        
        min_start_idx = np.argmin(dist_starts)
        min_end_idx = np.argmin(dist_ends)
        
        d_start = dist_starts[min_start_idx]
        d_end = dist_ends[min_end_idx]
        
        if d_start <= d_end:
            best_idx = min_start_idx
            best_dist = np.sqrt(d_start)
            reverse_path = False
        else:
            best_idx = min_end_idx
            best_dist = np.sqrt(d_end)
            reverse_path = True
            
        opt_travel += best_dist
        next_path = remaining.pop(best_idx)
        if reverse_path:
            next_path = next_path[::-1]
            
        if max_join_dist > 0.0 and best_dist <= max_join_dist:
            # Merge paths to prevent pen lift
            current_path.extend(next_path)
        else:
            # Save the completed path and start a new one
            optimized.append(np.array(current_path, dtype=np.float32))
            current_path = list(next_path)
            
        current_pos = current_path[-1]
        
    if current_path:
        optimized.append(np.array(current_path, dtype=np.float32))
        
    return optimized, unopt_travel, opt_travel
