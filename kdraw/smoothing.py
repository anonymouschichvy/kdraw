import numpy as np

def smooth_paths_laplacian(path, iterations=3, weight=0.5):
    """
    Apply 1D Laplacian coordinate smoothing (moving average filter) to path.
    For closed loops, it wraps around to keep the loop closed.
    For open paths, endpoints are kept fixed.
    """
    if len(path) < 3:
        return path
        
    smoothed = np.copy(path)
    is_closed = np.linalg.norm(path[0] - path[-1]) < 1.0
    
    for _ in range(iterations):
        temp = np.copy(smoothed)
        if is_closed:
            n = len(temp) - 1
            for i in range(n):
                prev_idx = (i - 1) % n
                next_idx = (i + 1) % n
                smoothed[i] = (1 - weight) * temp[i] + weight * 0.5 * (temp[prev_idx] + temp[next_idx])
            smoothed[-1] = smoothed[0]
        else:
            for i in range(1, len(temp) - 1):
                smoothed[i] = (1 - weight) * temp[i] + weight * 0.5 * (temp[i-1] + temp[i+1])
    return smoothed

def smooth_paths_chaikin(path, iterations=2):
    """
    Apply Chaikin's corner-cutting subdivision algorithm to smooth a path.
    For closed loops, it wraps around to keep the loop closed.
    For open paths, endpoints are kept fixed.
    """
    if len(path) < 3:
        return path
        
    smoothed = np.copy(path)
    is_closed = np.linalg.norm(path[0] - path[-1]) < 1.0
    
    for _ in range(iterations):
        n = len(smoothed)
        new_pts = []
        if is_closed:
            for i in range(n - 1):
                p0 = smoothed[i]
                p1 = smoothed[i+1]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.extend([q, r])
            new_pts.append(new_pts[0])
        else:
            new_pts.append(smoothed[0])
            for i in range(n - 1):
                p0 = smoothed[i]
                p1 = smoothed[i+1]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.extend([q, r])
            new_pts.append(smoothed[-1])
        smoothed = np.array(new_pts, dtype=np.float32)
    return smoothed
