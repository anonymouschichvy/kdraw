import collections
import numpy as np

def build_and_prune_graph(skel_bool, min_spur_length=16, collapse_dist=8):
    pixels = set(zip(*np.where(skel_bool)))
    
    # Compute adjacency
    adj = {}
    for p in pixels:
        y, x = p
        candidates = [
            (y-1, x-1), (y-1, x), (y-1, x+1),
            (y, x-1),             (y, x+1),
            (y+1, x-1), (y+1, x), (y+1, x+1)
        ]
        adj[p] = [c for c in candidates if c in pixels]
        
    # Classify pixels
    endpoints = {p for p, neighbors in adj.items() if len(neighbors) == 1}
    junctions = {p for p, neighbors in adj.items() if len(neighbors) >= 3}
    regular = {p for p, neighbors in adj.items() if len(neighbors) == 2}
    
    # Group junction pixels into clusters (each cluster is a node)
    visited_junc = set()
    junc_clusters = []
    for j in junctions:
        if j in visited_junc:
            continue
        cluster = []
        queue = [j]
        visited_junc.add(j)
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            for n in adj[curr]:
                if n in junctions and n not in visited_junc:
                    visited_junc.add(n)
                    queue.append(n)
        junc_clusters.append(cluster)
        
    # Create node mapping: pixel -> node_id
    node_to_pixels = {}
    pixel_to_node = {}
    node_id_counter = 0
    
    for ep in endpoints:
        node_to_pixels[node_id_counter] = [ep]
        pixel_to_node[ep] = node_id_counter
        node_id_counter += 1
        
    for jc in junc_clusters:
        node_to_pixels[node_id_counter] = jc
        for j in jc:
            pixel_to_node[j] = node_id_counter
        node_id_counter += 1
        
    # Trace edges connecting nodes
    edges = []
    edge_id_counter = 0
    visited_regular = set()
    added_direct = set()
    
    def get_node_of_pixel(px):
        return pixel_to_node.get(px, None)
        
    for node_id, node_pxs in node_to_pixels.items():
        for start_px in node_pxs:
            for neighbor in adj[start_px]:
                if neighbor in regular and neighbor not in visited_regular:
                    # Start tracing an edge through regular pixels
                    path = [start_px, neighbor]
                    visited_regular.add(neighbor)
                    curr = neighbor
                    
                    while True:
                        next_candidates = [n for n in adj[curr] if n != path[-2]]
                        if not next_candidates:
                            break
                        next_px = None
                        for n in next_candidates:
                            if n in regular:
                                if n not in visited_regular:
                                    next_px = n
                                    break
                            elif n in pixel_to_node:
                                next_px = n
                                break
                        if next_px is None:
                            break
                            
                        path.append(next_px)
                        if next_px in regular:
                            visited_regular.add(next_px)
                            curr = next_px
                        else:
                            break
                            
                    end_px = path[-1]
                    end_node_id = get_node_of_pixel(end_px)
                    if end_node_id is not None:
                        edges.append({
                            'id': edge_id_counter,
                            'p1': node_id,
                            'p2': end_node_id,
                            'path': path
                        })
                        edge_id_counter += 1
                elif neighbor in pixel_to_node:
                    # Direct node-to-node connection
                    neighbor_node_id = pixel_to_node[neighbor]
                    if node_id != neighbor_node_id:
                        pair = tuple(sorted((node_id, neighbor_node_id)))
                        if pair not in added_direct:
                            added_direct.add(pair)
                            edges.append({
                                'id': edge_id_counter,
                                'p1': node_id,
                                'p2': neighbor_node_id,
                                'path': [start_px, neighbor]
                            })
                            edge_id_counter += 1
                            
    # Find isolated loops
    for p in regular:
        if p not in visited_regular:
            path = [p]
            visited_regular.add(p)
            curr = p
            while True:
                next_candidates = [n for n in adj[curr] if n in regular and n not in visited_regular]
                if not next_candidates:
                    break
                next_px = next_candidates[0]
                path.append(next_px)
                visited_regular.add(next_px)
                curr = next_px
            if len(path) > 2:
                if path[0] in adj[path[-1]]:
                    path.append(path[0])
                    dummy_node = node_id_counter
                    node_to_pixels[dummy_node] = [path[0]]
                    node_id_counter += 1
                    edges.append({
                        'id': edge_id_counter,
                        'p1': dummy_node,
                        'p2': dummy_node,
                        'path': path
                    })
                    edge_id_counter += 1
                    
    # Prune spurs and collapse short edges
    changed = True
    while changed:
        changed = False
        node_degrees = collections.defaultdict(int)
        for e in edges:
            node_degrees[e['p1']] += 1
            node_degrees[e['p2']] += 1
            
        spur_to_remove = None
        for e in edges:
            u, v = e['p1'], e['p2']
            if u == v:
                continue
            deg_u = node_degrees[u]
            deg_v = node_degrees[v]
            length = len(e['path'])
            
            is_spur = False
            if (deg_u == 1 and deg_v >= 3) or (deg_v == 1 and deg_u >= 3):
                is_spur = (length < min_spur_length)
            elif deg_u == 1 and deg_v == 1:
                # Isolated path (i-dots, punctuation). Keep all of them.
                is_spur = False
                
            if is_spur:
                spur_to_remove = e
                break
                
        if spur_to_remove:
            edges.remove(spur_to_remove)
            changed = True
            continue
            
        edge_to_collapse = None
        for e in edges:
            u, v = e['p1'], e['p2']
            if u == v:
                continue
            deg_u = node_degrees[u]
            deg_v = node_degrees[v]
            length = len(e['path'])
            
            if deg_u >= 3 and deg_v >= 3 and length <= collapse_dist:
                edge_to_collapse = e
                break
                
        if edge_to_collapse:
            u = edge_to_collapse['p1']
            v = edge_to_collapse['p2']
            edges.remove(edge_to_collapse)
            for e in edges:
                if e['p1'] == v: e['p1'] = u
                if e['p2'] == v: e['p2'] = u
            node_to_pixels[u].extend(node_to_pixels[v])
            del node_to_pixels[v]
            changed = True
            continue
            
    # Merge degree 2 nodes
    node_degrees = collections.defaultdict(int)
    node_edges = collections.defaultdict(list)
    for e in edges:
        node_edges[e['p1']].append(e)
        node_edges[e['p2']].append(e)
        node_degrees[e['p1']] += 1
        node_degrees[e['p2']] += 1
        
    degree_2_nodes = [node_id for node_id, deg in node_degrees.items() if deg == 2]
    for node_id in degree_2_nodes:
        node_es = node_edges[node_id]
        if len(node_es) == 2:
            e1, e2 = node_es[0], node_es[1]
            if e1['id'] != e2['id']:
                p1_pts = list(e1['path'])
                p2_pts = list(e2['path'])
                shared_pixels = set(node_to_pixels[node_id])
                
                p1_start_in_shared = p1_pts[0] in shared_pixels
                p1_end_in_shared = p1_pts[-1] in shared_pixels
                p2_start_in_shared = p2_pts[0] in shared_pixels
                p2_end_in_shared = p2_pts[-1] in shared_pixels
                
                if p1_end_in_shared and p2_start_in_shared:
                    merged_path = p1_pts[:-1] + p2_pts
                    new_p1 = e1['p1'] if e1['p2'] == node_id else e1['p2']
                    new_p2 = e2['p2'] if e2['p1'] == node_id else e2['p1']
                elif p1_end_in_shared and p2_end_in_shared:
                    merged_path = p1_pts[:-1] + p2_pts[::-1]
                    new_p1 = e1['p1'] if e1['p2'] == node_id else e1['p2']
                    new_p2 = e2['p1'] if e2['p2'] == node_id else e2['p2']
                elif p1_start_in_shared and p2_start_in_shared:
                    merged_path = p1_pts[::-1][:-1] + p2_pts
                    new_p1 = e1['p2'] if e1['p1'] == node_id else e1['p1']
                    new_p2 = e2['p2'] if e2['p1'] == node_id else e2['p1']
                else:
                    merged_path = p2_pts[:-1] + p1_pts
                    new_p1 = e2['p1'] if e2['p2'] == node_id else e2['p2']
                    new_p2 = e1['p2'] if e1['p1'] == node_id else e1['p1']
                    
                edges.remove(e1)
                edges.remove(e2)
                new_edge = {
                    'id': e1['id'],
                    'p1': new_p1,
                    'p2': new_p2,
                    'path': merged_path
                }
                edges.append(new_edge)
                node_edges[new_p1] = [e for e in node_edges[new_p1] if e['id'] not in (e1['id'], e2['id'])] + [new_edge]
                node_edges[new_p2] = [e for e in node_edges[new_p2] if e['id'] not in (e1['id'], e2['id'])] + [new_edge]
                
    return [np.array([(pt[1], pt[0]) for pt in e['path']], dtype=np.float32) for e in edges]
