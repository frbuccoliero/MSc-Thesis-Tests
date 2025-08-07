from concurrent.futures import ThreadPoolExecutor
import os
import time
import pickle
import argparse 
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import KDTree
from shapely.geometry import Polygon as Poly

from numba import njit

@njit
def compute_color_masks(pixels, reference_RGB, excluded_mask, rgb_threshold_sq):
    n_pixels = pixels.shape[0]
    n_colors = reference_RGB.shape[0]
    masks = np.zeros((n_pixels, n_colors), dtype=np.uint8)

    for i in range(n_pixels):
        if excluded_mask[i]:
            continue

        min_dist_sq = 1e12
        min_idx = -1
        for j in range(n_colors):
            dist_sq = 0.0
            for c in range(3):
                diff = pixels[i, c] - reference_RGB[j, c]
                dist_sq += diff * diff
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                min_idx = j

        if min_dist_sq < rgb_threshold_sq:
            masks[i, min_idx] = 255

    return masks



def get_rgb_masks(full_img, reference_RGB, 
    rgb_threshold, 
    black_val_threshold, 
    white_sat_threshold, 
    white_val_threshold):

    '''
    Combines RGB nearest color for chromatic masks + explicit cuts for black/white.

    Args:
        full_img: RGB image uint8 (H,W,3)
        reference_RGB: list of RGB colors
        rgb_threshold: fraction of max RGB distance (0-1)
        black_val_threshold: max intensity for black (0-255)
        white_sat_threshold: max (std dev / mean) across RGB for white
        white_val_threshold: min intensity for white (0-255)

    Returns:
        list of masks for each reference color + black mask + white mask
    '''
    h, w, _ = full_img.shape
    pixels = full_img.reshape(-1, 3).astype(np.float32)

    # Compute black/white masks
    pixel_vals = pixels.mean(axis=1)
    pixel_std = pixels.std(axis=1)

    black_mask_flat = (pixel_vals < black_val_threshold)
    white_mask_flat = ((pixel_std / (pixel_vals + 1e-5)) < white_sat_threshold) & \
                      (pixel_vals > white_val_threshold)
    excluded_mask = black_mask_flat | white_mask_flat

    # Numba step
    reference_RGB_np = np.array(reference_RGB, dtype=np.float32)
    rgb_threshold_sq = (rgb_threshold * 441.67295593) ** 2
    masks_flat = compute_color_masks(pixels, reference_RGB_np, excluded_mask, rgb_threshold_sq)

    # Reshape outputs
    results_masks = [masks_flat[:, i].reshape(h, w) for i in range(len(reference_RGB))]
    results_masks.append(black_mask_flat.reshape(h, w).astype(np.uint8) * 255)
    results_masks.append(white_mask_flat.reshape(h, w).astype(np.uint8) * 255)

    return results_masks
        
def line_intersection(line1, line2):
	"""
	Finds the intersection of two lines. Each line is defined by two points.
	Returns the intersection point (x, y) or None if lines are parallel.
	"""
	(x1, y1), (x2, y2) = line1
	(x3, y3), (x4, y4) = line2

	den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
	if abs(den) < 1e-6:  # Use a small tolerance for floating point comparison
		return None  # Lines are parallel or collinear

	t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
	t = t_num / den
	
	intersect_x = x1 + t * (x2 - x1)
	intersect_y = y1 + t * (y2 - y1)
	
	return (intersect_x, intersect_y)

def get_internal_angle(p1, p2, p3):
	"""Calculates the internal angle at vertex p2, formed by p1-p2-p3."""
	v1 = np.subtract(p1, p2)
	v2 = np.subtract(p3, p2)

	# suppress warning for division by zero in case of zero-length vectors
	if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
		return 0.0
	
	cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	
	# Clip for numerical stability and convert to degrees
	angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
	return angle

def euclidean_dist(p1, p2):
		return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def is_reasonable_quad(ordered_points, min_angle=60, max_angle=120, min_width_height_ratio=0.5, max_width_height_ratio=2):
	"""
	Checks if a quadrilateral, defined by ordered points, has reasonable
	internal angles. Also checks if the quadrilateral is convex, and has a reasonable width/height ratio.
	"""
	angles = []
	num_points = len(ordered_points)
	if num_points != 4: 
		return False

	# Check convexity using cross products (all z-components should have the same sign)
	def cross_z(a, b, c):
		ab = np.subtract(b, a)
		bc = np.subtract(c, b)
		return ab[0]*bc[1] - ab[1]*bc[0]

	cross_signs = []
	for i in range(num_points):
		p0 = ordered_points[i]
		p1 = ordered_points[(i + 1) % num_points]
		p2 = ordered_points[(i + 2) % num_points]
		cross = cross_z(p0, p1, p2)
		cross_signs.append(np.sign(cross))
	# All cross products should have the same sign (convex)
	if not (all(s > 0 for s in cross_signs) or all(s < 0 for s in cross_signs)):
		return False

	# Compute angles
	for i in range(num_points):
		p_prev = ordered_points[(i - 1 + num_points) % num_points]
		p_curr = ordered_points[i]
		p_next = ordered_points[(i + 1) % num_points]
		angle = get_internal_angle(p_prev, p_curr, p_next)
		angles.append(angle)
	if not all(min_angle <= a <= max_angle for a in angles):
		return False

	# Check width/height ratio
	side1 = euclidean_dist(ordered_points[0], ordered_points[1])
	side2 = euclidean_dist(ordered_points[1], ordered_points[2])
	side3 = euclidean_dist(ordered_points[2], ordered_points[3])
	side4 = euclidean_dist(ordered_points[3], ordered_points[0])

	# Assume opposite sides should be roughly equal, average them
	width = (side1 + side3) / 2
	height = (side2 + side4) / 2

	if height == 0:
			return False
	ratio = width / height
	if not (min_width_height_ratio <= ratio <= max_width_height_ratio):
			return False

	return True

def find_loops(segments, adjacency_radius=None):
		"""
		Greedily searches for 4-line cycles, returning them as ordered paths.
		adjacency_radius: if None, adaptively set to 10% of mean segment length (min 5px).
		angle_prune_range: tuple (min_angle, max_angle) for early pruning between consecutive lines.
		"""
		if not segments:
				return []
		
		# Compute adaptive radius if not provided
		if adjacency_radius is None:
				seg_lengths = [np.linalg.norm(np.subtract(seg[0], seg[1])) for seg in segments]
				mean_length = np.mean(seg_lengths) if seg_lengths else 10
				adjacency_radius = max(5, 0.05 * mean_length)
		
		# Build spatial index
		points = [pt for seg in segments for pt in seg]
		kdtree = KDTree(points)
		
		# Build adjacency graph
		adjacency = defaultdict(set)
		total_connections = 0
		
		for i, segment in enumerate(segments):
			p1, p2 = segment
			for point in [p1, p2]:
				proximal_point_indices = kdtree.query_ball_point(np.array(point).reshape(1, -1), r=adjacency_radius)[0]
				for j in proximal_point_indices:
						other_line_idx = j // 2
						if other_line_idx != i:
								adjacency[i].add(other_line_idx)
								total_connections += 1
		
		# Find 4-line cycles
		loops = []
		seen_loops = set()
		cycles_explored = 0
		
		for a in adjacency:
				for b in adjacency[a]:
						for c in adjacency[b]:
								if c == a:
										continue
								if a in adjacency[c]:
										continue
								for d in adjacency[c]:
										cycles_explored += 1
										if d == a or d == b:
												continue
										if a in adjacency[d]:
												loop = (a, b, c, d)
												canonical_loop = frozenset(loop)
												if canonical_loop not in seen_loops:
														loops.append(loop)
														seen_loops.add(canonical_loop)
		
		return loops

def quad_distance(q1, q2):
		"""Sum of Manhattan distances between corresponding vertices."""
		return sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(q1, q2))

def prune_quadrilaterals(quads, dedup_thresh=10):
		"""
		Prune and deduplicate quadrilaterals.
		- Remove quads fully contained in others.
		- Replace smaller with larger ones.
		- Deduplicate near-identical ones based on vertex distance.
		"""
		processed_polys = []
		processed_quads = []

		quads_poly_obj = [Poly(q).convex_hull for q in quads]
		quads_poly = zip(quads, quads_poly_obj)

		for quad, poly in quads_poly:
				keep_new = True
				to_remove = []

				for i, existing_poly in enumerate(processed_polys):
						if existing_poly.covers(poly):
								keep_new = False
								break
						elif poly.covers(existing_poly):
								to_remove.append(i)

				for idx in reversed(to_remove):
						del processed_polys[idx]
						del processed_quads[idx]

				if keep_new:
						processed_polys.append(poly)
						processed_quads.append(quad)

		# Post-process deduplication: remove near-identical quadrilaterals
		final_quads = []
		seen = []

		for quad in processed_quads:
				is_duplicate = False
				for seen_quad in seen:
						if quad_distance(quad, seen_quad) < dedup_thresh:
								is_duplicate = True
								break
				if not is_duplicate:
						final_quads.append(quad)
						seen.append(quad)

		return final_quads

def sort_diamond_points(pts):
		"""
		Sorts the 4 points of a diamond-like quadrilateral into a canonical
		[Top, Right, Bottom, Left] order, taking into account image coordinates
		(origin at top-left, y increases downward).
		"""
		pts = np.array(pts)
		if pts.shape != (4, 2):
				raise ValueError("Input must be 4 points of shape (4,2)")

		# Find the centroid
		cx, cy = np.mean(pts, axis=0)

		# Compute angle from centroid to each point (atan2 uses (y, x))
		# In image coordinates, y increases downward, so angle 0 is to the right,
		# pi/2 is downward, pi is left, -pi/2 is upward.
		angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)

		# Sort points in clockwise order starting from the top (smallest y)
		# Find the index of the point with the smallest y (topmost)
		top_idx = np.argmin(pts[:,1])
		# Get the angle of the topmost point
		top_angle = angles[top_idx]
		# Shift all angles so that the topmost point is first
		angle_shift = -top_angle
		shifted_angles = (angles + 2*np.pi + angle_shift) % (2*np.pi)
		# Sort indices by shifted angle
		sort_idx = np.argsort(shifted_angles)
		ordered = pts[sort_idx]

		# Now, assign [Top, Right, Bottom, Left] based on their relative positions
		# Top: smallest y
		# Right: largest x among the remaining
		# Bottom: largest y
		# Left: smallest x among the remaining
		top = ordered[np.argmin(ordered[:,1])]
		bottom = ordered[np.argmax(ordered[:,1])]
		# Remove top and bottom from ordered
		remaining = [pt for pt in ordered if not np.all(pt == top) and not np.all(pt == bottom)]
		right = remaining[np.argmax([pt[0] for pt in remaining])]
		left = remaining[np.argmin([pt[0] for pt in remaining])]

		return np.array([top, right, bottom, left], dtype=np.int32)

def get_quadrilaterals_from_loops(loops, segments, debug=False):
	"""
	Processes ordered loops to create a binary mask of valid quadrilaterals.
	"""

	quads = []
	
	for loop_indices in loops:  # e.g., (a, b, c, d)
		lines = [segments[i] for i in loop_indices]
		
		# 1. Find the 4 intersection points (vertices) from the ordered path
		v1 = line_intersection(lines[0], lines[1]) # Intersection of a & b
		v2 = line_intersection(lines[1], lines[2]) # Intersection of b & c
		v3 = line_intersection(lines[2], lines[3]) # Intersection of c & d
		v4 = line_intersection(lines[3], lines[0]) # Intersection of d & a
		
		vertices = [v for v in [v1, v2, v3, v4] if v is not None]
		
		# We need exactly 4 vertices to form a quadrilateral
		if len(vertices) != 4:
			continue
		
		# The vertices are already ordered by path traversal.
		ordered_vertices = sort_diamond_points(vertices)

		# 2. Validate the quad's shape using its internal angles
		if is_reasonable_quad(ordered_vertices):
			quads.append(ordered_vertices)
	
	quads = prune_quadrilaterals(quads)
	return quads

def get_segments(image, min_line_length_percent, scale, num_octaves):
		min_line_length = min(image.shape[0], image.shape[1]) * min_line_length_percent

		# LSD Line Segment Detector
		lsd = cv2.line_descriptor.LSDDetector.createLSDDetector()
		keylines_raw = lsd.detect(image, scale=scale, numOctaves=num_octaves)

		if keylines_raw is None or len(keylines_raw) < 4:
				return []

		# Filter by minimum length
		segments = []

		for line in keylines_raw:
				p1 = np.array([line.startPointX, line.startPointY])
				p2 = np.array([line.endPointX, line.endPointY])
				length = np.linalg.norm(p1 - p2)
				
				if length > min_line_length:
						segments.append(((p1[0], p1[1]), (p2[0], p2[1])))

		return segments

def merge_collinear_segments(segments, angle_tolerance, distance_tolerance):
    """
    Merges nearly collinear and close line segments using an iterative, proximity-aware approach.

    Args:
            segments: A list of line segments, e.g., [((x1, y1), (x2, y2)), ...].
            angle_tolerance: Max angle difference (degrees) to be considered collinear.
            distance_tolerance: Max distance between endpoints to be considered for merging.

    Returns:
            A new list of merged line segments.
    """
    if not segments:
        return []

    # Precompute angles and flatten endpoints
    angles = []
    all_points = []
    for seg in segments:
        (x1, y1), (x2, y2) = seg
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle if angle >= 0 else angle + 180)
        all_points.append((x1, y1))
        all_points.append((x2, y2))

    angles = np.array(angles)
    all_points = np.array(all_points)
    tree = KDTree(all_points)

    used = np.zeros(len(segments), dtype=bool)
    merged_lines = []

    for i, seed in enumerate(segments):
        if used[i]:
            continue

        seed_angle = angles[i]
        current_group = [i]
        used[i] = True
        group_indices = set([i])

        # BFS-like expansion using index sets
        queue = [i]
        while queue:
            idx = queue.pop()
            seg = segments[idx]
            endpoints = np.array([seg[0], seg[1]])
            neighbors_idx = tree.query_ball_point(endpoints, r=distance_tolerance)
            neighbors_idx = set(j // 2 for pair in neighbors_idx for j in pair)  # map back to segment index

            for j in neighbors_idx:
                if used[j] or j == i:
                    continue
                angle_diff = abs(angles[j] - seed_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                if angle_diff < angle_tolerance:
                    used[j] = True
                    queue.append(j)
                    group_indices.add(j)

        # Merge group via PCA
        group_points = [pt for j in group_indices for pt in segments[j]]
        points_np = np.array(group_points, dtype=np.float32)
        mean = np.mean(points_np, axis=0)
        _, _, vt = np.linalg.svd(points_np - mean)
        direction = vt[0]
        projections = np.dot(points_np - mean, direction)
        pt1 = mean + direction * projections.min()
        pt2 = mean + direction * projections.max()
        merged_lines.append((tuple(pt1), tuple(pt2)))

    return merged_lines

def check_template_match(candidate_rect, template_info, image, ncc_threshold):
    """
    Check if a candidate rectangle matches a template using homography and photometric verification.
    """

    # Get quadrilateral vertices
    quad_vertices = candidate_rect['vertices'].astype(np.float32)
    template_corners = template_info['corners']

    # Find homography
    H, mask = cv2.findHomography(
        template_corners, 
        quad_vertices, 
        method=cv2.RANSAC, 
        ransacReprojThreshold=2.0
    )

    if H is None:
        return None

    # Warp template to match candidate
    template_warped = cv2.warpPerspective(
        template_info['image'], 
        H, 
        (image.shape[1], image.shape[0])
    )

    # Extract region of interest
    mask_roi = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [quad_vertices.astype(np.int32)], (255,))

    # Get bounding box for efficiency
    x, y, w, h = cv2.boundingRect(quad_vertices.astype(np.int32))

    # Extract patches
    image_patch = image[y:y+h, x:x+w]
    template_patch = template_warped[y:y+h, x:x+w]
    mask_patch = mask_roi[y:y+h, x:x+w]

    # Photometric verification
    # Convert to grayscale for NCC
    if len(image_patch.shape) == 3:
        image_patch_gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    else:
        image_patch_gray = image_patch

    if len(template_patch.shape) == 3:
        template_patch_gray = cv2.cvtColor(template_patch, cv2.COLOR_BGR2GRAY)
    else:
        template_patch_gray = template_patch

    # Apply mask
    image_patch_gray = cv2.bitwise_and(image_patch_gray, mask_patch)
    template_patch_gray = cv2.bitwise_and(template_patch_gray, mask_patch)

    # Normalized Cross-Correlation
    if image_patch_gray.shape != template_patch_gray.shape:
        return None

    ncc_result = cv2.matchTemplate(image_patch_gray, template_patch_gray, cv2.TM_CCOEFF_NORMED)
    ncc_score = np.max(ncc_result) if ncc_result.size > 0 else 0

    if ncc_score > ncc_threshold:
        print(f"** Found {template_info['name']} with NCC score {ncc_score} **")
        return {
            'vertices': quad_vertices,
            'colors': candidate_rect['colors'],
            'ncc_score': ncc_score,
            'homography': H,
            'bounding_box': (x, y, w, h),
            'template_path': template_info['path'],
            'template_name': template_info['name']
        }
    else:
        print(f"NCC score {ncc_score} for {template_info['name']} is below threshold {ncc_threshold}")
        return None


def deduplicate_quadrilaterals_by_colors(quads, dedup_thresh):
    """
    Deduplicate quadrilaterals (dicts with 'vertices' and 'colors') by merging color lists
    for near-identical quads (within dedup_thresh vertex distance).
    Args:
        quads: list of dicts {'vertices': quad_pts, 'colors': [color_indices]}
        dedup_thresh: float, threshold for considering two quads identical
    Returns:
        List of dicts {'vertices': quad_pts, 'colors': [color_indices]} with merged colors
    """
    deduped = []
    for i, quad in enumerate(quads):
        quad_pts = quad['vertices']
        quad_colors = set(quad['colors'])
        found_duplicate = False
        for j, existing in enumerate(deduped):
            dist = quad_distance(quad_pts, existing['vertices'])
            if dist < dedup_thresh:
                existing['colors'] = sorted(set(existing['colors']).union(quad_colors))
                found_duplicate = True
                break
        if not found_duplicate:
            deduped.append({'vertices': quad_pts, 'colors': sorted(list(quad_colors))})
    return deduped

def symbol_detection_pipeline(image_path, templates, reference_colors, rgb_threshold, black_val_threshold, white_sat_threshold, white_val_threshold, ncc_threshold, min_line_length_percent, adjacency_radius, dedup_thresh, scale, num_octaves, angle_tolerance, distance_tolerance):
    """
    Complete pipeline for symbol detection using color + line-based approach.
    
    Args:
        image_path: Path to input image
        template_paths: List of paths to template images
        reference_colors: Array of reference RGB colors
        rgb_threshold: RGB threshold for color splitting
        black_val_threshold: Black value threshold for color splitting
        white_sat_threshold: White saturation threshold for color splitting
        white_val_threshold: White value threshold for color splitting
        ncc_threshold: NCC threshold for template matching
        min_line_length_percent: Minimum length of a line segment as a percentage of the image width
        adjacency_radius: Radius for finding adjacent lines
        dedup_thresh: Deduplication threshold
        scale: Scale for LSD line segment detector
        num_octaves: Number of octaves for LSD line segment detector
        angle_tolerance: Angle tolerance for merging collinear segments
        distance_tolerance: Distance tolerance for merging collinear segments
    Returns:
        List of detected matches with their information
    """

    time_start = time.time()
        
    # Step 0: Load input image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    time_color_start = time.time()
    print(f"Time taken to load image: {time_color_start - time_start} seconds")
        
    # Step 1-2: Color splitting using RGB
    color_masks = get_rgb_masks(image, reference_colors, rgb_threshold, black_val_threshold, white_sat_threshold, white_val_threshold)

    time_color_end = time.time()
    print(f"Time taken to get {len(color_masks)} color masks: {time_color_end - time_color_start} seconds")

    time_line_start = time.time()
    # Step 3-4: Line detection and rectangle finding
    all_quads = []

    # Parallelize get_segments for each mask, collect per-mask segments in all_segments
    all_segments = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_segments, mask, min_line_length_percent, scale, num_octaves) for mask in color_masks]
        results = [future.result() for future in futures]
        # Each result is a list of segments for that mask; add directly to all_segments
        for segs in results:
            all_segments.append(segs)
    print(f"Found {sum(len(segs) for segs in all_segments)} segments over all masks")
    
    # no need to parallelize this
    for idx_mask, mask in enumerate(color_masks):
        print(f"Post-processing mask {idx_mask}")
        segments = merge_collinear_segments(all_segments[idx_mask], angle_tolerance, distance_tolerance)
        print(f"Found {len(segments)} segments after merging")
        loops = find_loops(segments, adjacency_radius)
        print(f"Found {len(loops)} loops")
        quads = [
            {
            'vertices': np.array(q),
            'colors': [idx_mask],
            }
            for q in get_quadrilaterals_from_loops(loops, segments)
            ]
        all_quads.extend(quads)
        print(f"Found {len(quads)} quads")
    print(f"Found {len(all_quads)} quads over all masks before deduplication")
    all_quads = deduplicate_quadrilaterals_by_colors(all_quads, dedup_thresh)
    print(f"Found {len(all_quads)} quads over all masks after deduplication")

    time_line_end = time.time()
    print(f"Time taken to get {len(all_quads)} quads: {time_line_end - time_line_start} seconds")
    
    time_template_start = time.time()
    # Step 5: Template matching and verification
    final_matches = []
    template_attempts = 0
    successful_matches = 0
    
    for candidate_idx, candidate in enumerate(all_quads):
        print(f"Checking candidate {candidate_idx}")
        candidate_matches = []
        for template_idx, template_info in enumerate(templates):
            template_attempts += 1

            if not all(color in template_info['colors'] for color in candidate['colors']):
                continue

            match_result = check_template_match(
                candidate, template_info, image, ncc_threshold
            )
            if match_result:
                match_result['template_info'] = template_info
                match_result['candidate_idx'] = candidate_idx
                match_result['template_idx'] = template_idx
                candidate_matches.append(match_result)
        if candidate_matches:
            # Select the match with the best (highest) NCC score
            best_match = max(candidate_matches, key=lambda m: m['ncc_score'])
            final_matches.append(best_match)
            successful_matches += 1
    print(f"Found {len(final_matches)} matches")
    time_template_end = time.time()
    print(f"Time taken to get {len(final_matches)} matches: {time_template_end - time_template_start} seconds")

    results = {
        'matches': final_matches,
        'num_candidates': len(all_quads),
        'num_templates': len(templates),
        'template_attempts': template_attempts,
        'image_shape': image.shape,
        'pipeline_config': {
            'ncc_threshold': ncc_threshold,
            'reference_colors': reference_colors
        }
    }
    print(f"Time taken to run pipeline: {time.time() - time_start} seconds")
    return results

def visualize_results(image_path, results, output_path=None):
    """
    Visualize detection results on the original image.
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Draw detected matches
    for i, match in enumerate(results['matches']):
        vertices = match['vertices']
        ncc_score = match['ncc_score']
        
        # Draw quadrilateral
        polygon = Polygon(vertices, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(polygon)
        
        # Add label
        center_x = np.mean(vertices[:, 0])
        center_y = np.mean(vertices[:, 1]) - 10 # offset to avoid overlap with quadrilateral
        ax.text(center_x, center_y, f'#{i} {match["template_name"]} \n NCC: {ncc_score:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                ha='center', va='center', fontsize=8)
    
    ax.set_title(f'Symbol Detection Results - {len(results["matches"])} matches found')
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to the image", required=True)
    parser.add_argument("--pickle", type=str, default="data/templates.pkl", help="Path to the pickle file")
    parser.add_argument("--ncc_threshold", type=float, default=0.9, help="NCC threshold")
    parser.add_argument("--rgb_threshold", type=float, default=0.1, help="RGB threshold for color splitting")
    parser.add_argument("--black_val_threshold", type=float, default=50, help="Black value threshold for color splitting")
    parser.add_argument("--white_sat_threshold", type=float, default=0.2, help="White saturation threshold for color splitting")
    parser.add_argument("--white_val_threshold", type=float, default=180, help="White value threshold for color splitting")
    parser.add_argument("--min_line_length_percent", type=float, default=0.05, help="Minimum line length percentage")
    parser.add_argument("--adjacency_radius", type=float, default=5, help="Adjacency radius")
    parser.add_argument("--dedup_thresh", type=float, default=100, help="Deduplication threshold")
    parser.add_argument("--scale", type=float, default=2, help="Scale for LSD line segment detector")
    parser.add_argument("--num_octaves", type=int, default=4, help="Number of octaves for LSD line segment detector")
    parser.add_argument("--angle_tolerance", type=float, default=5, help="Angle tolerance for merging collinear segments")
    parser.add_argument("--distance_tolerance", type=float, default=5, help="Distance tolerance for merging collinear segments")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualize results")
    args = parser.parse_args()

    pickle_path = os.path.normpath(os.path.join(os.path.dirname(__file__), args.pickle))

    with open(pickle_path, "rb") as f:
        reference_colors, all_templates = pickle.load(f)    

    results = symbol_detection_pipeline(
        args.image, all_templates, reference_colors, 
        rgb_threshold=args.rgb_threshold, 
        black_val_threshold=args.black_val_threshold, 
        white_sat_threshold=args.white_sat_threshold, 
        white_val_threshold=args.white_val_threshold, 
        ncc_threshold=args.ncc_threshold, 
        min_line_length_percent=args.min_line_length_percent, 
        adjacency_radius=args.adjacency_radius, 
        dedup_thresh=args.dedup_thresh, 
        scale=args.scale, 
        num_octaves=args.num_octaves, 
        angle_tolerance=args.angle_tolerance, 
        distance_tolerance=args.distance_tolerance,
    )
    if args.visualize:
        visualize_results(args.image, results)
    else:
        print(f"Found {len(results['matches'])} matches:")
        for i, r in enumerate(results['matches']):
            print(f"- Match {i} - {r['template_info']['name']} - {r['ncc_score']}")