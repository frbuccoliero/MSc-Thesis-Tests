import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
import argparse

def get_template_colors(templates_path, min_pixel_percentage=0.1, filename_prefix=None, skip_black_white=True, clustering_eps=1):
	png_files = [f for f in os.listdir(templates_path) if f.lower().endswith('.png')]
	if filename_prefix:
		png_files = [f for f in png_files if f.startswith(filename_prefix)]

	# Global list to store all valid colors
	all_valid_colors = []
	processed_files = 0

	# Process each PNG image
	for png_file in png_files:
		img_path = os.path.join(templates_path, png_file)
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		if img is None:
			raise ValueError(f"Could not load image: {img_path}")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		pixel_colors = img.reshape(-1, 3)
		
		# Color analysis
		color_counts = Counter(map(tuple, pixel_colors))
		total_pixels = len(pixel_colors)
		threshold = total_pixels * min_pixel_percentage
		
		# Extract significant colors
		colors_found_in_image = 0
		for color, count in color_counts.items():
			if count >= threshold:
				color_list = list(color)
				if color_list not in all_valid_colors:
					if skip_black_white and (np.all(np.array(color_list) == 0) or np.all(np.array(color_list) == 255)):
						continue
					all_valid_colors.append(color_list)
					colors_found_in_image += 1
			
		processed_files += 1


	# Cluster colors using DBSCAN
	color_clusters = {}
	colors_array = np.array(all_valid_colors)
	scaler = StandardScaler()
	colors_scaled = scaler.fit_transform(colors_array)
	dbscan = DBSCAN(eps=clustering_eps, min_samples=1)
	clusters = dbscan.fit_predict(colors_scaled)
	for color, cluster in zip(all_valid_colors, clusters):
		if cluster not in color_clusters:
			color_clusters[cluster] = []
		color_clusters[cluster].append(color)
	

	# the list to return is now the list of average colors for each cluster
	toReturn = [np.mean(color_clusters[cluster], axis=0) for cluster in color_clusters]
	return toReturn, color_clusters


def load_template_info(template_path, clusters, rotated=False):
	"""
	Load template and extract corner points for homography matching.
	Adjusts corners when template is rotated 45 degrees clockwise.
	"""
	template = cv2.imread(template_path, cv2.IMREAD_COLOR)
	template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
	if template is None:
		raise ValueError(f"Could not load template: {template_path}")
	
	# Define template corners (clockwise from top-left)
	h, w = template.shape[:2]
	
	if not rotated:
		template_corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
	else:
		assert abs(w-h) < 5, "Rotated template must be square"
		template_corners = np.array([[h//2-1, 0], [h-1, h//2-1], [h//2-1, h-1], [0, h//2-1]], dtype=np.float32)

	# temporarly add black and white to the reference colors as single item clusters
	clusters_temp = clusters.copy()
	clusters_temp[len(clusters_temp)] = [[0, 0, 0]]
	clusters_temp[len(clusters_temp)] = [[255, 255, 255]]
	
	colors = []

	# Check if template contains any of the reference colors. If so, add the index of the color to colors.
	# Search inside each cluster for exact matches, not the average color
	template_reshaped = template.reshape(-1, 3)
	for cluster_idx, cluster in clusters_temp.items():
		for color in cluster:
			# Find if any pixel matches this color exactly
			if np.any(np.all(template_reshaped == color, axis=1)):
				colors.append(cluster_idx)
				break
	
	
	return {
		'image': template,
		'corners': template_corners,
		'shape': (h, w),
		'path': template_path,
		'colors': colors,
		'name': template_path.split('/')[-1].split('.')[0]
	}

def get_color_template_freq(templates_info):
  '''
  For each color, count the number of times it appears in the templates.
  Then, sort the colors by frequency, and return the indices of the colors in the order of appearance in the templates (highest frequency first).

  Args: 
    templates_info: list of template info dictionaries, each containing a 'colors' field with color indices

  Return: 
    color_template_freq: array containing the indices of the colors, in order of appearance in the templates (highest frequency first)
  '''
  # Count the frequency of each color index across all templates
  color_counts = Counter()
  for info in templates_info:
    # info['colors'] is a list of color indices for this template
    color_counts.update(info.get('colors', []))

  # Sort color indices by frequency (highest first), then by color index (for stable ordering)
  sorted_colors = sorted(color_counts.items(), key=lambda x: (-x[1], x[0]))

  # Return only the color indices, as a list or numpy array
  color_template_freq = [color_idx for color_idx, count in sorted_colors]
  return color_template_freq

if __name__ == "__main__":
	
	# parse args
	parser = argparse.ArgumentParser()
	parser.add_argument("--pickle_path", type=str, default="data/templates.pkl", help="Path to the pickle file")
	parser.add_argument("--templates_path", type=str, default="data/templates", help="Path to the templates")
	parser.add_argument("--min_pixel_percentage", type=float, default=0.1, help="Minimum pixel percentage")
	parser.add_argument("--clustering_eps", type=float, default=1, help="Clustering epsilon")
	parser.add_argument("--skip_black_white", type=bool, default=True, help="Skip black and white colors")
	parser.add_argument("--rotated", type=bool, default=False, help="Rotate the template")
	parser.add_argument("--filename_prefix", type=str, default=None, help="Filename prefix")

	args = parser.parse_args()

	my_path = os.path.dirname(os.path.abspath(__file__))
	pickle_path = os.path.normpath(os.path.join(my_path, args.pickle_path))
	templates_path = os.path.normpath(os.path.join(my_path, args.templates_path))

	print(f"Pickle path: {pickle_path}")
	print(f"Templates path: {templates_path}")

	reference_colors, clusters = get_template_colors(templates_path, min_pixel_percentage=args.min_pixel_percentage, clustering_eps=args.clustering_eps, skip_black_white=args.skip_black_white, filename_prefix=args.filename_prefix)

	all_templates = []
	for template in sorted(os.listdir(templates_path)):
		if not ".DS_Store" in template:
			template_info = load_template_info(os.path.join(templates_path, template), clusters, rotated=args.rotated)
			all_templates.append(template_info)
		print("Added template: ", template)

	color_template_freq = get_color_template_freq(all_templates)

	pickle_save = (reference_colors, all_templates, color_template_freq)
	
	with open(pickle_path, "wb") as f:
		pickle.dump(pickle_save, f)
	print(f"Saved {len(all_templates)} templates and {len(reference_colors)} reference colors")
	