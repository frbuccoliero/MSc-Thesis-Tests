# This script generates and saves to a file the labels for the CVAT annotation tool
# Labels are geneerated from the files in the 'templates' folder
# The type is "rectangle" to have bounding boxes

'''[
  {
    "name": "<filename>",
    "type": "rectangle",
    "attributes": []
  }
]
'''

import os
import json

# Path to the templates folder
templates_folder = '../dataset/templates'
templates_folder = os.path.join(os.path.dirname(__file__), templates_folder)

# Path to the labels file
labels_file = 'CVAT_labels.json'
labels_file = os.path.join(os.path.dirname(__file__), labels_file)

# Get the list of files in the templates folder
files = os.listdir(templates_folder)

# Create the labels list
labels = []
for file in files:
    labels.append({
        "name": file,
        "type": "rectangle",
        "attributes": []
    })

# Save the labels to the file
with open(labels_file, 'w') as f:
    json.dump(labels, f, indent=2)

print(f'Labels saved to {labels_file}')
