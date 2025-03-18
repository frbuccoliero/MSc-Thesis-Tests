# This script generates and saves to a file the labels for the labelstudio annotation tool
# Labels are generated from the files in the 'templates' folder
# Since labelstudio allows for manual input the file output is just a newline separated list of filenames



import os
import json

# Path to the templates folder
templates_folder = '../dataset/templates'
templates_folder = os.path.join(os.path.dirname(__file__), templates_folder)

# Path to the labels file
labels_file = 'labelstudio_labels.txt'
labels_file = os.path.join(os.path.dirname(__file__), labels_file)

# Get the list of files in the templates folder
files = os.listdir(templates_folder)

# Create the labels list
labels = []

# Save the labels to the file
with open(labels_file, 'w') as f:
    for file in files:
        f.write(file + '\n')

print(f'Labels saved to {labels_file}')
