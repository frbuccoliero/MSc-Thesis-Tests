import os


other_templates_path = os.path.join(os.path.dirname(__file__), 'other_templates')
templates_path = os.path.join(os.path.dirname(__file__), 'templates')

# Remove all files in templates folder that are present in other_templates folder

other_files = os.listdir(other_templates_path)

for file in other_files:
    if os.path.exists(os.path.join(templates_path, file)):
        os.remove(os.path.join(templates_path, file))
        print(f'Removed {file}')
    else:
        print(f'File {file} not found in templates folder')