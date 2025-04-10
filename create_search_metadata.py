import json
import pandas as pd

# Load output.json
with open('python/output.json', 'r') as f:
    output_data = json.load(f)

# Load Excel file
df = pd.read_excel('python/metadata_results.xlsx')

# Get unique image names and their first direct_image_link
image_urls_dict = {}
for _, row in df.iterrows():
    image_name = row['image_name']
    if image_name not in image_urls_dict:
        image_urls_dict[image_name] = row['direct_image_link']

# Create metadata and image_urls arrays
metadata = [item['metadata'] for item in output_data]
image_urls = []

# Match image_urls with metadata
for item in output_data:
    row_index = item['row_index']
    # Find the image name for this row index
    if row_index < len(df['image_name'].unique()):
        image_name = df['image_name'].unique()[row_index]
        if image_name in image_urls_dict:
            image_urls.append(image_urls_dict[image_name])
        else:
            # If no match found, add a placeholder
            image_urls.append("")
    else:
        # If row_index is out of bounds, add a placeholder
        image_urls.append("")

# Create the final data dictionary
data_dict = {
    'metadata': metadata,
    'image_urls': image_urls
}

# Save to search_metadata.json
with open('python/search_metadata.json', 'w') as f:
    json.dump(data_dict, f, indent=2)

print(f"Created search_metadata.json with {len(metadata)} metadata items and {len(image_urls)} image URLs")
