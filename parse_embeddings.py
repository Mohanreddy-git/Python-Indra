import pandas as pd
import numpy as np
import json
import re

# Function to parse embedding string into numpy array
def parse_embedding(embedding_str):
    # Remove brackets and ellipsis
    cleaned_str = embedding_str.replace('[', '').replace(']', '').replace('...', '')
    # Split by whitespace and convert to float
    values = [float(val) for val in cleaned_str.split() if val.strip()]
    return np.array(values)

# Load the Excel file with embeddings
print("Loading embeddings from Excel...")
df = pd.read_excel('DesignAI-main/DataFiles/embeddings.xlsx')
print(f"Loaded {len(df)} rows from Excel")

# Check if the embedding column exists
if 'embedding' not in df.columns:
    raise ValueError("Embedding column not found in the Excel file")

# Sample the first embedding to determine the expected dimension
sample_embedding_str = df.iloc[0]['embedding']
sample_embedding = parse_embedding(sample_embedding_str)
expected_dim = len(sample_embedding)
print(f"Expected embedding dimension: {expected_dim}")

# Parse all embeddings
print("Parsing embeddings...")
embeddings = []
valid_indices = []
for i, row in df.iterrows():
    try:
        embedding_str = row['embedding']
        if pd.isna(embedding_str) or not isinstance(embedding_str, str):
            print(f"Skipping row {i}: Invalid embedding format")
            continue
            
        embedding = parse_embedding(embedding_str)
        
        # Verify dimension
        if len(embedding) < expected_dim * 0.9:  # Allow some flexibility
            print(f"Skipping row {i}: Embedding dimension mismatch (got {len(embedding)}, expected ~{expected_dim})")
            continue
            
        # If dimension is larger than expected, truncate
        if len(embedding) > expected_dim:
            embedding = embedding[:expected_dim]
        
        # If dimension is smaller, pad with zeros
        if len(embedding) < expected_dim:
            padding = np.zeros(expected_dim - len(embedding))
            embedding = np.concatenate([embedding, padding])
        
        embeddings.append(embedding)
        valid_indices.append(i)
    except Exception as e:
        print(f"Error parsing embedding at row {i}: {str(e)}")

# Convert to numpy array
embeddings_np = np.array(embeddings)
print(f"Successfully parsed {len(embeddings)} embeddings")
print(f"Embeddings shape: {embeddings_np.shape}")

# Save embeddings as numpy file
np.save('python/image_embeddings.npy', embeddings_np)
print("Saved embeddings to python/image_embeddings.npy")

# Create metadata and image_urls for the valid embeddings
metadata = []
image_urls = []

for idx in valid_indices:
    row = df.iloc[idx]
    
    # Create metadata object
    meta_obj = {
        "detailed_caption": row['caption'],
        "objects": [
            {
                "type": row['type'],
                "color": row['colour'],
                "style": row['style'],
                "material": row['material'],
                "details": row['detail'],
                "count": int(row['count']) if not pd.isna(row['count']) else 1
            }
        ]
    }
    
    # Add keywords if available
    if 'keywords' in row and not pd.isna(row['keywords']):
        meta_obj["keywords"] = row['keywords'].split(', ') if isinstance(row['keywords'], str) else []
    
    metadata.append(meta_obj)
    image_urls.append(row['direct_image_link'])

# Create search_metadata.json
data_dict = {
    'metadata': metadata,
    'image_urls': image_urls
}

with open('python/search_metadata.json', 'w') as f:
    json.dump(data_dict, f, indent=2)

print(f"Created search_metadata.json with {len(metadata)} items")
print("Done!")
