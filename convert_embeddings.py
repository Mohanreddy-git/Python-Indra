import pandas as pd
import numpy as np
import json

def convert_embeddings():
    # Load the Excel file containing embeddings
    df = pd.read_excel('../DesignAI-main/DataFiles/embeddings.xlsx')
    
    # Convert string embeddings to numpy arrays
    def parse_embedding(embedding_str):
        try:
            import re
            numbers = re.findall(r'-?\d+\.?\d*e?-?\d*', embedding_str)
            float_array = np.array([float(num) for num in numbers])
            if len(float_array) != 1536:
                print(f"Warning: Embedding dimension: {len(float_array)}")
                return np.zeros(1536)
            return float_array
        except Exception as e:
            print(f"Parsing error: {str(e)}")
            return np.zeros(1536)
    
    print("Converting embeddings to numpy arrays...")
    df['embedding_array'] = df['embedding'].apply(parse_embedding)
    
    # Stack all embeddings into a single numpy array
    embeddings_array = np.stack(df['embedding_array'].values)
    
    # Save embeddings to .npy file
    np.save('image_embeddings.npy', embeddings_array)
    print(f"Saved embeddings array with shape: {embeddings_array.shape}")
    
    # Save metadata separately in JSON format
    metadata_dict = {
        'metadata': df[['image_name', 'company_terms', 'product_terms']].to_dict('records'),
        'image_urls': df['direct_image_link'].tolist()
    }
    
    with open('search_metadata.json', 'w') as f:
        json.dump(metadata_dict, f)
    print("Saved metadata to search_metadata.json")

if __name__ == '__main__':
    convert_embeddings()