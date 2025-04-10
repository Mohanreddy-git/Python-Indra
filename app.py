from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import torch
import clip
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import os
import json
import pandas as pd
from dotenv import load_dotenv
import time
import random
from numpy.linalg import norm
from openai import OpenAI, RateLimitError
from sklearn.neighbors import NearestNeighbors
import datetime
# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",  # Allow all origins
}})

# # CORS headers
# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5000')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# Load dataset
def load_dataset():
    try:
        excel_path = os.path.join(os.path.dirname(__file__), "metadata_final.xlsx")
        df = pd.read_excel(excel_path)
        print("Dataset loaded. Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

# OpenAI system prompt for metadata generation
system_prompt = '''
    You are an expert interior design metadata generator. You are provided with an image URL (accessible publicly) of a commercial interior design scene and an optional title. Your task is to analyze the image and generate detailed, structured metadata in a valid JSON format with the following keys:

    - "detailed_caption": Provide a detailed descriptive caption that focuses on the furniture in the image. The caption should describe each piece of furniture and any notable design aspects.
    - "objects": Provide a list of objects (i.e. only furniture items) detected in the image. For each object, output a JSON object with the following keys:
        - "type": The type of furniture (e.g., "desk", "chair", "table" etc...).
        - "color": The primary color(s) of the object.
        - "style": The design style (e.g., "modern", "vintage", "industrial" etc...).
        - "material": The main material(s) used (e.g., "wood", "metal", "glass" etc...).
        - "details": Any additional descriptive details (e.g., "ergonomic", "adjustable", "with integrated storage" etc...).
        - "count": The number of such items visible in the image.
    - "keywords": Provide an array of highly relevant search keywords (min. 5) that could be used to search for these images by an interior designer/user in a marketplace/website(covering furniture type, style, material, design features, company, product etc.). for example table with a wodden top, green couch,(describing what they're looking for in simple words) etc... and any other relvant keywords you might think is gonna be used to search

    Do not include any extra commentary or text— only return a valid JSON object.
'''.strip()

# Helper function for OpenAI API calls with backoff
def call_openai_with_backoff(model, messages, max_tokens=600, max_retries=10):
    delay = 1
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response
        except RateLimitError as e:
            print(f"Rate limit error: {e}. Attempt {attempt+1}/{max_retries}. Retrying after {delay:.2f}s...")
            time.sleep(delay)
            delay *= 2 * (1 + random.random())
    raise Exception("Maximum number of retries exceeded for the API call.")

# Clean JSON response from OpenAI
def clean_response_content(content):
    content = content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content

# Generate metadata for an image
def generate_metadata(image_url, title=""):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
        ]}
    ]
    if title.strip():
        messages.append({"role": "user", "content": title})

    response = call_openai_with_backoff(model="gpt-4o-mini", messages=messages, max_tokens=600)
    raw_content = response.choices[0].message.content
    cleaned_content = clean_response_content(raw_content)

    if not cleaned_content or not cleaned_content.startswith("{"):
        raise Exception("Empty or malformed output after cleaning.")
    try:
        metadata = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print("Error decoding JSON. Raw cleaned content:")
        print(cleaned_content)
        raise e
    return metadata

# Process metadata to rows for database
def process_metadata_to_rows(row, metadata):
    base_output = {
        "image_name": row.get("image_filename", ""),
        "direct_image_link": row.get("direct_image_link", ""),
        "company_terms": str(row.get("company_terms", "")).strip(),
        "product_terms": str(row.get("product_terms", "")).strip(),
        "caption": metadata.get("detailed_caption", "")
    }
    output_rows = []
    objects = metadata.get("objects", [])
    if objects:
        for obj in objects:
            obj_row = base_output.copy()
            obj_row["type"] = obj.get("type", "")
            obj_row["colour"] = obj.get("color", "")
            obj_row["style"] = obj.get("style", "")
            obj_row["material"] = obj.get("material", "")
            obj_row["detail"] = obj.get("details", "")
            obj_row["count"] = str(obj.get("count", 0))
            output_rows.append(obj_row)
    else:
        base_output["type"] = ""
        base_output["colour"] = ""
        base_output["style"] = ""
        base_output["material"] = ""
        base_output["detail"] = ""
        base_output["count"] = ""
        output_rows.append(base_output)
    return output_rows

# Load CLIP model for image search
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Global variables for different embedding types
image_embeddings_np = None
text_embeddings_np = None
metadata = None
image_urls = None
image_knn = None  # For CLIP image embeddings (512 dimensions)
text_knn = None   # For OpenAI text embeddings (1536 dimensions)

# Load embeddings and metadata
try:
    # Load image embeddings (CLIP - 512 dimensions)
    image_embeddings_path = os.path.join(os.path.dirname(__file__), 'image_embeddings.npy')
    metadata_path = os.path.join(os.path.dirname(__file__), 'search_metadata.json')

    image_embeddings_np = np.load(image_embeddings_path)
    print(f"Loaded image embeddings with shape: {image_embeddings_np.shape}")

    # Load text embeddings (OpenAI - 1536 dimensions)
    text_embeddings_path = os.path.join(os.path.dirname(__file__), 'text_embeddings.npy')
    text_embeddings_np = np.load(text_embeddings_path)
    print(f"Loaded text embeddings with shape: {text_embeddings_np.shape}")

    with open(metadata_path, 'r') as f:
        data_dict = json.load(f)
        metadata = data_dict['metadata']
        image_urls = data_dict['image_urls']
    print(f"Loaded metadata with {len(metadata)} items and {len(image_urls)} image URLs")

    # Initialize KNN models
    image_knn = NearestNeighbors(n_neighbors=20, metric='cosine')  # Increased to handle duplicates
    image_knn.fit(image_embeddings_np)

    text_knn = NearestNeighbors(n_neighbors=20, metric='cosine')  # Increased to handle duplicates
    text_knn.fit(text_embeddings_np)

    print("Embeddings and KNN models loaded successfully")

except Exception as e:
    print(f"Warning: Could not load embeddings or metadata: {str(e)}")
    print("Search functionality will not be available until embeddings are generated")
    image_embeddings_np = None
    text_embeddings_np = None
    metadata = None
    image_urls = None
    image_knn = None
    text_knn = None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview')
def preview():
    return render_template('image_preview.html')

@app.route('/generate-metadata', methods=['POST'])
def api_generate_metadata():
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        title = data.get('title', '')

        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400

        metadata = generate_metadata(image_url, title)
        return jsonify({'metadata': metadata})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    embedding = response.data[0].embedding
    emb_array = np.array(embedding)
    normalized_emb = emb_array / norm(emb_array)
    return normalized_emb
def search_images(query,df, top_k=5):
    query_emb = get_embedding(query)
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(np.array(x), query_emb))
    grouped = df.groupby("direct_image_link")["similarity"].max().reset_index()
    top_results = grouped.sort_values("similarity", ascending=False).head(top_k)
    return top_results
def parse_embedding(embedding_str):
    try:
        # Extract numbers from the string, handling scientific notation
        import re
        numbers = re.findall(r'-?\d+\.?\d*e?-?\d*', embedding_str)
        float_array = np.array([float(num) for num in numbers])

        if len(float_array) != 1536:
            print(f"Embedding dimension: {len(float_array)}")
            return np.zeros(1536)
        return float_array
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return np.zeros(1536)

@app.route('/search/text', methods=['POST'])
def search_by_text():
    """
    Text-based search endpoint using OpenAI embeddings
    """
    try:
        if text_embeddings_np is None or text_knn is None:
            return jsonify({'error': 'Text search functionality not available. Embeddings not loaded.'}), 503

        data = request.get_json()
        query = data.get('query')
        top_k = data.get('top_k', 5)

        # Get query embedding using OpenAI
        query_emb = get_embedding(query)
        query_emb = query_emb.reshape(1, -1)

        # Find similar items using text_knn - get more than top_k to handle duplicates
        n_neighbors = min(20, text_embeddings_np.shape[0])  # Get more neighbors to handle duplicates
        distances, indices = text_knn.kneighbors(query_emb, n_neighbors=n_neighbors)

        # Debug info
        print(f"Found {len(indices[0])} initial matches before deduplication")

        # Load metadata for results
        df = pd.read_excel(os.path.join(os.path.dirname(__file__), "metadata_final.xlsx"))

        # Format results and handle duplicates
        results = []
        seen_urls = set()  # Track seen image URLs to avoid duplicates
        duplicates_count = 0

        for i, idx in enumerate(indices[0]):
            image_url = df.iloc[idx]['direct_image_link']

            # Skip if we've already seen this URL
            if image_url in seen_urls:
                duplicates_count += 1
                continue

            # Add to results and mark as seen
            seen_urls.add(image_url)

            result = {
                'image_name': str(df.iloc[idx].get('image_name', '')) or 'Untitled',
                'similarity_score': float(1 - distances[0][i]),  # Convert distance to similarity
                'company_terms': str(df.iloc[idx].get('company_terms', '')) or '',
                'product_terms': str(df.iloc[idx].get('product_terms', '')) or '',
                'image_url': image_url or ''
            }
            results.append(result)

            # Stop once we have top_k unique results
            if len(results) >= top_k:
                break

        # Debug info
        print(f"Found {duplicates_count} duplicates. Returning {len(results)} unique results.")

        return jsonify({
            'query': query,
            'results': results
        })

    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/search/image/url', methods=['POST'])
def search_by_image_url():
    """
    Image URL-based search endpoint using CLIP embeddings
    Request body: {"image_url": "url", "top_k": 5}
    """
    try:
        # Check if image search is available
        if image_embeddings_np is None or image_knn is None:
            return jsonify({'error': 'Image search functionality not available. Image embeddings not loaded.'}), 503

        data = request.get_json()
        image_url = data.get('image_url')
        top_k = data.get('top_k', 5)

        # Download and process image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Generate image embedding using CLIP
        with torch.no_grad():
            image_embedding = model.encode_image(image_input)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            image_embedding_np = image_embedding.cpu().numpy()

        # Find similar images using image_knn (not text_knn) - get more than top_k to handle duplicates
        n_neighbors = min(20, image_embeddings_np.shape[0])  # Get more neighbors to handle duplicates
        distances, indices = image_knn.kneighbors(image_embedding_np, n_neighbors=n_neighbors)

        # Format results and handle duplicates
        results = []
        seen_urls = set()  # Track seen image URLs to avoid duplicates
        duplicates_count = 0
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_data_updated.csv"))

        for i, idx in enumerate(indices[0]):
            image_url = df.iloc[idx]['image_link']

            # Skip if we've already seen this URL
            if image_url in seen_urls:
                duplicates_count += 1
                continue

            # Add to results and mark as seen
            seen_urls.add(image_url)

            result = {
                'image_name': str(df.iloc[idx].get('image_filename', '')) or 'Untitled',
                'similarity_score': float(1 - distances[0][i]),  # Convert distance to similarity
                'company_terms': str(df.iloc[idx].get('company_terms', '')) or '',
                'product_terms': str(df.iloc[idx].get('product_terms', '')) or '',
                'image_url': image_url or ''
            }
            results.append(result)

            # Stop once we have top_k unique results
            if len(results) >= top_k:
                break

        # Debug info
        print(f"Found {duplicates_count} duplicates. Returning {len(results)} unique results.")

        return jsonify({'results': results})

    except Exception as e:
        print(f"Image search error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/search/image/upload', methods=['POST'])
def search_by_image_upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Get top_k parameter from form data or use default
        top_k = int(request.form.get('top_k', 5))

        image_file = request.files['image']
        image = Image.open(image_file)
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Generate image embedding with proper normalization
        with torch.no_grad():
            image_embedding = model.encode_image(image_input)
            # Ensure proper normalization
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            image_embedding_np = image_embedding.cpu().numpy()

        # Find similar images - get more than top_k to handle duplicates
        n_neighbors = min(20, image_embeddings_np.shape[0])  # Get more neighbors to handle duplicates
        distances, indices = image_knn.kneighbors(image_embedding_np, n_neighbors=n_neighbors)
        print(f"Distances: {distances}, Indices: {indices}")

        # Format results and handle duplicates
        results = []
        seen_urls = set()  # Track seen image URLs to avoid duplicates
        duplicates_count = 0
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_data_updated.csv"))

        for i, idx in enumerate(indices[0]):
            image_url = df.iloc[idx]['image_link']

            # Skip if we've already seen this URL
            if image_url in seen_urls:
                duplicates_count += 1
                continue

            # Add to results and mark as seen
            seen_urls.add(image_url)

            result = {
                'image_name': str(df.iloc[idx].get('image_filename', '')) or 'Untitled',
                'similarity_score': float(1 - distances[0][i]),  # Convert distance to similarity
                'company_terms': str(df.iloc[idx].get('company_terms', '')) or '',
                'product_terms': str(df.iloc[idx].get('product_terms', '')) or '',
                'image_url': image_url or ''
            }
            results.append(result)

            # Stop once we have top_k unique results
            if len(results) >= top_k:
                break

        # Debug info
        print(f"Found {duplicates_count} duplicates. Returning {len(results)} unique results.")

        return jsonify({'results': results})

    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings():
    """
    Generate embeddings for all images in the dataset
    """
    try:
        df = load_dataset()
        if df is None:
            return jsonify({'error': 'Failed to load dataset'}), 500

        # Check if direct_image_link column exists
        if 'direct_image_link' not in df.columns:
            return jsonify({'error': 'direct_image_link column not found in dataset'}), 500

        # Get image URLs
        global image_urls
        image_urls = df['direct_image_link'].tolist()

        # Prepare metadata
        metadata_list = []
        for _, row in df.iterrows():
            metadata_item = {
                'image_name': row.get('image_filename', ''),
                'company_terms': str(row.get('company_terms', '')).strip(),
                'product_terms': str(row.get('product_terms', '')).strip(),
                'designer_terms': str(row.get('designer_terms', '')).strip(),
                'location': str(row.get('location', '')).strip()
            }
            metadata_list.append(metadata_item)

        # Generate embeddings
        embeddings = []
        for i, url in enumerate(image_urls):
            try:
                print(f"Processing image {i+1}/{len(image_urls)}: {url}")
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                image_input = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_embedding = model.encode_image(image_input)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

                embeddings.append(image_embedding.cpu().numpy()[0])
            except Exception as e:
                print(f"Error processing image {i+1}: {str(e)}")
                # Use a zero embedding as placeholder for failed images
                embeddings.append(np.zeros(512))

        # Save embeddings and metadata
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        embeddings_filename = f'image_embeddings_{timestamp}.npy'
        metadata_filename = f'search_metadata_{timestamp}.json'
        
        # Save embeddings with timestamp
        embeddings_np = np.array(embeddings)
        np.save(embeddings_filename, embeddings_np)

        data_dict = {
            'metadata': metadata_list,
            'image_urls': image_urls
        }
        
        # Save metadata with timestamp
        with open(metadata_filename, 'w') as f:
            json.dump(data_dict, f)

        # Initialize KNN model
        global image_knn, image_embeddings_np, metadata , text_knn
        image_embeddings_np = embeddings_np
        metadata = metadata_list
        image_urls = image_urls
        image_knn = NearestNeighbors(n_neighbors=20, metric='cosine')  # Increased to handle duplicates
        image_knn.fit(embeddings_np)

        return jsonify({'success': True, 'message': f'Generated embeddings for {len(embeddings)} images'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load dataset on startup
    df = load_dataset()
    if df is not None:
        print(f"Dataset loaded with {len(df)} rows")
        print(df[['direct_image_link']].head())
    app.run(debug=True, port=5000)
