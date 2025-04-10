from flask import Flask, request, jsonify
import torch
import clip
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from sklearn.neighbors import NearestNeighbors
import os
import json
# Add these imports at the top
from flask import render_template , send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": ["http://127.0.0.1:5000", "http://localhost:5000", "http://localhost:3000"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5000')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Add new route for proxying images
@app.route('/proxy-image', methods=['POST'])
def proxy_image():
    try:
        data = request.get_json()
        image_url = data.get('url')

        # Download image using the existing function
        image_content = download_from_gdrive(image_url)

        # Create BytesIO object from image content
        image_io = BytesIO(image_content)

        # Send file with correct mimetype
        return send_file(
            image_io,
            mimetype='image/jpeg',  # Adjust if needed for other formats
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400
# Update the loading section in your app.py
@app.route('/')
def index():
    return render_template('index.html')
# Add this function after the imports
def download_from_gdrive(url):
    """
    Download image from Google Drive link
    """
    try:
        # Convert to direct download URL
        if 'drive.google.com' in url:
            if 'file/d/' in url:
                file_id = url.split('file/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            else:
                raise ValueError("Invalid Google Drive URL format")

            url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # Download with headers to avoid permission issues
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, stream=True)

        # Handle Google Drive download page
        if 'Content-Disposition' not in response.headers:
            token = response.cookies.get('download_warning')
            if token:
                response = requests.get(f"{url}&confirm={token}", headers=headers)

        return response.content
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")

# Load pre-trained model and embeddings
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load embeddings and metadata separately
try:
    image_embeddings_np = np.load('python/image_embeddings.npy')
    print(f"Loaded embeddings with shape: {image_embeddings_np.shape}")

    with open('python/search_metadata.json', 'r') as f:
        data_dict = json.load(f)
        metadata = data_dict['metadata']
        image_urls = data_dict['image_urls']
    print(f"Loaded metadata with {len(metadata)} items and {len(image_urls)} image URLs")

    if len(metadata) != image_embeddings_np.shape[0] or len(image_urls) != image_embeddings_np.shape[0]:
        print(f"WARNING: Mismatch between embeddings ({image_embeddings_np.shape[0]}) and metadata ({len(metadata)}) or image URLs ({len(image_urls)})")

    # Ensure all arrays have the same length
    min_length = min(image_embeddings_np.shape[0], len(metadata), len(image_urls))
    image_embeddings_np = image_embeddings_np[:min_length]
    metadata = metadata[:min_length]
    image_urls = image_urls[:min_length]
    print(f"Using {min_length} items for search")

except Exception as e:
    print(f"Error loading embeddings or metadata: {str(e)}")
    print("Please run python/parse_embeddings.py to generate the required files")
    image_embeddings_np = np.zeros((1, 512))
    metadata = [{}]
    image_urls = [""]


# Initialize KNN model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(image_embeddings_np)

@app.route('/search/text', methods=['POST'])
def search_by_text():
    """
    Text-based search endpoint
    Request body: {"query": "text query", "top_k": 5}
    """
    try:
        data = request.get_json()
        query = data.get('query')
        top_k = data.get('top_k')

        # Encode text query
        text_tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_embedding = model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        # Compute similarities
        similarity_scores = torch.matmul(
            torch.tensor(image_embeddings_np),
            text_embedding.cpu().T
        ).squeeze(1)

        # Get top results
        top_k_values, top_k_indices = similarity_scores.topk(top_k)

        results = [
            {
                'metadata': metadata[idx],
                'image_url': image_urls[idx],
                'similarity_score': float(top_k_values[i])
            }
            for i, idx in enumerate(top_k_indices)
        ]

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/search/image/url', methods=['POST'])
def search_by_image_url():
    """
    Image URL-based search endpoint
    Request body: {"image_url": "url", "top_k": 5}
    """
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        top_k = data.get('top_k', 5)

        # Download and process image
        image_content = download_from_gdrive(image_url)
        image = Image.open(BytesIO(image_content))
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Generate image embedding
        with torch.no_grad():
            image_embedding = model.encode_image(image_input)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Find similar images
        distances, indices = knn.kneighbors(image_embedding.cpu().numpy())

        results = [
            {
                'metadata': metadata[idx],
                'image_url': image_urls[idx],
                'distance': float(distances[0][i])
            }
            for i, idx in enumerate(indices[0])
        ]

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/search/image/upload', methods=['POST'])
def search_by_image_upload():
    """
    Image upload-based search endpoint
    Request body: multipart/form-data with image file
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        top_k = int(request.form.get('top_k', 5))

        # Process uploaded image
        image = Image.open(image_file)
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Generate image embedding
        with torch.no_grad():
            image_embedding = model.encode_image(image_input)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Find similar images
        distances, indices = knn.kneighbors(image_embedding.cpu().numpy())

        results = [
            {
                'metadata': metadata[idx],
                'image_url': image_urls[idx],
                'distance': float(distances[0][i])
            }
            for i, idx in enumerate(indices[0])
        ]

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/search/knn', methods=['POST'])
def search_by_knn():
    """
    KNN search endpoint
    Request body: {"embedding": [array], "top_k": 5}
    """
    try:
        data = request.get_json()
        query_embedding = np.array(data.get('embedding'))
        top_k = data.get('top_k', 5)

        # Perform KNN search
        distances, indices = knn.kneighbors(query_embedding.reshape(1, -1))

        results = [
            {
                'metadata': metadata[idx],
                'image_url': image_urls[idx],
                'distance': float(distances[0][i])
            }
            for i, idx in enumerate(indices[0])
        ]

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)