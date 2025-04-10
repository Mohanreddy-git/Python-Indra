import pandas as pd
metadata_path = "DesignAI-main/DataFiles/metadata_final.xlsx"
df = pd.read_excel(metadata_path)
print(df.head())
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    embedding = response.data[0].embedding
    emb_array = np.array(embedding)
    normalized_emb = emb_array / norm(emb_array)
    return normalized_emb