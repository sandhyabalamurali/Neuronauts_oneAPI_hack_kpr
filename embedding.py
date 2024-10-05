from transformers import AutoTokenizer, AutoModel
import torch

class TransformerEmbedding:
    def __init__(self, model_name='medicalai/ClinicalBERT',clean_up_tokenization_spaces=True,weights_only=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_query(self, query):
        return self.embed_documents([query])[0]

    def embed_documents(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()  # Convert to list
        return embeddings

# Example function to get the embedding instance
def get_embedding_function():
    return TransformerEmbedding()
