from openai import OpenAI


class OpenAIEmbeddingModel:
    """Turns text into vectors using an OpenAI embedding model."""

    def __init__(self, model: str = "text-embedding-3-small", client: OpenAI | None = None):
        self.model = model
        self.client = client or OpenAI()

    def embed(self, text: str) -> list[float]:
        """Embed ONE text -> one vector."""
        return self.embed_all([text])[0]

    def embed_all(self, texts: list[str]) -> list[list[float]]:
        """Embed MANY texts in a single API call -> one vector per text."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]