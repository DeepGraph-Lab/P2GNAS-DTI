import torch
from transformers import EsmTokenizer, EsmModel
import re
import time


class ProteinEmbedder:
    """
    A class for loading ESM models and generating embeddings for protein sequences.
    Optimized for output control and provides more concise calling methods.
    """

    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D", use_fp16=False, verbose=True):
        """
        Initialize the Embedder.

        Parameters:
        model_name (str): The model name to load.
        use_fp16 (bool): Whether to use FP16 half-precision.
        verbose (bool): Whether to print loading process and other information.
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.verbose:
            print(f"Using computation device: {self.device}")

        self._load_model()

    def _load_model(self):
        """Private method to load the model."""
        if self.verbose:
            print(f"Loading model: {self.model_name} ...")

        self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        self.model = EsmModel.from_pretrained(self.model_name).to(self.device)

        if self.use_fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
            if self.verbose:
                print("Model loaded successfully! (Switched to FP16 half-precision)")
        elif self.verbose:
            print("Model loaded successfully! (Using FP32 single-precision)")

    def get_embedding(self, sequence):
        """
        Core method to generate embeddings for a single protein sequence.

        Returns:
        - per_residue_embeddings (torch.Tensor): Embedding for each amino acid.
        - whole_protein_embedding (torch.Tensor): Mean embedding of the entire protein.
        """
        sequence = re.sub(r"[^A-Z]", "", sequence.upper())
        if not sequence: return None, None

        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        sequence_len = len(sequence)
        per_residue_embeddings = outputs.last_hidden_state[0, 1:sequence_len + 1].cpu()
        whole_protein_embedding = per_residue_embeddings.mean(dim=0).cpu()

        return per_residue_embeddings, whole_protein_embedding

    def get_protein_embedding_vector(self, sequence):
        """
        [Concise method]: Directly obtain the embedding vector representing the entire protein.
        This is the method you are most concerned with.
        """
        _, whole_protein_embedding = self.get_embedding(sequence)
        return whole_protein_embedding


# ===================================================================
#  Testing section
# ===================================================================
if __name__ == '__main__':

    # --- Configuration area ---
    # Your hardware limit choices
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    USE_FP16 = True

    # Sequence to test
    PROTEIN_SEQUENCE = "MNGSAEGPOELNAFGETETSEAWDGALLI"

    # --- 1. Standard call (will print loading info) ---
    print("--- Standard call demo ---")
    # verbose=True by default during initialization
    embedder_verbose = ProteinEmbedder(model_name=MODEL_NAME, use_fp16=USE_FP16)
    # Call the concise method to directly get the final vector
    embedding_vector = embedder_verbose.get_protein_embedding_vector(PROTEIN_SEQUENCE)
    if embedding_vector is not None:
        print(f"\nRetrieved embedding vector shape: {embedding_vector.shape}")
        print(f"First 5 values of the vector: {embedding_vector[:5]}\n")

    # --- 2. Quiet call (no process output) ---
    print("--- Quiet call demo ---")
    # Set verbose=False during initialization to disable all prints
    embedder_quiet = ProteinEmbedder(model_name=MODEL_NAME, use_fp16=USE_FP16, verbose=False)
    # Call again; the console will remain clean this time
    embedding_vector_quiet = embedder_quiet.get_protein_embedding_vector(PROTEIN_SEQUENCE)
    if embedding_vector_quiet is not None:
        print(f"Retrieved embedding vector shape: {embedding_vector_quiet.shape}")
        print("Task completed with no process printing.")