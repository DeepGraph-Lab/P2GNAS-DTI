import torch
# from transformers import RobertaTokenizer, RobertaModel
import time
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel


# --- Helper functions, can be outside the class, as general utilities ---
def get_memory_usage_gb(device):
    """Get the current allocated GPU memory (in GB) for the specified device."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / (1024 ** 3)
    return 0


def get_max_memory_usage_gb(device):
    """Get the peak GPU memory usage (in GB) for the specified device."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return 0


class MolecularEmbedder:
    """
    A class for loading a ChemBERTa model and generating embeddings for SMILES strings.
    """

    def __init__(self, model_checkpoint="DeepChem/ChemBERTa-77M-MTR", use_fp16=False, verbose=True):
        """
        Initialize the embedder and load the model and tokenizer.

        Parameters:
            model_checkpoint (str): Model name to load from the Hugging Face Hub.
            use_fp16 (bool): Whether to use FP16 half-precision to save memory.
            verbose (bool): Whether to print detailed information during loading.
        """
        self.model_checkpoint = model_checkpoint
        self.use_fp16 = use_fp16
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.verbose:
            print(f"Using computation device: {self.device}")

        # Load the model directly when creating an instance
        self._load_model()

    def _load_model(self):
        """Private method responsible for loading the model to the specified device."""
        if self.verbose:
            print(f"Loading model: {self.model_checkpoint} ...")

        try:

            # under code use for "ibm-research/MoLFormer-XL-both-10pct" model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_checkpoint, use_safetensors=True,trust_remote_code=True).to(self.device)

            # under code use for "seyonec/PubChem10M_SMILES_BPE_450k" model
            # self.tokenizer = RobertaTokenizer.from_pretrained(self.model_checkpoint)
            # self.model = RobertaModel.from_pretrained(self.model_checkpoint).to(self.device)
            #

            # some test i forget ,sorry!!!
            # self.tokenizer = RobertaTokenizer.from_pretrained(self.model_checkpoint)
            # self.model = RobertaModel.from_pretrained(self.model_checkpoint, use_safetensors=True).to(self.device)

            if self.use_fp16 and self.device.type == 'cuda':
                self.model.half()
                if self.verbose:
                    print("Model loaded successfully! (Switched to FP16 half-precision)\n")
            elif self.verbose:
                print("Model loaded successfully! (Using FP32 single-precision)\n")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_embedding(self, smiles_string, embedding_type='cls', measure_memory=False):
        """
        Generate an embedding vector for a single SMILES string.

        Parameters:
            smiles_string (str): Input SMILES string.
            embedding_type (str): 'cls' or 'mean'. Choose to return the [CLS] token embedding or the mean-pooled embedding.
            measure_memory (bool): Whether to print GPU memory usage for this inference.

        Returns:
            torch.Tensor: Embedding vector representing the entire molecule.
        """
        if self.device.type == 'cuda' and measure_memory:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            print(f"Peak memory reset before inference starts.")

        # Prepare input and move to device
        inputs = self.tokenizer(smiles_string, return_tensors="pt") #use for "ibm-research/MoLFormer-XL-both-10pct" model


        # In the get_embedding method of the MolecularEmbedder class, for "seyonec/PubChem10M_SMILES_BPE_450k" model use
        inputs = self.tokenizer(
            smiles_string,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )


        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.device.type == 'cuda' and measure_memory:
            print(f"Peak memory after inference: {get_max_memory_usage_gb(self.device):.3f} GB")

        # Extract embedding based on selection
        if embedding_type == 'cls':
            embedding = outputs.last_hidden_state[0, 0, :]
        elif embedding_type == 'mean':
            embedding = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0)
        else:
            raise ValueError("embedding_type must be 'cls' or 'mean'")

        return embedding.cpu()


# ===================================================================
#  Main function - Demo of how to use this class
# ===================================================================
if __name__ == '__main__':
    # 1. Create an instance of the MolecularEmbedder class
    # The model and tokenizer will be downloaded and loaded in this step, only once.
    # verbose=True prints loading information
    embedder = MolecularEmbedder(model_checkpoint="DeepChem/ChemBERTa-77M-MTR", verbose=True)

    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    # 2. Call method A ([CLS] token embedding)
    print("\n--- Testing method A: Get [CLS] embedding ---")
    cls_emb = embedder.get_embedding(caffeine_smiles, embedding_type='cls', measure_memory=True)
    print(f"Retrieved [CLS] embedding shape: {cls_emb.shape}")
    print(f"First 10 dimensions of the vector: {cls_emb[:10]}")

    # 3. Call method B (mean-pooled embedding)
    print("\n--- Testing method B: Get mean-pooled embedding ---")
    mean_emb = embedder.get_embedding(caffeine_smiles, embedding_type='mean', measure_memory=True)
    print(f"Retrieved mean-pooled embedding shape: {mean_emb.shape}")
    print(f"First 10 dimensions of the vector: {mean_emb[:10]}")