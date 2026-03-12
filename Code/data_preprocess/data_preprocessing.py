"""
Optimized data preprocessing with caching and ChemBERTa drug embeddings.
All expensive computations are cached as pkl files to avoid recomputation.

TO BE EXECUTED FROM MAIN FOLDER (relative paths)
"""

import os
import pandas as pd
import json
import pickle
import logging
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from Code.data_preprocess.ProteinEmbedder import ProteinEmbedder
from Code.data_preprocess.MolecularEmbedder import MolecularEmbedder
from Code.utils import get_sequences, pubchem2smiles_batch

def save_cache(data, filepath, description=""):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved {description} cache: {filepath}")
    except Exception as e:
        logging.warning(f"Failed to save {description} cache: {e}")

def load_cache(filepath, description=""):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Loaded {description} cache: {filepath}")
            return data
        except Exception as e:
            logging.warning(f"Failed to load {description} cache: {e}")
    return None

def get_or_generate_sequences(proteins_unique, cache_dir):
    cache_file = os.path.join(cache_dir, 'protein_sequences.pkl')
    cached_data = load_cache(cache_file, "Protein sequences")
    if cached_data is not None:
        return cached_data
    logging.info("Fetching protein sequences...")
    prot2seq = get_sequences(proteins_unique)
    save_cache(prot2seq, cache_file, "Protein sequences")
    return prot2seq

def get_or_generate_protein_embeddings(protein_features, cache_dir, model_name, use_fp16):
    cache_file = os.path.join(cache_dir, 'protein_embeddings.pkl')
    cached_data = load_cache(cache_file, "Protein embeddings")
    if cached_data is not None and len(cached_data) == len(protein_features):
        return cached_data
    logging.info("Generating protein embeddings...")
    embedder_quiet = ProteinEmbedder(model_name=model_name, use_fp16=use_fp16, verbose=False)
    tqdm.pandas(desc="Generating protein embeddings")
    embeddings = protein_features['Sequence'].progress_apply(
        lambda seq: embedder_quiet.get_protein_embedding_vector(seq).tolist()
    )
    save_cache(embeddings.tolist(), cache_file, "Protein embeddings")
    return embeddings

def get_or_generate_drug_smiles(drugs_unique, cache_dir):
    cache_file = os.path.join(cache_dir, 'drug_smiles.pkl')
    cached_data = load_cache(cache_file, "Drug SMILES")
    if cached_data is not None:
        return cached_data
    logging.info("Fetching drug SMILES...")
    pub2smiles = pubchem2smiles_batch(drugs_unique, size=25)
    save_cache(pub2smiles, cache_file, "Drug SMILES")
    return pub2smiles


def get_or_generate_drug_molformer_embeddings(drug_features, cache_dir, use_fp16):
    """Generate drug embeddings using the MoLFormer model."""

    model_checkpoint = "ibm-research/MoLFormer-XL-both-10pct"
    cache_file = os.path.join(cache_dir, 'drug_embeddings_molformer_xl.pkl')

    cached_data = load_cache(cache_file, "MoLFormer drug embeddings")
    if cached_data is not None and len(cached_data) == len(drug_features):
        return cached_data

    logging.info(f"Generating MoLFormer drug embeddings using model '{model_checkpoint}'...")
    embedder = MolecularEmbedder(model_checkpoint=model_checkpoint, use_fp16=use_fp16, verbose=False)

    tqdm.pandas(desc="Generating MoLFormer [CLS] embeddings")
    embeddings = drug_features['SMILES'].progress_apply(
        lambda smi: embedder.get_embedding(smi, embedding_type='cls').tolist() if smi else None
    )

    valid_embeddings = embeddings.dropna()
    save_cache(valid_embeddings.tolist(), cache_file, "MoLFormer drug embeddings")

    return embeddings


def get_or_generate_drug_seyonec_embeddings(drug_features, cache_dir, use_fp16):
    """Generate drug embeddings using the seyonec/PubChem10M model."""

    model_checkpoint = "seyonec/PubChem10M_SMILES_BPE_450k"
    cache_file = os.path.join(cache_dir, 'drug_embeddings_seyonec_pubchem.pkl')

    cached_data = load_cache(cache_file, "Seyonec (PubChem) drug embeddings")
    if cached_data is not None and len(cached_data) == len(drug_features):
        return cached_data

    logging.info(f"Generating Seyonec (PubChem) drug embeddings using model '{model_checkpoint}'...")
    embedder = MolecularEmbedder(model_checkpoint=model_checkpoint, use_fp16=use_fp16, verbose=False)

    tqdm.pandas(desc="Generating Seyonec (PubChem) [CLS] embeddings")
    embeddings = drug_features['SMILES'].progress_apply(
        lambda smi: embedder.get_embedding(smi, embedding_type='cls').tolist() if smi else None
    )

    valid_embeddings = embeddings.dropna()
    save_cache(valid_embeddings.tolist(), cache_file, "Seyonec (PubChem) drug embeddings")

    return embeddings

def process_database(database):
    logging.info(f'\nProcessing DATABASE: {database}')
    PATH_DTI_FILE = f'Data/Raw/{database}_dtis_pubchem_uniprot.tsv'

    try:
        dti = pd.read_csv(PATH_DTI_FILE, sep='\t')
    except FileNotFoundError:
        logging.error(f"File not found: {PATH_DTI_FILE}")
        return

    dti['Drug'] = dti.Drug.astype(str)

    OUTPUT_PATH = f'Data/{database.upper()}'
    CACHE_DIR = os.path.join(OUTPUT_PATH, 'cache')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    clear_cache = False
    if clear_cache:
        logging.info("Clearing all cache files...")
        cache_files = [
            'protein_sequences.pkl', 'protein_embeddings.pkl',
            'drug_smiles.pkl', 'drug_chemberta_cls_embeddings.pkl'
        ]
        for cache_file in cache_files:
            cache_path = os.path.join(CACHE_DIR, cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logging.info(f"Deleted cache file: {cache_file}")

    #### GENERATE FEATURES: PROTEINS
    PROT_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    PROT_USE_FP16 = False
    proteins_unique = dti.Protein.unique().tolist()
    protein_features = pd.DataFrame(proteins_unique, columns=['Uniprot'])
    prot2seq = get_or_generate_sequences(proteins_unique, CACHE_DIR)
    protein_features['Sequence'] = protein_features.Uniprot.map(prot2seq)
    protein_features = protein_features.dropna()
    protein_embeddings = get_or_generate_protein_embeddings(protein_features, CACHE_DIR, PROT_MODEL_NAME, PROT_USE_FP16)
    protein_features['Ratios'] = protein_embeddings
    available_proteins = protein_features.Uniprot.unique().tolist()
    logging.debug(protein_features.head(3))

    #### GENERATE FEATURES: DRUGS (ChemBERTa)
    drugs_unique = dti.Drug.unique().tolist()
    pub2smiles = get_or_generate_drug_smiles(drugs_unique, CACHE_DIR)
    drug_features = pd.DataFrame(drugs_unique, columns=['PubChem'])
    drug_features['PubChem'] = drug_features.PubChem.astype(str)
    drug_features['SMILES'] = drug_features.PubChem.map(pub2smiles)
    drug_features = drug_features.dropna()

    # ==================== Model Selection Switch ====================
    # !!!!!!!!
    # also change  self.tokenizer and self.model in MolecularEmbedder class

    # local at :    def _load_model(self):
    #         if self.verbose:
    #             print(f"Loading model: {self.model_checkpoint} ...")
    #
    #         try:  this Location you can see

    # !!!!!!!

    # --- Option A: Use the new seyonec/PubChem10M model ---
    # logging.info("--- Using Seyonec/PubChem10M model ---")
    # drug_embeddings = get_or_generate_drug_seyonec_embeddings(drug_features, CACHE_DIR, use_fp16=False)

    # --- Option B: Use the original MoLFormer-XL model ---
    logging.info("--- Using MoLFormer-XL model ---")
    drug_embeddings = get_or_generate_drug_molformer_embeddings(drug_features, CACHE_DIR, use_fp16=False)

    # =======================================================


    drug_features['Embedding'] = drug_embeddings
    drug_features = drug_features.dropna(subset=['Embedding']).reset_index(drop=True)

    available_drugs = drug_features.PubChem.unique().tolist()
    logging.debug(drug_features.head())

    # Filter valid samples
    new_df = dti[dti.Drug.isin(available_drugs) & dti.Protein.isin(available_proteins)]
    new_df = new_df.drop_duplicates(keep='first')
    print(f'New shape after retrieving features for {database}: {new_df.shape}')
    print(f'Unique drugs for {database}: {len(new_df.Drug.unique())}')
    print(f'Unique proteins for {database}: {len(new_df.Protein.unique())}')

    drug_features = drug_features[drug_features.PubChem.isin(new_df.Drug)].reset_index(drop=True)
    protein_features = protein_features[protein_features.Uniprot.isin(new_df.Protein)].reset_index(drop=True)

    # Drug and protein feature matrices
    drug_embeddings_df = pd.DataFrame(drug_features.Embedding.tolist())
    drug_x = torch.tensor(drug_embeddings_df.values, dtype=torch.float32)
    drug_mapping = {index: i for i, index in enumerate(drug_features.PubChem.tolist())}

    protein_features_matrix = pd.concat([protein_features, protein_features.Ratios.apply(pd.Series)], axis=1)
    protein_features_matrix = protein_features_matrix.drop(columns=['Uniprot', 'Sequence', 'Ratios'])
    protein_x = torch.tensor(protein_features_matrix.values, dtype=torch.float32)
    protein_mapping = {index: i for i, index in enumerate(protein_features.Uniprot.tolist())}

    # Index
    src = [drug_mapping[index] for index in new_df['Drug']]
    dst = [protein_mapping[index] for index in new_df['Protein']]
    edge_index = torch.tensor([src, dst])

    # HeteroData object
    data = HeteroData()
    data['drug'].x = drug_x
    data['protein'].x = protein_x
    data['drug', 'interaction', 'protein'].edge_index = edge_index
    print(data)
    print(f'Drug features shape: {drug_x.shape}')
    print(f'Protein features shape: {protein_x.shape}')

    # Determine the device
    if torch.cuda.is_available():
        device = 'cuda'
        data = data.to(device)
        logging.info("Data moved to GPU")
    else:
        device = 'cpu'
        logging.info("Using CPU for processing")

    # SAVING
    logging.info(f'Saving files in {OUTPUT_PATH}')
    with open(os.path.join(OUTPUT_PATH, 'drug_mapping.json'), "w") as outfile:
        json.dump(drug_mapping, outfile)
    with open(os.path.join(OUTPUT_PATH, 'protein_mapping.json'), "w") as outfile:
        json.dump(protein_mapping, outfile)

    # Save HeteroData object to file
    data_to_save = data.cpu() if device == 'cuda' else data
    torch.save(data_to_save, os.path.join(OUTPUT_PATH, f'hetero_data_{database.lower()}.pt'))

    with open(os.path.join(OUTPUT_PATH, f'info_dti_{database.lower()}.out'), "w") as f:
        f.write(f'New shape after retrieving features for {database}: {new_df.shape}\n')
        f.write(f'Unique drugs for {database}: {len(new_df.Drug.unique())}\n')
        f.write(f'Unique proteins for {database}: {len(new_df.Protein.unique())}\n')
        f.write(f'Drug features dimension: {drug_x.shape[1]}\n')
        f.write(f'Protein features dimension: {protein_x.shape[1]}\n')
        f.write(f'Total edges: {edge_index.shape[1]}\n')
        f.write(f'Cache directory: {CACHE_DIR}\n')

    with open(os.path.join(OUTPUT_PATH, f'dti_{database.lower()}.pkl'), 'wb') as f:
        pickle.dump(new_df, f)

    logging.info("All files saved successfully!")
    logging.info(f"Cache files stored in: {CACHE_DIR}")
    logging.info("Next run will use cache automatically, significantly speeding up processing!")

######################################## START MAIN #########################################
def main():
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

    # databases = ['nr','e',  'ic', 'gpcr', 'davis', 'bindingdb', 'biosnap', 'drugbank']

    databases = ['biosnap','bindingdb']

    for database in databases:
        process_database(database)

if __name__ == "__main__":
    main()