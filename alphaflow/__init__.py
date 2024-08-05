import torch
import numpy as np
from collections import defaultdict
from alphaflow.model.wrapper import ESMFoldWrapper
from alphaflow.utils.logging import get_logger
from alphaflow.config import model_config

logger = get_logger(__name__)
torch.set_float32_matmul_precision("high")


class AlphaFlowEmbedding:
    def __init__(self, device="cuda", model_path="esmfold_3B_v1.pt"):
        self.device = torch.device(device)
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        config = model_config("initial_training", train=False, low_prec=True)
        model = ESMFoldWrapper(config, None, training=False)
        model_data = torch.load(model_path, map_location=self.device)
        model_state = model_data["model"]
        model.model.load_state_dict(model_state, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def get_embedding(self, protein_sequence):
        try:
            protein_data = {"seqres": protein_sequence, "name": "test_protein"}
            batch = {
                "seq": torch.tensor([protein_data["seqres"]]),
                "name": protein_data["name"],
            }
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                result = self.model.inference(batch, as_protein=False)
            embedding = result[-1]["embedding_vector"].cpu().numpy()
            return embedding
        except RuntimeError as e:
            logger.error(f"Error processing protein sequence: {str(e)}")
            return None
