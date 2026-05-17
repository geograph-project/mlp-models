# @title Updated GeographModel with Save/Load  (use for TRAINING + INFERENCE)
import torch
import torch.nn as nn
import os

class GeographSubjectModel(nn.Module):
    def __init__(self, clip_dim=512, dist_embed_dim=16, hidden_layers=[256], num_classes=None, metadata=None, dist_map=None):
        super().__init__()

        # Store metadata within the class for easy access
        # metadata should be a dict: {"names": [...], "id_map": {...}}
        self.metadata = metadata
        if not num_classes:
            num_classes = len(self.metadata['names'])

        self.dist_map = dist_map if dist_map else {}

        self.dist_emb = nn.Embedding(len(self.dist_map) if self.dist_map else 15, dist_embed_dim)

        # Store this so save_checkpoint can find it
        self.hidden_layers = hidden_layers
        self.clip_dim = clip_dim
        self.dist_embed_dim = dist_embed_dim

        layers = []
        input_dim = clip_dim + dist_embed_dim

        # Dynamically build the layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = h_dim

        # Add the final output layer (no activiation, leave that to be done externally - although the get_top_k demonstrates softmax)
        layers.append(nn.Linear(input_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, clip_vec, dist_idx):
        d_feat = self.dist_emb(dist_idx)
        combined = torch.cat([clip_vec, d_feat], dim=1)
        return self.fc(combined)

    def get_top_k(self, clip_vec, dist_idx, k=10):
        """Helper for inference that returns human-readable results"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(clip_vec, dist_idx)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probs, k=k)

            results = []
            for i in range(k):
                idx = top_idxs[0][i].item()
                results.append({
                    "subject": self.metadata['names'][idx],
                    "subject_id": self.metadata['id_map'][str(idx)], # JSON keys are strings
                    "confidence": top_probs[0][i].item()
                })
            return results


    def predict_api(self, batch_inputs, k=10):
      self.eval()
      with torch.no_grad():
        # 1. Forward pass (Batch operation)
        logits = self.forward(batch_inputs['clip-image'], batch_inputs['distance'])

        # 2. Vectorized Softmax and TopK
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=min(k, probs.size(1)), dim=1)

        # 3. Move to CPU once
        top_probs = top_probs.cpu().numpy()
        top_idxs = top_idxs.cpu().numpy()
        image_ids = batch_inputs.get('image_ids', [])

        # 4. Create a single FLAT list
        flat_results = []

        for i in range(logits.shape[0]):
            img_id = image_ids[i] if i < len(image_ids) else None

            for rank in range(top_probs.shape[1]):
                idx = top_idxs[i][rank]
                # Each dict is a complete row for the server
                flat_results.append({
                    "image_id": img_id,
                    "model": "subjects",
                    "label": self.metadata['names'][idx],
                    "score": float(top_probs[i][rank])
                })

        return flat_results # Simple List[Dict]


    @classmethod
    def load_checkpoint(cls, filepath, device='cpu'):
        """
        Loads the model and metadata.
        Returns an initialized model ready for inference.
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Reconstruct the class with the saved metadata
        model = cls(
            clip_dim=checkpoint['clip_dim'],
            dist_embed_dim=checkpoint['dist_embed_dim'],
            hidden_layers=checkpoint.get('hidden_layers', [256]), #default to the original size
            num_classes=checkpoint['num_classes'],
            metadata=checkpoint['metadata'],
            dist_map=checkpoint['dist_map']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # Set to evaluation mode by default
        print(f"Model loaded from {filepath} (Epoch: {checkpoint.get('epoch')})")
        return model, checkpoint





'''
#@title downloaded the 'released' model from kaggle

## wget https://www.kaggle.com/api/v1/models/barrybhunter/geograph-subject-model/pyTorch/v1/1/download/best_geograph_model_subject_v1.pth

import kagglehub
import shutil
import os

model_local_path = "best_geograph_model_subject_v1.pth"

if not os.path.exists(model_local_path):
    # Download latest version
    path = kagglehub.model_download("barrybhunter/geograph-subject-model/pyTorch/v1")

    print("Path to model files:", path)

    # 2. Extract the actual .pth file path
    # path is a directory, so we look for the file inside it
    model_file = os.path.join(path, model_local_path)

    print(f"Model downloaded to: {model_file}")

    # 3. Optional: Move it to your current folder for easier access
    shutil.copy(model_file, model_local_path)
'''
