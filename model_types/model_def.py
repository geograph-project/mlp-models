# @title Updated GeographModel with Save/Load  (use for TRAINING + INFERENCE)
import torch
import torch.nn as nn
import os

class GeographTypesModel(nn.Module):
    def __init__(self, clip_dim=512, dist_embed_dim=16, classes=None, dist_map=None):
        super().__init__()
        # Store metadata within the class for easy access
        self.classes = classes if classes else ["Aerial", "Close Look", "Cross Far", "Extra", "Geograph", "Inside", "From Drone"]
        self.clip_dim = clip_dim
        self.dist_embed_dim = dist_embed_dim
        self.dist_map = dist_map if dist_map else {}

        self.dist_emb = nn.Embedding(len(self.dist_map) if self.dist_map else 15, dist_embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(clip_dim + dist_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.classes))
        )

    def forward(self, clip_vec, dist_idx):
        d_feat = self.dist_emb(dist_idx)
        combined = torch.cat([clip_vec, d_feat], dim=1)
        return self.fc(combined)


    def predict_api(self, batch_inputs, threshold=0.5):
      self.eval()
      with torch.no_grad():
        # 1. Forward pass
        outputs = self.forward(batch_inputs['clip-image'], batch_inputs['distance'])

        # 2. Multi-label probabilities
        # Move to CPU and convert to list once for performance
        probs = torch.sigmoid(outputs).cpu().tolist()
        image_ids = batch_inputs.get('image_ids', [])

        flat_results = []

        # 3. Process Batch
        for i, img_id in enumerate(image_ids):
            # Find all indices above threshold
            # Using self.classes as per your __init__
            matches = [
                (self.classes[j], p)
                for j, p in enumerate(probs[i]) if p > threshold
            ]

            # 4. Fallback: If nothing hits threshold, take the single best guess
            if not matches:
                # Find the index of the maximum value in this row
                top_idx = 0
                max_val = -1.0
                for idx, val in enumerate(probs[i]):
                    if val > max_val:
                        max_val = val
                        top_idx = idx
                matches = [(f"{self.classes[top_idx]} (Low)", max_val)]

            # 5. Flatten into rows for the server
            for tag, score in matches:
                flat_results.append({
                    "image_id": img_id,
                    "model": "types",
                    "label": tag,
                    "score": float(score)
                })

        return flat_results


    def save_json_checkpoint(self, filepath):
        import json
        state_dict = self.state_dict()
        serializable_state_dict = {k: v.cpu().tolist() for k, v in state_dict.items()}

        data = {
            "metadata": {
                "clip_dim": self.clip_dim,
                "dist_embed_dim": self.dist_embed_dim,
                "classes": self.classes,
                "dist_map": self.dist_map
            },
            "state_dict": serializable_state_dict
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

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
            classes=checkpoint['classes'],
            dist_map=checkpoint['dist_map']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # Set to evaluation mode by default
        print(f"Model loaded from {filepath} (Epoch: {checkpoint.get('epoch')})")
        return model, checkpoint
