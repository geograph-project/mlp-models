#@title the new model
import torch
import torch.nn as nn

class GeographAssessModel(nn.Module):
    def __init__(self, clip_dim=512, pe_dim=1024, hidden_layers=[256]):
        super().__init__()

        # Store dims for checkpointing
        self.hidden_layers = hidden_layers
        self.clip_dim = clip_dim
        self.pe_dim = pe_dim

        layers = []
        input_dim = clip_dim + pe_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = h_dim

        # REGRESSION CHANGE: Final layer is always output_dim=1
        # .. now Multi-Target Regression
        layers.append(nn.Linear(input_dim, 2))

        self.fc = nn.Sequential(*layers)

    def forward(self, clip_vec, pe_vec):
        combined = torch.cat([clip_vec, pe_vec], dim=1)
        # Output will be a tensor of shape [batch_size, 1]
        return self.fc(combined)

    def predict_score(self, clip_vec, dist_idx):
        """Regression helper for inference"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(clip_vec, dist_idx)

            # Use .squeeze() to get a 1D tensor [2] and convert to numpy/list
            # Scaling back to 0.0-10.0 range immediately
            results = scores.squeeze().cpu().numpy() * 10.0

            # Returning as a dict makes the calling code much more readable
            return {
                "aesthetic": float(results[0]),
                "technical": float(results[1])
            }


    def predict_api(self, batch_inputs):
      self.eval()
      with torch.no_grad():
        # 1. Forward pass
        scores = self.forward(batch_inputs['clip-image'], batch_inputs['pe-image'])

        # 2. Move to CPU and convert to a nested Python list [N, 2]
        # This is MUCH faster than accessing a GPU tensor in a loop
        results = scores.cpu().tolist()
        image_ids = batch_inputs.get('image_ids', [])

        flat_results = []

        for i, img_id in enumerate(image_ids):
            # results[i][0] is now a standard Python float
            flat_results.append({
                "image_id": img_id,
                "model": "aesthetic",
                "score": results[i][0] * 10.0,
            })

            flat_results.append({
                "image_id": img_id,
                "model": "technical",
                "score": results[i][1] * 10.0,
            })

        return flat_results


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
            pe_dim=checkpoint['pe_dim'],
            hidden_layers=checkpoint.get('hidden_layers', [256]) #default to the original size
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # Set to evaluation mode by default
        print(f"Model loaded from {filepath} (Epoch: {checkpoint.get('epoch')})")
        return model, checkpoint
