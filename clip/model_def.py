import torch
import torch.nn as nn

'''
ACKNOWLEDGMENTS:
This model logic and architecture are extracted/adapted from the 'ClipTheLandscape' project:
https://github.com/SpaceTimeLab/ClipTheLandscape

The code has been normalized into a reusable modular structure for the Geograph project.
It utilizes pre-computed CLIP embeddings as input to a lightweight MLP (Multi-Layer Perceptron) head.

Note: This specific implementation is hardcoded for the 'mlp-img-txt-mixup' variant,
which uses fused Image + Title CLIP embeddings (ViT-B/32). While the original project
noted that location-fused models scored slightly higher on some metrics, this version prioritizes
computational simplicity by excluding location embeddings.

@misc{ilyankou2025cliplandscapeautomatedtagging,
      title={CLIP the Landscape: Automated Tagging of Crowdsourced Landscape Images},
      author={Ilya Ilyankou and Natchapon Jongwiriyanurak and Tao Cheng and James Haworth},
      year={2025},
      eprint={2506.12214},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.12214},
}

Licence: The code is released under the MIT license.
'''

class GeographClipModel(nn.Module):
    def __init__(self, num_labels = 49):
        super().__init__()

        # these are hardcoded, beucase need to match the data the model was trained on
        self.clip_model = "ViT-B/32" # just for reference
        self.num_labels= 49  #could be len(tags_human)
        self.classes = ["Energy infrastructure", "Coastal", "Lowlands", "Geological interest", "Historic sites and artefacts", "Uplands", "Paths", "Boundary, Barrier", "Sport, Leisure", "Farm, Fishery, Market Gardening", "Grassland", "Rivers, Streams, Drainage", "Village, Rural settlement", "Housing, Dwellings", "Public buildings and spaces", "Religious sites", "Quarrying, Mining", "Park and Public Gardens", "Business, Retail, Services", "Lakes, Wetland, Bog", "Suburb, Urban fringe", "Educational sites", "Wild Animals, Plants and Mushrooms", "Islands", "Roads, Road transport", "Air transport", "Country estates", "City, Town centre", "Railways", "Derelict, Disused", "Health and social services", "Construction, Development", "People, Events", "Burial ground, Crematorium", "Woodland, Forest", "Industry", "Communications", "Water resources", "Air, Sky, Weather", "Waste, Waste management", "Heath, Scrub", "Canals", "Docks, Harbours", "Defence, Military", "Flat landscapes", "Estuary, Marine", "Moorland", "Rocks, Scree, Cliffs", "Barren Plateaux"]

        # Based on 'mlp-img-txt-mixup', we need 512 (img) + 512 (txt) = 1024
        self.input_dim = 1024

        # The specific MLP architecture from the project
        self.head = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_labels)
        )

    def forward(self, img_vec, txt_vec):
        # Concatenate Image and Text embeddings along the feature dimension
        fused = torch.cat([img_vec, txt_vec], dim=1)
        return self.head(fused)


    def predict_api(self, batch_inputs, threshold=0.2):
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # Pull the two required embeddings from our standard batch keys
            img_embeds = batch_inputs['clip-image']
            txt_embeds = batch_inputs['clip-title']

            # 1. Inference
            logits = self.forward(img_embeds, txt_embeds)
            probs = torch.sigmoid(logits).cpu().tolist()
            image_ids = batch_inputs.get('image_ids', [])

            flat_results = []

            # 2. Process results
            for i, img_id in enumerate(image_ids):
                found = False
                for j, prob in enumerate(probs[i]):
                    if prob > threshold:
                        found = True
                        flat_results.append({
                            "image_id": img_id,
                            "model": "clip", #as our first model we just called it 'clip'!
                            "label": self.classes[j] if j < len(self.classes) else f"tag_{j}",
                            "score": float(prob)
                        })

                # 3. Fallback if no tag hits 0.2
                if not found:
                    flat_results.append({
                        "image_id": img_id,
                        "model": "clip",
                        "label": "None",
                        "score": 0.0
                    })

            return flat_results


    @classmethod
    def load_checkpoint(cls, filepath, device='cpu'):

        checkpoint = torch.load(filepath, map_location=device)

        model = cls()

        # (we created a 'module' with the layers as 'head', but the checkpoint
        # may have been created by saving the head directly originally)
        if '0.weight' in checkpoint and 'head.0.weight' not in checkpoint:
            model.head.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model, checkpoint
