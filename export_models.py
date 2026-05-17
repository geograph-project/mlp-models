import os
import sys
import torch
import json

# Add current directory to path so we can import the models
sys.path.append(os.getcwd())

def export_gallery():
    from gallery.model_def import GeographGalleryModel
    model = GeographGalleryModel()
    # If a .pth exists, load it. Otherwise use random weights
    pth_path = 'gallery/model.pth'
    if os.path.exists(pth_path):
        model, _ = GeographGalleryModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('gallery/weights.json')
    print("Exported gallery model")

def export_assess():
    from assess.model_def import GeographAssessModel
    model = GeographAssessModel()
    pth_path = 'assess/model.pth'
    if os.path.exists(pth_path):
        model, _ = GeographAssessModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('assess/weights.json')
    print("Exported assess model")

def export_clip():
    from clip.model_def import GeographClipModel
    model = GeographClipModel()
    pth_path = 'clip/model.pth'
    if os.path.exists(pth_path):
        model, _ = GeographClipModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('clip/weights.json')
    print("Exported clip model")

def export_scenic():
    from scenic.model_def import GeographScenicModel
    model = GeographScenicModel()
    pth_path = 'scenic/model.pth'
    if os.path.exists(pth_path):
        model, _ = GeographScenicModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('scenic/weights.json')
    print("Exported scenic model")

def export_subject():
    from subject.model_def import GeographSubjectModel
    metadata = {
        "names": ["Urban", "Rural", "Coastal"],
        "id_map": {"0": "U", "1": "R", "2": "C"}
    }
    dist_map = {"far": 0, "near": 1}
    model = GeographSubjectModel(metadata=metadata, dist_map=dist_map)
    pth_path = 'subject/model.pth'
    if os.path.exists(pth_path):
        model, _ = GeographSubjectModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('subject/weights.json')
    print("Exported subject model")

def export_types():
    from model_types.model_def import GeographTypesModel
    classes = ["Aerial", "Close Look", "Cross Far", "Extra", "Geograph", "Inside", "From Drone"]
    dist_map = {"far": 0, "near": 1}
    model = GeographTypesModel(classes=classes, dist_map=dist_map)
    pth_path = 'model_types/model.pth'
    if os.path.exists(pth_path):
        model, _ = GeographTypesModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('model_types/weights.json')
    print("Exported types model")

if __name__ == "__main__":
    export_gallery()
    export_assess()
    export_clip()
    export_scenic()
    export_subject()
    export_types()
