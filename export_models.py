import os
import sys

# we have to do this becore improting torch and json, as they import 're' which looks for the 'types' module

# 1. Pop the automatic current directory off the front of the list
if sys.path and sys.path[0] in ('', os.getcwd(), os.path.dirname(__file__)):
    sys.path.pop(0)

# Add current directory to path so we can import the models
sys.path.append(os.getcwd())

print(sys.path)

import torch
import json
import importlib.util
from pathlib import Path

def export_gallery():
    from gallery.model_def import GeographGalleryModel
    model = GeographGalleryModel()
    # If a .pth exists, load it. Otherwise use random weights
    pth_path = 'gallery/best_geograph_model_gallery_v2.1.pth'
    if os.path.exists(pth_path):
        model, _ = GeographGalleryModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('gallery/weights.json')
    print("Exported gallery model")

def export_assess():
    from assess.model_def import GeographAssessModel
    model = GeographAssessModel()
    pth_path = 'assess/best_assess_model-v1.pth'
    if os.path.exists(pth_path):
        model, _ = GeographAssessModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('assess/weights.json')
    print("Exported assess model")

def export_clip():
    from clip.model_def import GeographClipModel
    model = GeographClipModel()
    pth_path = 'clip/mlp-img-txt-mixup.pth'
    if os.path.exists(pth_path):
        model, _ = GeographClipModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('clip/weights.json')
    print("Exported clip model")

def export_scenic():
    from scenic.model_def import GeographScenicModel
    model = GeographScenicModel()
    pth_path = 'scenic/best_geograph_model_scenic_v1.pth'
    if os.path.exists(pth_path):
        model, _ = GeographScenicModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('scenic/weights.json')
    print("Exported scenic model")

def export_subject():
    from subject.model_def import GeographSubjectModel
    #model = GeographSubjectModel()
    pth_path = 'subject/best_geograph_model_subject_v1.pth'
    if os.path.exists(pth_path):
        model, _ = GeographSubjectModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('subject/weights.json')
    print("Exported subject model")

def export_types():
    # specifcally lookf for the types folder locally
    #from .types.model_def import GeographTypesModel

    file_path = Path("types/model_def.py")
    module_name = "geograph_model"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    GeographTypesModel = module.GeographTypesModel

    model = GeographTypesModel()
    pth_path = 'types/model.pth'
    if os.path.exists(pth_path):
        model, _ = GeographTypesModel.load_checkpoint(pth_path)
    model.save_json_checkpoint('types/weights.json')
    print("Exported types model")

if __name__ == "__main__":
    export_gallery()
    export_assess()
    export_clip()
    export_scenic()
    export_subject()
    export_types()
