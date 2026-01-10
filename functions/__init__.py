import importlib

import functions.parse_darknet_cfg as parse_darknet_cfg_mod
import functions.postprocess_output as postprocess_output_mod
import functions.preprocess_image as preprocess_image_mod
import functions.visualize_detections as visualize_detections_mod

def parse_darknet_cfg(*args, **kwargs):
    return parse_darknet_cfg_mod.parse_darknet_cfg(*args, **kwargs)

def postprocess_output(*args, **kwargs):
    return postprocess_output_mod.postprocess_output(*args, **kwargs)

def parse_onnx_outputs(*args, **kwargs):
    return postprocess_output_mod.parse_onnx_outputs(*args, **kwargs)

def preprocess_image(*args, **kwargs):
    return preprocess_image_mod.preprocess_image(*args, **kwargs)

def visualize_detections(*args, **kwargs):
    return visualize_detections_mod.visualize_detections(*args, **kwargs)

def reload():
    importlib.reload(parse_darknet_cfg_mod)
    importlib.reload(postprocess_output_mod)
    importlib.reload(preprocess_image_mod)
    importlib.reload(visualize_detections_mod)
