from pathlib import Path
from time import perf_counter
import glob
import os
from typing import Any, Generator, Tuple
import numpy as np
import darknet
import cv2
from functions import parse_darknet_cfg, postprocess_output, preprocess_image, visualize_detections
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter, load_delegate

CURRENT_DIR = Path(__file__).resolve().parent

class_names = (CURRENT_DIR / "models/LegoGears_nn/LegoGears.names").read_text().splitlines()
class_colors = darknet.class_colors(class_names)

print("Loading darknet...")
darknet_net = darknet.load_net_custom(str(CURRENT_DIR / "models/LegoGears_nn/LegoGears.cfg").encode("ascii"), str(CURRENT_DIR / "models/LegoGears_nn/LegoGears_best.weights").encode("ascii"), 0, 1)

width = darknet.network_width(darknet_net)
height = darknet.network_height(darknet_net)

# Load darknet cfg
net, layers = parse_darknet_cfg(CURRENT_DIR / "models/LegoGears_nn/LegoGears.cfg")
yolo_layers = [layer for layer in layers if layer["type"] == "yolo"]

# ONNX Inference session
# TODO: Build ONNX with XNNPACK support
print("Loading ONNX model...")
onnx_session = session = ort.InferenceSession(CURRENT_DIR / "models/LegoGears.onnx", providers=["CPUExecutionProvider"])

def load_tflite_interpreter(model_path, use_edgetpu=False):
    """
    Load a TensorFlow Lite or EdgeTPU interpreter.
    """
    print(f"Loading tflite model at {model_path}")
    if use_edgetpu:
        delegates = [load_delegate('libedgetpu.so.1')]
        interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates, num_threads=4)
    else:
        interpreter = Interpreter(model_path=model_path, num_threads=4)
    print("... allocate tensors")
    interpreter.allocate_tensors()
    return interpreter

def is_raspberry_pi():
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        return "raspberry pi" in cpuinfo or "bcm" in cpuinfo
    except FileNotFoundError:
        return False

print("loading TF lite models...")
tflite_interpreters = [
    (load_tflite_interpreter(str(CURRENT_DIR / "models/LegoGears_float32.tflite")), "tflite_float32"),
    (load_tflite_interpreter(str(CURRENT_DIR / "models/LegoGears_float16.tflite")), "tflite_float16"),
    (load_tflite_interpreter(str(CURRENT_DIR / "models/LegoGears_full_integer_quant.tflite")), "tflite_int"),
]

if is_raspberry_pi():
    # Use only one edgetpu model at once as otherwise models will be reloaded to the tpu between every image
    tflite_interpreters.append(
        (load_tflite_interpreter(str(CURRENT_DIR / "models/LegoGears_full_integer_quant_edgetpu.tflite"), use_edgetpu=True), "tflite_edgetpu"),
    )
    # manual relu6 conversion shows less detections
    #tflite_interpreters.append(
    #    (load_tflite_interpreter(str(CURRENT_DIR / "models/LegoGears_full_integer_quant_edgetpu_relu6.tflite"), use_edgetpu=True), "tflite_edgetpu_relu6"),
    #)

print("TF lite models loaded")

def load_images() -> Generator[Tuple[str, Any, np.ndarray, float, Tuple[float, float]], None, None]:
    image_paths = glob.glob(str(CURRENT_DIR / "images/*.jpg"))

    for image_path in image_paths:
        if image_path.endswith("-darknet.jpg"):
            continue
        original_image = cv2.imread(image_path)

        np_image, scale, pad, cv_image = preprocess_image(width, height, original_image)

        yield image_path, cv_image, np_image, scale, pad, original_image

def bench_darknet(image_path: str, image: Any) -> Tuple[darknet.DETECTION, float]:
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image.tobytes())
    # Perform object detection on the image using Darknet
    start = perf_counter()
    detections = darknet.detect_image(darknet_net, class_names, darknet_image, thresh=0.25)
    end = perf_counter()
    duration = end - start

    # Free the memory used by the Darknet IMAGE object
    darknet.free_image(darknet_image)

    # Draw bounding boxes and labels on the image based on the detected objects
    image = darknet.draw_boxes(detections, image, class_colors)

    (CURRENT_DIR/"out/darknet").mkdir(exist_ok=True, parents=True)
    (CURRENT_DIR/"out/darknet").mkdir(exist_ok=True, parents=True)
    out_path = Path(image_path.replace("images/", "out/darknet/"))
    out_path.unlink(missing_ok=True)
    cv2.imwrite(str(out_path), image)

    return detections, duration

def bench_onnx(image_path: str, input_tensor: np.ndarray, original_image: Any, scale: float, padding: Tuple[float, float]) -> Tuple[Any, float]:
    input_name = session.get_inputs()[0].name

    start = perf_counter()
    output_tensor = session.run(None, {input_name: input_tensor})
    
    detections = postprocess_output(
        output_tensor,
        yolo_layers_cfg=yolo_layers,
        input_size=(width, height),
        conf_threshold=0.25,
        iou_threshold=0.45,
        scale=scale,
        padding=padding,
        class_names=class_names,
    )

    end = perf_counter()
    duration = end - start

    # Draw bounding boxes and labels on the image based on the detected objects
    image = visualize_detections(
        image=original_image, detections=detections[0], class_names=class_names
    )
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    (CURRENT_DIR/"out/onnx").mkdir(exist_ok=True, parents=True)
    (CURRENT_DIR/"out/onnx").mkdir(exist_ok=True, parents=True)
    out_path = Path(image_path.replace("images/", "out/onnx/"))
    out_path.unlink(missing_ok=True)
    cv2.imwrite(str(out_path), image)

    return detections[0], duration

def dequant(tensor, details):
    if details["dtype"] != np.float32:
        scale_out, zero_point_out = details['quantization']

        return (tensor.astype(np.float32) - zero_point_out) * scale_out
    
    return tensor

def bench_tflite(interpreter: Interpreter, out_name: str, image_path: str, input_tensor: np.ndarray, original_image: Any, scale: float, padding: Tuple[float, float]) -> Tuple[Any, float]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    start = perf_counter()

    yolo_layers_local=yolo_layers
    # Int Quant
    if input_details[0]["dtype"] != np.float32:
        quant_scale, quant_zero = input_details[0]['quantization']

        input_tensor = (input_tensor.astype(np.float32) / quant_scale + quant_zero).round().astype(input_details[0]["dtype"])

        # TODO: Detect output reversion instead of just assuming it for int8
        yolo_layers_local = list(reversed(yolo_layers_local))

    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    interpreter.invoke()

    # Gather outputs (NHWC -> NCHW)
    output_tensor = [np.transpose(dequant(interpreter.get_tensor(o['index']), o), (0, 3, 1, 2)) for o in output_details]

    detections = postprocess_output(
        output_tensor,
        yolo_layers_cfg=yolo_layers_local,
        input_size=(width, height),
        conf_threshold=0.25,
        iou_threshold=0.45,
        scale=scale,
        padding=padding,
        class_names=class_names,
    )

    end = perf_counter()
    duration = end - start

    # Draw bounding boxes and labels on the image based on the detected objects
    image = visualize_detections(
        image=original_image, detections=detections[0], class_names=class_names
    )
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    (CURRENT_DIR/"out"/out_name).mkdir(exist_ok=True, parents=True)
    (CURRENT_DIR/"out"/out_name).mkdir(exist_ok=True, parents=True)
    out_path = Path(image_path.replace("images/", f"out/{out_name}"))
    out_path.unlink(missing_ok=True)
    cv2.imwrite(str(out_path), image)

    return detections[0], duration

def run_benches():
    max_class_len = max(len(s) for s in class_names)

    for image_path, cv_image, np_image, scale, pad, original_image in load_images():
        print(f"Image {image_path}")

        # NCHW -> NHWC
        tflite_image = np.transpose(np_image, (0, 2, 3, 1))

        # === Darktnet
        darknet_detections, darknet_duration = bench_darknet(image_path, cv_image)

        name = "Darknet:".ljust(19)
        print(f"\\_ {name} {round(darknet_duration * 1000)}ms   {len(darknet_detections)} detections")
        for class_name, conf, bbox in darknet_detections:
            print(f"     \\_ {class_name.ljust(max_class_len)}    {round(float(conf) / 100, 2)}")

        # === ONNX
        onnx_detections, onnx_duration = bench_onnx(image_path, np_image, original_image, scale, pad)

        name = "ONNX:".ljust(19)
        print(f"\\_ {name} {round(onnx_duration * 1000)}ms   {len(onnx_detections)} detections")
        for det in onnx_detections:
            print(f"     \\_ {det['class_name'].ljust(max_class_len)}    {round(det['confidence'], 2)}")

        # === TFLite
        for interpreter, out_name in tflite_interpreters:
            tflite_detections, tflite_duration = bench_tflite(
                interpreter=interpreter,
                out_name=out_name,
                image_path=image_path,
                input_tensor=tflite_image,
                original_image=original_image,
                scale=scale, 
                padding=pad
            )

            name = f"{out_name}:".ljust(19)
            print(f"\\_ {name} {round(tflite_duration * 1000)}ms   {len(tflite_detections)} detections")
            for det in tflite_detections:
                print(f"     \\_ {det['class_name'].ljust(max_class_len)}    {round(det['confidence'], 2)}")

        print("")
