# darknet tflite

> Jupyter / Colab Notebooks about exporting darknet models to onnx, tflite and finally edgetpu (coral / ai-edge-lite)

## About

This repository contains notbooks to train, export, convert and infer with darknet yolo models in the native format, ONNX and a tflite/edgtpu export.

In order to run a model successfully on the edgetpu, the activation is adjusted from leaky ReLu to ReLu. ReLu 6 would be the preferred activation method but it is currently not supported in the darknet ONNX export.

## Running things

### Locally

Execute `run-local.sh` in order to start a colab runtime locally. It will use GPUs on Linux by default but not on Mac OS. Adjust the script if you do not have a CUDA GPU available on your Linux machine.

Open the notebooks in your favourtie Jupyter editor (i.e. VSCode) and connect to the Jupyter kernel with the URL `http://127.0.0.1:9000/?token=supersecret`.

### Colab

TBD

## Building darknet

> Building darknet is optional, the notebooks allow to install a pre-compiled version of darknet.

Use the `build-darknet.ipynb` to build darknet, i.e. to try a new version. Once built, move the package from `/content/build/darknet-5.0.167-Linux.deb` to `/packages`, add the `-cpu` suffix if GPU support has not been enabled during build.

## LegoGears

The `legogears.ipynb` notebook contains code to fine-tune the LegoGears model, export it as an ONNX model, convert it to a TF Lite model and compile it to run on the EdgeTPU.

## Cats

The `cats.ipynb` notebook contains code to train a model on the custom Cats dataset, export it as an ONNX model, convert it to a TF Lite model and compile it to run on the EdgeTPU.
