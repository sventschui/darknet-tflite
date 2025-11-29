# darknet tflite

> Jupyter / Colab Notebooks about exporting darknet models to onnx, tflite and finally edgetpu (coral / ai-edge-lite)

## Running things

### Locally

Execute `run-local.sh` in order to start a colab runtime locally. It will use GPUs on Linux by default but not on Mac OS. Adjust the script if you do not have a CUDA GPU available on your Linux machine.

Open the notebooks in your favourtie Jupyter editor (i.e. VSCode) and connect to the Jupyter kernel with the URL `http://127.0.0.1:9000/?token=supersecret`.

### Colab

TBD

## Buiilding darknet

> Building darknet is optional, the notebooks allow to install a pre-compiled version of darknet.

Use the `build-darknet.ipynb` to build darknet, i.e. to try a new version. Once built, move the package from `/content/build/darknet-5.0.167-Linux.deb` to `/packages`, add the `-cpu` suffix if GPU support has not been enabled during build.

## Exporting to onnx

Exporting an onnx model is done with the native darknet onnx export capabilities introduced in darknet v5.

Use the `darknet-onnx.ipynb` notebook to convert a model to onnx.

## onnx to tensorflow

Converting onnx to tensorflow, tflite and int8 quantization is done using `onnx2tf`.

Use the `onnx-tflite.ipynb` notebook to convert the onnx model to tensorflow / tflite and run int8 quantization.

NOTE: There is no inference script yet for the tensorflow model (.pb), just of the tflite models (.tflite).
