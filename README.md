# Real-ESRGAN webgl experiment
https://javierur.github.io/Real-ESRGAN-webgl-experiment/

A web app for image upscaling made to experiment with the onnx webgl backend

* Uses a small Real-ESRGAN network exported to onnx format
* Process the image in small tiles for faster inference and support for large images
* The network was splitted due to webgl incompatibility with the last function in the network (PixelShuffle)
