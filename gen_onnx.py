from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import torch
from torch import nn as nn
from torch.nn import functional as F

# network of convolutions
class SRVGGNetCompact2(SRVGGNetCompact):
    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        return out

# module for composing final image
class ModelEnd(torch.nn.Module):
    def __init__(self, upscale=4):
        super(ModelEnd, self).__init__()
        self.upscale = upscale
    
    def forward(self, x, out):
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out

if __name__ == '__main__':
    model_path = "weights/realesr-animevideov3.pth"
    
    # Split esrgan network in two parts for webgl backend compatibility
    modelPre = SRVGGNetCompact2(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    loadnet = torch.load(model_path, map_location=torch.device('cpu'))
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    modelPre.load_state_dict(loadnet[keyname], strict=True)
    modelPre.eval()
    
    modelEnd = ModelEnd()
    
    # Export convolutions network
    dummy_input = torch.randn(1, 3, 300, 300)
    dynamic_axes = {
        "input": {2: "height", 3: "width"}
    }
    input_names = [ "input" ]
    output_names = [ "output" ]
    torch.onnx.export(modelPre,
                     (dummy_input),
                     "esrgan-small-pre.onnx",
                     verbose=True,
                     input_names=input_names,
                     output_names=output_names,
                     export_params=True,
                     dynamic_axes=dynamic_axes,
                     opset_version=12
                     )
    
    # Export netowrk final module
    dummy_input_pre = modelPre(dummy_input.detach()).detach()
    print(dummy_input_pre.shape)
    dynamic_axes = {
        "input": {2: "height", 3: "width"},
        "input_pre": {2: "height", 3: "width"}
    }
    input_names = [ "input", "input_pre" ]
    output_names = [ "output" ]
    torch.onnx.export(modelEnd,
                     (dummy_input,dummy_input_pre),
                     "esrgan-small-end.onnx",
                     verbose=True,
                     input_names=input_names,
                     output_names=output_names,
                     export_params=True,
                     dynamic_axes=dynamic_axes,
                     opset_version=12
                     )