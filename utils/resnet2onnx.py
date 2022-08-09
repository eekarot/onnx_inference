import torch
from torchvision.models import resnet18

model = resnet18()
checkpoint = torch.load('../models/classification/resnet18.pth')
model.load_state_dict(checkpoint)
model.eval()

x = torch.randn(1, 3, 224, 224)  #(num,channel,heigh,weight)
export_onnx_file = "../models/classification/resnet18.onnx"		#save path

torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,
                    input_names=["input"],	# input name
                    output_names=["output"],	# output name
                    dynamic_axes={"input":{0:"batch_size"},
                                    "output":{0:"batch_size"}})