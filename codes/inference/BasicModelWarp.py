import os
import torch
from .BasicAlgorithm import BasicAlgorithm
import numpy as np
import cv2


class Inferrer(BasicAlgorithm):
    scale = 2
    model = None
    stride = 2
    flip_inference = False

    def __init__(self, device="cuda"):
        self.device = device

    def set_scale(self, scale):
        self.scale = scale

    def set_model(self, model):
        self.model = model.to(self.device)

    def single_forward(self, model, inp):
        """PyTorch model forward (single test), it is just a simple warpper
        Args:
            model (PyTorch model)
            inp (Tensor): inputs defined by the model

        Returns:
            output (Tensor): outputs of the model. float, in CPU
        """

        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
        return output

    def flipx4_forward(self, model, inp):
        """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
        Args:
            model (PyTorch model)
            inp (Tensor): inputs defined by the model

        Returns:
            output (Tensor): outputs of the model. float, in CPU
        """
        # normal
        output_f = self.single_forward(model, inp)

        # flip W
        output = self.single_forward(model, torch.flip(inp, (-1,)))
        output_f = output_f + torch.flip(output, (-1,))
        # flip H
        output = self.single_forward(model, torch.flip(inp, (-2,)))
        output_f = output_f + torch.flip(output, (-2,))
        # flip both H and W
        output = self.single_forward(model, torch.flip(inp, (-2, -1)))
        output_f = output_f + torch.flip(output, (-2, -1))

        return output_f / 4

    def infer(self, im):
        # dst_shape = list(im.shape)
        dst_shape = [0, 0]
        if not self.is_infer_multi_frame:
            dst_shape[0] = im.shape[0]*self.scale
            dst_shape[1] = im.shape[1]*self.scale
        else:
            dst_shape[0] = im.shape[1]*self.scale
            dst_shape[1] = im.shape[2]*self.scale

        # inp = im[:, :, ::-1]
        # inp = np.ascontiguousarray(inp)
        inp = im.astype("float32")
        inp = inp/255
        inp = self.pad_input(inp, self.stride)
        inp_tensor = self.np_to_tensor(inp)
        inp_tensor = inp_tensor.to(self.device)

        with torch.no_grad():
            if self.flip_inference:
                prediction = self.flipx4_forward(self.model, inp_tensor)
            else:
                prediction = self.single_forward(self.model, inp_tensor)
            prediction = prediction*255
        result = prediction.detach().cpu().squeeze().clamp(0, 255).numpy().transpose((1, 2, 0))
        result = result[:dst_shape[0], :dst_shape[1]]
        # result = result.astype("uint8")
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result









