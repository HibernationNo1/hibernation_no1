import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from torch.autograd import Function
from torch.autograd.function import once_differentiable

from sub_module.mmdet.utils import load_ext
ext_module = load_ext('_ext',['roi_align_forward', 'roi_align_backward'])

class RoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale, sampling_ratio,
                 pool_mode, aligned):
        from ..onnx import is_custom_op_loaded
        has_custom_op = is_custom_op_loaded()
        if has_custom_op:
            return g.op(
                'mmcv::MMCVRoiAlign',
                input,
                rois,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=sampling_ratio,
                mode_s=pool_mode,
                aligned_i=aligned)
        else:
            from torch.onnx import TensorProtoDataType
            from torch.onnx.symbolic_helper import _slice_helper
            from torch.onnx.symbolic_opset9 import squeeze, sub

            # batch_indices = rois[:, 0].long()
            batch_indices = _slice_helper(
                g, rois, axes=[1], starts=[0], ends=[1])
            batch_indices = squeeze(g, batch_indices, 1)
            batch_indices = g.op(
                'Cast', batch_indices, to_i=TensorProtoDataType.INT64)
            # rois = rois[:, 1:]
            rois = _slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
            if aligned:
                # rois -= 0.5/spatial_scale
                aligned_offset = g.op(
                    'Constant',
                    value_t=torch.tensor([0.5 / spatial_scale],
                                         dtype=torch.float32))
                rois = sub(g, rois, aligned_offset)
            # roi align
            return g.op(
                'RoiAlign',
                input,
                rois,
                batch_indices,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=max(0, sampling_ratio),
                mode_s=pool_mode)

    @staticmethod
    def forward(ctx,
                input,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                pool_mode='avg',
                aligned=True):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        assert pool_mode in ('max', 'avg')
        ctx.pool_mode = 0 if pool_mode == 'max' else 1
        ctx.aligned = aligned
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        if ctx.pool_mode == 0:
            argmax_y = input.new_zeros(output_shape)
            argmax_x = input.new_zeros(output_shape)
        else:
            argmax_y = input.new_zeros(0)
            argmax_x = input.new_zeros(0)

        ext_module.roi_align_forward(
            input,
            rois,
            output,
            argmax_y,
            argmax_x,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)

        ctx.save_for_backward(rois, argmax_y, argmax_x)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, argmax_y, argmax_x = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        # complex head architecture may cause grad_output uncontiguous.
        grad_output = grad_output.contiguous()
        ext_module.roi_align_backward(
            grad_output,
            rois,
            argmax_y,
            argmax_x,
            grad_input,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)
        return grad_input, None, None, None, None, None, None