import numpy as np
import mxnet as mx
from ast import literal_eval

from operator_py.detectron_bbox_utils import bbox_transform, clip_tiled_boxes


class DecodeBboxOperator(mx.operator.CustomOp):
    def __init__(self, class_agnostic, bbox_mean, bbox_std, xywh):
        super().__init__()
        self._class_agnostic = class_agnostic
        self._bbox_mean = bbox_mean
        self._bbox_std = bbox_std
        self._xywh = xywh
    
    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0].asnumpy()
        bbox_pred = in_data[1].asnumpy()
        im_info = in_data[2].asnumpy()
        inv_stds = list(1.0 / std for std in self._bbox_std)

        batch_size = rois.shape[0]
        bbox = np.zeros_like(bbox_pred, dtype=np.float32)

        for i in range(batch_size):
            bbox[i] = bbox_transform(rois[i], bbox_pred[i], inv_stds, self._xywh)
            bbox[i] = clip_tiled_boxes(bbox[i], im_info[i][:2])
        
        if self._class_agnostic:
            bbox = bbox[:, :, 4:]
        self.assign(out_data[0], req[0], bbox)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError
        

@mx.operator.register('decodebbox')
class DecodeBboxProp(mx.operator.CustomOpProp):
    def __init__(self, class_agnostic, bbox_mean, bbox_std, xywh='True'):
        super().__init__(need_top_grad=False)
        self._xywh = literal_eval(xywh)
        self._class_agnostic = literal_eval(class_agnostic)
        self._bbox_mean = literal_eval(bbox_mean)
        self._bbox_std = literal_eval(bbox_std)
        if self._xywh:
            print('bbox_predict decode type: xywh')
        else:
            print('bbox_predict decode type: xyxy')

    def list_arguments(self):
        return ['rois', 'bbox_pred', 'im_info']
    
    def list_outputs(self):
        return ['bbox']
    
    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        bbox_pred_shape = in_shape[1]
        im_info_shape = in_shape[2]

        batch_size = rois_shape[0]
        num_rois = rois_shape[1]

        if self._class_agnostic:
            bbox_shape = (batch_size, num_rois, 4)
        else:
            bbox_shape = bbox_pred_shape
        return [rois_shape, bbox_pred_shape, im_info_shape], [bbox_shape]
    
    def create_operator(self, ctx, shapes, dtypes):
        return DecodeBboxOperator(
            self._class_agnostic,
            self._bbox_mean,
            self._bbox_std,
            self._xywh
        )
    
    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

