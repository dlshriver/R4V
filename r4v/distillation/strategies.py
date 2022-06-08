from functools import partial
from typing import List, Optional, Tuple, Union

from ..nn import DNN


def forall(model, layer_type, strategy, excluding=None, **params) -> DNN:
    action = partial(getattr(model, strategy), **params)
    if excluding is None:
        excluding = {}
    return model.forall_layers(layer_type, action, excluding=excluding)


def drop_layer(
    model, layer_id: Union[List[int], int], layer_type: Optional[str] = None
) -> DNN:
    if isinstance(layer_id, int):
        layer_id = [layer_id]
    for lid in layer_id:
        model = model.drop_layer(lid, layer_type=layer_type)
    return model


def scale_layer(
    model,
    layer_id: Union[List[int], int],
    factor: float,
    layer_type: Optional[str] = None,
) -> DNN:
    if isinstance(layer_id, int):
        layer_id = [layer_id]
    for lid in layer_id:
        model = model.scale_layer(lid, factor=factor, layer_type=layer_type)
    return model


def scale_input(
    model, factor: Optional[float] = None, shape: Optional[Tuple[int, ...]] = None,
) -> DNN:
    return model.scale_input(factor=factor, shape=shape)


def linearize(
    model, layer_id: Union[List[int], int], layer_type: Optional[str] = None
) -> DNN:
    if isinstance(layer_id, int):
        layer_id = [layer_id]
    for lid in layer_id:
        model = model.linearize(lid, layer_type=layer_type)
    return model


# def drop_operation(model, layer_id, op_type, layer_type=None):
#     if layer_type is not None:
#         model.drop_operation(layer_id, op_type, layer_type=layer_type)
#     else:
#         model.drop_operation(layer_id, op_type)


# def scale_input(model, factor):
#     model.scale_input(factor)


# def scale_layer(model, layer_id, factor, layer_type=None):
#     if layer_type is not None:
#         model.scale_layer(layer_id, factor, layer_type=layer_type)
#     else:
#         model.scale_layer(layer_id, factor)


# def scale_convolution_stride(model, layer_id, factor, layer_type=None):
#     if layer_type is not None:
#         model.scale_convolution_stride(layer_id, factor, layer_type=layer_type)
#     else:
#         model.scale_convolution_stride(layer_id, factor)


# def replace_convolution_padding(model, layer_id, padding, layer_type=None):
#     if layer_type is not None:
#         model.replace_convolution_padding(layer_id, padding, layer_type=layer_type)
#     else:
#         model.replace_convolution_padding(layer_id, padding)
