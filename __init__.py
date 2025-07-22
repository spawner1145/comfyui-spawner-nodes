from .utils.read_meta import ImageMetadataReader
from .utils.json_process import json_process
from .utils.text_process import TextEncoderDecoder
from .conditioning.node import ConditioningInspector, TensorInspector, ConditioningPacker, ConditioningCrossAttention, TensorShapeAdapter
from .conditioning.data import SaveTrainingDataPair

NODE_CLASS_MAPPINGS = {
    "ImageMetadataReader": ImageMetadataReader,
    "json_process": json_process,
    "TextEncoderDecoder": TextEncoderDecoder,

    "ConditioningInspector": ConditioningInspector,
    "TensorInspector": TensorInspector,
    "ConditioningPacker": ConditioningPacker,
    "ConditioningCrossAttention": ConditioningCrossAttention,
    "TensorShapeAdapter": TensorShapeAdapter,

    "SaveTrainingDataPair": SaveTrainingDataPair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMetadataReader": "Read Image Metadata",
    "json_process": "JSON process",
    "TextEncoderDecoder": "Text Encoder/Decoder",

    "ConditioningInspector": "Conditioning 信息全览",
    "TensorInspector": "张量信息探针",
    "ConditioningPacker": "Conditioning 构造器",
    "ConditioningCrossAttention": "Conditioning 交叉注意力",
    "TensorShapeAdapter": "张量形状适配器",

    "SaveTrainingDataPair": "保存训练数据对 (UUID命名)",
}
