from .utils.read_meta import ImageMetadataReader
from .utils.json_process import json_process
from .utils.text_process import TextEncoderDecoder
from .conditioning.node import ConditioningInspector, TensorInspector, ConditioningPacker, ConditioningCrossAttention, TensorShapeAdapter, ConditioningConcatenation, ConditioningPooledMerge, TensorConcatenation, TensorPooledMerge, TensorAttentionFusion, TensorCrossAttention, AllOnesMaskGenerator, TensorUnsqueeze
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
    "ConditioningConcatenation": ConditioningConcatenation,
    "ConditioningPooledMerge": ConditioningPooledMerge,
    "TensorConcatenation": TensorConcatenation,
    "TensorPooledMerge": TensorPooledMerge,
    "TensorAttentionFusion": TensorAttentionFusion,
    "TensorCrossAttention": TensorCrossAttention,
    "AllOnesMaskGenerator": AllOnesMaskGenerator,
    "TensorUnsqueeze": TensorUnsqueeze,

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
    "ConditioningConcatenation": "Conditioning 拼接融合",
    "ConditioningPooledMerge": "Conditioning 池化合并",
    "TensorConcatenation": "张量拼接",
    "TensorPooledMerge": "张量池化合并",
    "TensorAttentionFusion": "张量注意力融合",
    "TensorCrossAttention": "张量交叉注意力",
    "AllOnesMaskGenerator": "全1掩码生成器",
    "TensorUnsqueeze": "张量维度扩展",

    "SaveTrainingDataPair": "保存训练数据对 (UUID命名)",
}
