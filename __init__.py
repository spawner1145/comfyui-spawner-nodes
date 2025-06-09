from .utils.read_meta import ImageMetadataReader
from .utils.json_process import json_process
from .utils.text_process import TextEncoderDecoder

NODE_CLASS_MAPPINGS = {
    "ImageMetadataReader": ImageMetadataReader,
    "json_process": json_process,
    "TextEncoderDecoder": TextEncoderDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMetadataReader": "Read Image Metadata",
    "json_process": "JSON process",
    "TextEncoderDecoder": "Text Encoder/Decoder",
}
