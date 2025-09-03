import base64
import urllib.parse
import codecs
import json
import xmltodict

class TextEncoderDecoder:
    ENCODING_FORMATS = ["Base64", "URL (Percent-Encoding)", "Hex", "ROT13", "JSON String Escape", "字符集转换 (Character Set)"]
    OPERATIONS = ["解码 (Decode)", "编码 (Encode)"]
    CHARACTER_SETS = ["utf-8", "gbk", "gb2312", "big5", "latin-1", "iso-8859-1", "shift_jis"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "operation": (s.OPERATIONS, ),
                "encoding_format": (s.ENCODING_FORMATS, ),
                "post_decode_unicode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "source_encoding": (s.CHARACTER_SETS, {"default": "gbk"}),
                "target_encoding": (s.CHARACTER_SETS, {"default": "utf-8"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_text",)
    FUNCTION = "process_text"
    CATEGORY = "spawner/utils"

    def process_text(self, text, operation, encoding_format, post_decode_unicode, source_encoding='gbk', target_encoding='utf-8'):
        result = ""
        formats_to_skip_post_decode = ["JSON String Escape", "ROT13", "字符集转换 (Character Set)"]

        try:
            if encoding_format == "Base64":
                if operation == "编码 (Encode)":
                    result = base64.b64encode(text.encode('utf-8')).decode('utf-8')
                else:
                    result = base64.b64decode(text.encode('utf-8')).decode('utf-8', errors='replace')
            
            elif encoding_format == "URL (Percent-Encoding)":
                if operation == "编码 (Encode)":
                    result = urllib.parse.quote(text, encoding='utf-8')
                else:
                    result = urllib.parse.unquote(text, encoding='utf-8')

            elif encoding_format == "Hex":
                if operation == "编码 (Encode)":
                    result = text.encode('utf-8').hex()
                else:
                    result = bytes.fromhex(text).decode('utf-8', errors='replace')

            elif encoding_format == "ROT13":
                result = codecs.encode(text, 'rot_13')
            
            elif encoding_format == "JSON String Escape":
                if operation == "编码 (Encode)":
                    if post_decode_unicode:
                         result = text.encode('unicode-escape').decode('ascii')
                    else:
                         result = json.dumps(text)[1:-1]
                else:
                    result = json.loads(f'"{text}"')
            
            elif encoding_format == "字符集转换 (Character Set)":
                source_bytes = text.encode(source_encoding, errors='replace')
                result = source_bytes.decode(target_encoding, errors='replace')

        except Exception as e:
            print(f"TextEncoderDecoder Error (main processing): {e}")
            result = f"处理错误: {e}"
            return (result,)

        if operation == "解码 (Decode)" and post_decode_unicode and encoding_format not in formats_to_skip_post_decode:
            try:
                result = result.encode('latin-1').decode('unicode-escape')
            except Exception as e:
                print(f"TextEncoderDecoder Error (post-decoding): {e}")

        return (result,)

class XMLJSONConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "conversion_mode": (["xml_to_json", "json_to_xml"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_string",)
    FUNCTION = "convert"
    CATEGORY = "spawner/utils"

    def convert(self, text: str, conversion_mode: str):
        if not text.strip():
            return ("",)

        try:
            if conversion_mode == "xml_to_json":
                py_dict = xmltodict.parse(text)
                json_string = json.dumps(py_dict, indent=4, ensure_ascii=False)
                return (json_string,)
            
            elif conversion_mode == "json_to_xml":
                py_dict = json.loads(text)
                xml_string = xmltodict.unparse(py_dict, pretty=True)
                return (xml_string,)

        except Exception as e:
            print(f"Error during conversion: {e}\nReturning original text.")
            return (text,)
