import os
import json
from PIL import Image
from io import BytesIO
import piexif
import png
import folder_paths

class ImageMetadataReader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata_json",)
    FUNCTION = "read_metadata_from_image"
    CATEGORY = "spawner/utils"

    def read_metadata_from_image(self, image):
        file_path = folder_paths.get_annotated_filepath(image)
        
        result = {}

        try:
            with open(file_path, "rb") as f:
                img_data = f.read()

            with Image.open(BytesIO(img_data)) as img:
                result["filename"] = os.path.basename(file_path)
                result["width"], result["height"] = img.size
                
                try:
                    exif_bytes = img.info.get("exif")
                    if exif_bytes:
                        exif_data = piexif.load(exif_bytes)
                        cleaned_exif = {}
                        for ifd_name in exif_data:
                            if ifd_name == "thumbnail":
                                continue
                            tag_name = piexif.TAGS[ifd_name]
                            cleaned_exif[ifd_name] = {}
                            for tag, value in exif_data[ifd_name].items():
                                try:
                                    decoded_value = value.decode('utf-8', errors='ignore') if isinstance(value, bytes) else value
                                    cleaned_exif[ifd_name][tag_name[tag]["name"]] = decoded_value
                                except:
                                    continue
                        result["exif"] = cleaned_exif if cleaned_exif else "No valid EXIF tags found."
                    else:
                        result["exif"] = "No EXIF metadata found."
                except Exception as e:
                    result["exif"] = f"Error reading EXIF metadata: {e}"

            if file_path.lower().endswith(".png"):
                try:
                    reader = png.Reader(BytesIO(img_data))
                    text_chunks = {}
                    for chunk_type, chunk_data in reader.chunks():
                        if chunk_type in (b'tEXt', b'iTXt'):
                            key, separator, value = chunk_data.partition(b'\x00')
                            decoded_key = key.decode('utf-8', errors='ignore')
                            if decoded_key in ('prompt', 'workflow'):
                                try:
                                    text_chunks[decoded_key] = json.loads(value)
                                except json.JSONDecodeError:
                                    text_chunks[decoded_key] = value.decode('utf-8', errors='ignore')
                            else:
                                text_chunks[decoded_key] = value.decode('utf-8', errors='ignore')

                    result["png_text"] = text_chunks if text_chunks else "No PNG text chunks found."
                except Exception as e:
                    result["png_text"] = f"Error reading PNG text chunks: {e}"

        except Exception as e:
            error_json = json.dumps({"error": f"Failed to read or process image: {e}", "file_path": file_path}, indent=4)
            return (error_json,)
        formatted_json = json.dumps(result, ensure_ascii=False, indent=4)
        
        return (formatted_json,)
