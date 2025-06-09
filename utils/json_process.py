import json
import re

class json_process:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_data": ("STRING", {"forceInput": True}),
                "query_1": ("STRING", {"default": "width", "tooltip": "支持key1.key2[0].key3[2]这种查询语法"}),
                "query_2": ("STRING", {"default": "height", "tooltip": "支持key1.key2[0].key3[2]这种查询语法"}),
                "query_3": ("STRING", {"default": "png_text.parameters", "tooltip": "支持key1.key2[0].key3[2]这种查询语法"}),
                "query_4": ("STRING", {"default": "", "tooltip": "支持key1.key2[0].key3[2]这种查询语法"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("value_1", "value_2", "value_3", "value_4",)
    FUNCTION = "select_values"
    CATEGORY = "spawner/utils"

    def _get_value_by_path(self, data, path):
        """
        核心函数，根据路径字符串从数据中提取值,支持key1.key2.key3[114]这种混合查询模式
        """
        try:
            keys = re.split(r'\.(?!\d)|(?=\[)', path)

            current_value = data
            for key in keys:
                if not key: continue
                if key.startswith('[') and key.endswith(']'):
                    index = int(key[1:-1])
                    if isinstance(current_value, list):
                        current_value = current_value[index]
                    else:
                        return f"错误: 尝试对非列表使用索引 {key}"
                else:
                    try:
                        current_value = current_value[key]
                    except (KeyError, TypeError):
                         return f"错误: 未找到键 '{key}'"
            
            if isinstance(current_value, (dict, list)):
                return json.dumps(current_value, indent=2, ensure_ascii=False)
            
            return str(current_value)

        except (KeyError, IndexError, TypeError) as e:
            return f"查询路径 '{path}' 时出错: {e}"
        except Exception as e:
            return f"未知错误: {e}"

    def select_values(self, json_data, query_1, query_2, query_3, query_4):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            return ("无效的 JSON 字符串",) * 4
        
        queries = [query_1, query_2, query_3, query_4]
        results = []

        for query in queries:
            if not query:
                results.append("")
                continue
            
            # 调用核心函数处理每个查询
            results.append(self._get_value_by_path(data, query))
        
        return tuple(results)

