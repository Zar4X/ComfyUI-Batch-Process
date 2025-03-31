import re
import os


class TextModifyTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING",),
            },
            "optional": {
                "search": ("STRING", {"default": ""}),
                "change_to": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ""}),
                "option": (
                    ["whole texts", "prefix", "suffix"],
                    {"default": "whole texts"},
                ),
                "delete": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "modify_text"
    CATEGORY = "text_processing"

    def find_separator(self, text, find_last=False):
        """查找第一个或最后一个分隔符位置"""
        sep_chars = ["-", "_", "——", "#"]
        target_pos = -1
        found_sep = ""
        for sep in sep_chars:
            pos = text.rfind(sep) if find_last else text.find(sep)
            if pos == -1:
                continue
            if (find_last and pos > target_pos) or (
                not find_last and (target_pos == -1 or pos < target_pos)
            ):
                target_pos = pos
                found_sep = sep
        return target_pos, found_sep

    def modify_text(
        self,
        input_text,
        search="",
        change_to="",
        delimiter="",
        option="whole texts",
        delete=False,
    ):
        # 优先处理search功能
        if search:
            replacement = "" if delete else change_to

            if option == "whole texts":
                # 分离文件名和扩展名，仅处理文件名部分
                filename, ext = os.path.splitext(input_text)
                processed = filename.replace(search, replacement)
                return (processed,)

            elif option == "prefix":
                if delimiter:
                    parts = input_text.split(delimiter, 1)
                    sep = delimiter
                else:
                    # 自动检测第一个分隔符
                    pos, sep = self.find_separator(input_text)
                    if pos != -1:
                        parts = [input_text[:pos], input_text[pos + len(sep) :]]
                    else:
                        parts = [input_text]

                # 处理分割结果
                if len(parts) > 1:
                    prefix, rest = parts[0], sep + parts[1]
                else:
                    prefix, rest = input_text, ""

                # 执行前缀替换
                new_prefix = prefix.replace(search, replacement)
                return (new_prefix + rest,)

            elif option == "suffix":
                if delimiter:
                    parts = input_text.rsplit(delimiter, 1)
                    sep = delimiter
                else:
                    # 自动检测最后一个分隔符
                    pos, sep = self.find_separator(input_text, find_last=True)
                    if pos != -1:
                        parts = [input_text[:pos], input_text[pos + len(sep) :]]
                    else:
                        parts = [input_text]

                # 处理分割结果
                if len(parts) > 1:
                    rest, suffix = parts[0] + sep, parts[1]
                else:
                    rest, suffix = "", input_text

                # 执行后缀替换
                new_suffix = suffix.replace(search, replacement)
                return (rest + new_suffix,)

        # 处理没有search的情况
        if option == "whole texts":
            # 分离文件名和扩展名，仅处理文件名部分
            filename, ext = os.path.splitext(input_text)
            output_text = change_to if delete or change_to else filename
        elif option == "prefix":
            if delimiter:
                parts = input_text.split(delimiter, 1)
            else:
                parts = re.split(r"[-_——#]", input_text, 1)
            if delete:
                output_text = parts[1] if len(parts) > 1 else input_text
            else:
                output_text = (
                    f"{change_to}{delimiter if delimiter else '_'}{parts[1]}"
                    if len(parts) > 1
                    else change_to
                )
        elif option == "suffix":
            if delimiter:
                parts = input_text.rsplit(delimiter, 1)
            else:
                parts = re.split(r"[-_——#]", input_text)
                if len(parts) > 1:
                    last_delimiter_index = max(
                        input_text.rfind(d) for d in ["-", "_", "——", "#"]
                    )
                    parts = [
                        input_text[:last_delimiter_index],
                        input_text[last_delimiter_index + 1 :],
                    ]
            if delete:
                output_text = parts[0] if len(parts) > 1 else input_text
            else:
                output_text = (
                    f"{parts[0]}{delimiter if delimiter else '_'}{change_to}"
                    if len(parts) > 1
                    else change_to
                )
        else:
            output_text = input_text

        return (output_text,)
