import os
import random
import re


class TXTBatchLoader:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("content", "filename")
    FUNCTION = "load_batch_texts"
    CATEGORY = "Batch Process"

    def __init__(self):
        self.txt_files = []
        self.current_index = 0
        self.current_directory = ""
        self.search_states = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING",),
                "search_title": ("STRING", {"default": ""}),
                "search_content": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ""}),
                "mode": (
                    ["single_text", "incremental_text", "random"],
                    {"default": "incremental_text"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "filename_option": (
                    [
                        "filename",
                        "prefix",
                        "suffix",
                        "prefix & suffix",
                        "prefix nor suffix",
                    ],
                ),
            },
        }

    def set_directory(
        self,
        directory,
        filename_option="filename",
        search_title="",
        search_content="",
        delimiter="",
    ):
        if (
            directory != self.current_directory
            or filename_option
            or search_title
            or search_content
            or delimiter
        ):
            print(f"Received directory: {directory}")

            if not os.path.isdir(directory):
                raise ValueError(
                    f"The provided path '{directory}' is not a valid directory."
                )

            all_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

            filtered_files = self.filter_files(
                directory,
                all_files,
                filename_option,
                search_title,
                search_content,
                delimiter,
            )

            self.txt_files = sorted(
                [os.path.join(directory, f) for f in filtered_files]
            )

            self.current_directory = directory

            search_key = (
                directory,
                filename_option,
                search_title,
                search_content,
                delimiter,
            )
            if search_key not in self.search_states:
                self.search_states[search_key] = 0

            if not self.txt_files:
                print("No matching TXT files found in the provided directory.")
            else:
                # print(f'Matching TXT files found: {self.txt_files}')
                pass

    def filter_files(
        self, directory, files, filename_option, search_title, search_content, delimiter
    ):
        def get_prefix(filename):
            if delimiter:
                return filename.split(delimiter)[0]
            else:
                return re.split(r"[^a-zA-Z0-9]", filename)[0]

        def get_suffix(filename):
            name_without_ext = os.path.splitext(filename)[0]
            if delimiter:
                return name_without_ext.split(delimiter)[-1]
            else:
                return re.split(r"[^a-zA-Z0-9]", name_without_ext)[-1]

        filtered_files = files

        if search_title:
            if filename_option == "filename":
                filtered_files = [f for f in filtered_files if search_title in f]
            elif filename_option == "prefix":
                search_prefix = get_prefix(search_title)
                filtered_files = [
                    f for f in filtered_files if get_prefix(f) == search_prefix
                ]
            elif filename_option == "suffix":
                search_suffix = get_suffix(search_title)
                filtered_files = [
                    f for f in filtered_files if get_suffix(f) == search_suffix
                ]
            elif filename_option == "prefix & suffix":
                search_prefix = get_prefix(search_title)
                search_suffix = get_suffix(search_title)
                filtered_files = [
                    f
                    for f in filtered_files
                    if get_prefix(f) == search_prefix or get_suffix(f) == search_suffix
                ]
            elif filename_option == "prefix nor suffix":
                search_prefix = get_prefix(search_title)
                search_suffix = get_suffix(search_title)
                filtered_files = [
                    f
                    for f in filtered_files
                    if get_prefix(f) != search_prefix and get_suffix(f) != search_suffix
                ]

        if search_content:
            content_filtered_files = []
            for f in filtered_files:
                file_path = os.path.join(directory, f)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    pattern = (
                        r"\b(?<![a-zA-Z])"
                        + re.escape(search_content)
                        + r"(?![a-zA-Z])\b"
                    )
                    if re.search(pattern, content, re.IGNORECASE):
                        content_filtered_files.append(f)
            filtered_files = content_filtered_files

        return filtered_files

    def get_next_text(self, search_key):
        if not self.txt_files:
            print("No txt files loaded.")
            return None, None

        current_index = self.search_states[search_key]
        if current_index >= len(self.txt_files):
            current_index = 0

        file_path = self.txt_files[current_index]
        self.search_states[search_key] = (current_index + 1) % len(self.txt_files)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        print(f"Current index: {current_index}, Reading file: {file_path}")
        return content, os.path.basename(file_path)

    def load_batch_texts(
        self,
        directory,
        search_title="",
        search_content="",
        delimiter="",
        mode="incremental_text",
        seed=0,
        filename_option="filename",
    ):
        self.set_directory(
            directory, filename_option, search_title, search_content, delimiter
        )

        if not self.txt_files:
            return ("", "")

        search_key = (
            directory,
            filename_option,
            search_title,
            search_content,
            delimiter,
        )

        if mode == "single_text":
            content, filename = self.get_next_text(search_key)
        elif mode == "incremental_text":
            content, filename = self.get_next_text(search_key)
        elif mode == "random":
            random.seed(seed)
            rnd_index = random.randint(0, len(self.txt_files) - 1)
            file_path = self.txt_files[rnd_index]
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            filename = os.path.basename(file_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if content:
            print(f"Final output content: {content}")
            print(f"Final output filename: {filename}")
        else:
            print("Failed to read any content.")

        return (content, os.path.splitext(filename)[0])
