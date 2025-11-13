import os
import torch
import numpy as np
import json
import folder_paths
from server import PromptServer
import re
from comfy_api.latest._util.video_types import VideoContainer, VideoCodec


class VideoBatchSaver:
    INPUT_IS_LIST = True
    FUNCTION = "save"
    CATEGORY = "Batch Process"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    SUPPORTED_FORMATS = VideoContainer.as_input()
    SUPPORTED_CODECS = VideoCodec.as_input()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute - return a unique value each time
        import time

        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "videos": ("VIDEO",),
                "contents": ("STRING", {"forceInput": True}),
                "output_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "VID"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "format": (
                    cls.SUPPORTED_FORMATS,
                    {"default": VideoContainer.MP4.value},
                ),
                "codec": (
                    cls.SUPPORTED_CODECS,
                    {"default": VideoCodec.H264.value},
                ),
                "filename_number_padding": (
                    "INT",
                    {"default": 4, "min": 1, "max": 9, "step": 1},
                ),
                "filename_number": (
                    ["off", "start", "end"],
                    {"default": "end"},
                ),
                "embeded_workflow": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save(
        self,
        videos=None,
        contents=None,
        output_path="",
        filename_prefix="VID",
        filename_delimiter="_",
        filename_suffix="",
        format=VideoContainer.MP4.value,
        codec=VideoCodec.H264.value,  # 默认使用 H264
        filename_number_padding=4,
        filename_number="end",
        embeded_workflow=True,
        node_id=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        # 输入验证和处理
        format_value = self._get_first_or_default(format, VideoContainer.MP4.value)
        codec_value = self._get_first_or_default(codec, VideoCodec.H264.value)
        video_format = self._parse_video_container(format_value)
        video_codec = self._parse_video_codec(codec_value)
        extension = VideoContainer.get_extension(video_format)
        filename_prefix = self._get_first_or_default(filename_prefix, "VID")
        output_path = self._get_first_or_default(output_path, "")
        filename_number_padding = self._get_first_or_default(filename_number_padding, 4)
        filename_delimiter = self._get_first_or_default(filename_delimiter, "_")
        filename_suffix = self._get_first_or_default(filename_suffix, "")
        filename_suffix = filename_suffix.strip("'[]")
        filename_number = self._get_first_or_default(filename_number, "end")
        embeded_workflow = self._get_first_or_default(embeded_workflow, True)

        # 解析数字位置选项
        counter_start = filename_number in ["start"]
        counter_end = filename_number in ["end"]

        if filename_number_padding < 1:
            raise ValueError(
                f"filename_number_padding must be at least 1, got {filename_number_padding}"
            )

        # 处理 videos 输入
        if videos is not None:
            # 确保 videos 是列表
            if not isinstance(videos, list):
                videos = [videos]
            video_count = len(videos)
        else:
            videos = []
            video_count = len(contents) if contents is not None else 1

        # 处理 output_path
        output_path = self._normalize_input(output_path, video_count)

        # 处理 filename_prefix
        filename_prefix = self._normalize_input(filename_prefix, video_count)

        # 处理文本输入
        contents = (
            self._normalize_input(contents, video_count)
            if contents is not None
            else None
        )

        # 准备输出目录
        output_dir = folder_paths.get_output_directory()

        # 批量保存
        saved_files = []
        for idx, (prefix, path) in enumerate(zip(filename_prefix, output_path)):
            try:
                final_output_path = self._get_output_path(output_dir, path)
                os.makedirs(final_output_path, exist_ok=True)

                original_filename = os.path.splitext(os.path.basename(prefix))[0]
                base_filename = self._generate_filename(
                    prefix=original_filename,
                    suffix=filename_suffix,
                    padding=filename_number_padding,
                    counter_start=counter_start,
                    counter_end=counter_end,
                    delimiter=filename_delimiter,
                    final_output_path=final_output_path,
                    extension=extension,
                )

                # 保存视频
                if videos and idx < len(videos):
                    full_path = os.path.join(
                        final_output_path, f"{base_filename}.{extension}"
                    )

                    # 使用 ComfyUI 的视频保存方法
                    self._save_video_comfyui(
                        videos[idx],
                        full_path,
                        video_format=video_format,
                        codec=video_codec,
                        extension=extension,
                        embed_workflow=embeded_workflow,
                        prompt=prompt,
                        extra_pnginfo=extra_pnginfo,
                    )
                    saved_files.append(full_path)
                    print(f"Saved video: {full_path}")
                elif not videos:
                    # 如果没有视频，仍然生成文件路径（可能用于其他用途）
                    full_path = os.path.join(
                        final_output_path, f"{base_filename}.{extension}"
                    )
                    saved_files.append(full_path)
                    print(f"Generated path: {full_path}")

                # 保存文本
                if contents and idx < len(contents):
                    content_path = os.path.join(
                        final_output_path, f"{base_filename}.txt"
                    )
                    with open(content_path, "w", encoding="utf-8") as f:
                        f.write(str(contents[idx]).strip())
                    saved_files.append(content_path)
                    print(f"Saved content: {content_path}")

            except Exception as e:
                print(f"Error saving file {idx+1}: {str(e)}")
                import traceback

                traceback.print_exc()

            self._update_progress(node_id, idx + 1, video_count)

        return ()

    def _get_video_fps(self, video_obj):
        """
        从视频对象中提取 FPS，如果不可用则返回默认值 24
        """
        # 尝试从视频对象中获取 FPS
        if hasattr(video_obj, "fps"):
            fps = video_obj.fps
            if fps is not None and fps > 0:
                return float(fps)
        if hasattr(video_obj, "get_fps"):
            try:
                fps = video_obj.get_fps()
                if fps is not None and fps > 0:
                    return float(fps)
            except:
                pass
        # 尝试从元数据或属性中获取
        if hasattr(video_obj, "__dict__"):
            for attr in ["fps", "frame_rate", "rate"]:
                if hasattr(video_obj, attr):
                    fps = getattr(video_obj, attr)
                    if fps is not None and fps > 0:
                        return float(fps)
        # 默认返回 24
        return 24.0

    def _save_video_comfyui(
        self,
        video_obj,
        path,
        video_format,
        codec,
        extension,
        embed_workflow,
        prompt,
        extra_pnginfo,
    ):
        """
        使用 ComfyUI 的视频保存功能
        """
        try:
            # 准备元数据
            metadata = None
            if embed_workflow:
                metadata = {}
                if prompt is not None:
                    metadata["prompt"] = prompt
                if extra_pnginfo is not None:
                    if isinstance(extra_pnginfo, list):
                        merged_extra = {}
                        for item in extra_pnginfo:
                            if isinstance(item, dict):
                                merged_extra.update(item)
                        extra_pnginfo = merged_extra
                    if isinstance(extra_pnginfo, dict):
                        metadata.update(extra_pnginfo)

            # 检查视频对象是否有 save_to 方法
            if hasattr(video_obj, "save_to"):
                try:
                    # 使用解析后的容器和编码设置（当前 ComfyUI 仅支持 MP4/H264）
                    video_obj.save_to(
                        path,
                        format=video_format,
                        codec=codec,
                        metadata=metadata,
                    )
                    return  # 成功保存，直接返回
                except Exception as save_error:
                    error_msg = str(save_error).lower()
                    # 检查是否是音频提取错误
                    if "audio" in error_msg or "vhs failed" in error_msg:
                        print(
                            f"Warning: Audio extraction failed, attempting to save video without audio: {str(save_error)[:200]}"
                        )
                        # 尝试直接提取视频帧并保存
                        fps = self._get_video_fps(video_obj)
                        self._save_video_as_tensor(video_obj, path, extension, fps)
                        return
                    else:
                        # 其他错误，继续抛出
                        raise
            elif hasattr(video_obj, "save"):
                try:
                    # 有些视频对象可能使用 save 方法而不是 save_to
                    video_obj.save(path)
                    return
                except Exception as save_error:
                    error_msg = str(save_error).lower()
                    if "audio" in error_msg or "vhs failed" in error_msg:
                        print(
                            f"Warning: Audio extraction failed, attempting to save video without audio: {str(save_error)[:200]}"
                        )
                        fps = self._get_video_fps(video_obj)
                        self._save_video_as_tensor(video_obj, path, extension, fps)
                        return
                    else:
                        raise
            else:
                # 如果没有保存方法，尝试转换为张量并保存
                fps = self._get_video_fps(video_obj)
                self._save_video_as_tensor(video_obj, path, extension, fps)

        except Exception as e:
            print(f"Error in ComfyUI video save: {str(e)[:200]}")
            # 尝试使用更通用的保存方法
            try:
                fps = self._get_video_fps(video_obj)
                self._save_video_as_tensor(video_obj, path, extension, fps)
            except Exception as tensor_error:
                print(f"Error in tensor-based save: {str(tensor_error)[:200]}")
                # 最后尝试通用方法
                fps = self._get_video_fps(video_obj)
                self._save_video_generic(video_obj, path, extension, fps)

    def _save_video_as_tensor(self, video_obj, path, extension, fps):
        """
        将视频对象转换为张量并保存
        """
        try:
            tensor = None

            # 尝试多种方式获取视频张量
            # 1. 优先访问 __components.video（ComfyUI 视频容器，避免触发音频提取）
            if hasattr(video_obj, "__components"):
                try:
                    components = video_obj.__components
                    if hasattr(components, "video"):
                        video_comp = components.video
                        # 尝试从 video 组件中获取张量
                        if hasattr(video_comp, "tensor"):
                            tensor = video_comp.tensor
                        elif hasattr(video_comp, "get_tensor"):
                            tensor = video_comp.get_tensor()
                        elif hasattr(video_comp, "frames"):
                            tensor = video_comp.frames
                        elif isinstance(video_comp, torch.Tensor):
                            tensor = video_comp
                    elif hasattr(components, "frames"):
                        tensor = components.frames
                except Exception as comp_error:
                    print(
                        f"Warning: Could not access __components: {str(comp_error)[:100]}"
                    )

            # 2. 直接访问 tensor 属性
            if tensor is None and hasattr(video_obj, "tensor"):
                tensor = video_obj.tensor
            # 3. 调用 get_tensor 方法
            elif tensor is None and hasattr(video_obj, "get_tensor"):
                tensor = video_obj.get_tensor()
            # 4. 访问 frames 属性
            elif tensor is None and hasattr(video_obj, "frames"):
                tensor = video_obj.frames
            # 5. 访问 video 属性（ComfyUI 视频对象可能使用）
            elif tensor is None and hasattr(video_obj, "video"):
                tensor = video_obj.video
            # 6. 尝试从 __dict__ 中查找
            elif tensor is None and hasattr(video_obj, "__dict__"):
                video_dict = video_obj.__dict__
                for key in ["tensor", "frames", "video", "data"]:
                    if key in video_dict:
                        value = video_dict[key]
                        if isinstance(value, torch.Tensor):
                            tensor = value
                            break
            # 7. 如果对象本身就是张量
            elif tensor is None and isinstance(video_obj, torch.Tensor):
                tensor = video_obj

            # 如果找到了张量，保存它
            if tensor is not None and isinstance(tensor, torch.Tensor):
                self._save_tensor_video(tensor, path, extension, fps)
            else:
                raise Exception(
                    f"Could not extract tensor from video object. Type: {type(video_obj)}, Available attributes: {[attr for attr in dir(video_obj) if not attr.startswith('__')][:15]}"
                )

        except Exception as e:
            print(f"Error saving video as tensor: {str(e)}")
            raise e

    def _save_tensor_video(self, tensor, path, extension, fps):
        """
        保存张量视频
        """
        try:
            import imageio

            # 确保张量在 CPU 上
            if tensor.is_cuda:
                tensor = tensor.cpu()

            # 处理不同的张量形状
            if tensor.dim() == 5:  # (batch, frames, channels, height, width)
                tensor = tensor[0]  # 取第一个批次
            elif tensor.dim() == 4:  # (frames, channels, height, width)
                pass
            else:
                raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

            # 转换为 numpy 数组
            frames = []
            for i in range(tensor.shape[0]):
                frame = tensor[i]

                # 转换为 (height, width, channels) 格式
                if frame.shape[0] == 3:  # (channels, height, width)
                    frame = frame.permute(1, 2, 0)

                # 确保是 uint8 类型
                if frame.dtype != torch.uint8:
                    frame = torch.clamp(frame * 255, 0, 255).to(torch.uint8)

                frame_np = frame.numpy()
                frames.append(frame_np)

            # 保存视频
            imageio.mimwrite(path, frames, fps=fps, codec="libx264")

        except ImportError:
            # 如果 imageio 不可用，尝试使用 OpenCV
            self._save_tensor_video_opencv(tensor, path, extension, fps)

    def _save_tensor_video_opencv(self, tensor, path, extension, fps):
        """
        使用 OpenCV 保存张量视频
        """
        try:
            import cv2

            # 确保张量在 CPU 上
            if tensor.is_cuda:
                tensor = tensor.cpu()

            # 处理不同的张量形状
            if tensor.dim() == 5:  # (batch, frames, channels, height, width)
                tensor = tensor[0]  # 取第一个批次
            elif tensor.dim() == 4:  # (frames, channels, height, width)
                pass
            else:
                raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

            # 获取视频尺寸
            if tensor.shape[1] == 3:  # (frames, channels, height, width)
                height, width = tensor.shape[2], tensor.shape[3]
            else:
                height, width = tensor.shape[1], tensor.shape[2]

            # 设置 FourCC 编码器 (使用 H264)
            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H264 编码
            out = cv2.VideoWriter(path, fourcc, fps, (width, height))

            for i in range(tensor.shape[0]):
                frame = tensor[i]

                # 转换为 (height, width, channels) 格式
                if frame.shape[0] == 3:  # (channels, height, width)
                    frame = frame.permute(1, 2, 0)

                # 确保是 uint8 类型
                if frame.dtype != torch.uint8:
                    frame = torch.clamp(frame * 255, 0, 255).to(torch.uint8)

                frame_np = frame.numpy()

                # 转换为 BGR (OpenCV 使用 BGR)
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                out.write(frame_np)

            out.release()

        except ImportError:
            raise ImportError(
                "Neither imageio nor opencv-python are available for video saving"
            )

    def _save_video_generic(self, video_obj, path, extension, fps):
        """
        通用视频保存方法
        """
        try:
            # 首先尝试使用 _save_video_as_tensor，它已经包含了多种提取方法
            try:
                self._save_video_as_tensor(video_obj, path, extension, fps)
                return
            except Exception as tensor_error:
                print(
                    f"Tensor extraction failed, trying other methods: {str(tensor_error)[:100]}"
                )

            # 尝试各种可能的方法来获取视频数据
            if hasattr(video_obj, "_asdict"):
                # 如果是 namedtuple 或类似结构
                try:
                    video_dict = video_obj._asdict()
                    if "frames" in video_dict:
                        video_fps = self._get_video_fps(video_obj)
                        self._save_tensor_video(
                            video_dict["frames"], path, extension, video_fps
                        )
                        return
                    if "video" in video_dict:
                        video_fps = self._get_video_fps(video_obj)
                        self._save_tensor_video(
                            video_dict["video"], path, extension, video_fps
                        )
                        return
                except Exception:
                    pass

            if hasattr(video_obj, "__dict__"):
                # 尝试从对象的属性中获取帧数据
                try:
                    video_dict = video_obj.__dict__
                    for key, value in video_dict.items():
                        if isinstance(value, torch.Tensor) and value.dim() >= 4:
                            video_fps = self._get_video_fps(video_obj)
                            self._save_tensor_video(value, path, extension, video_fps)
                            return
                except Exception:
                    pass

            # 最后尝试直接调用对象的保存方法（不使用音频）
            if hasattr(video_obj, "save"):
                try:
                    video_obj.save(path)
                    return
                except Exception:
                    pass

            raise Exception(
                f"Could not find a way to save the video object. Type: {type(video_obj)}, Attributes: {[attr for attr in dir(video_obj) if not attr.startswith('_')][:10]}"
            )

        except Exception as e:
            print(f"Error in generic video save: {str(e)}")
            raise e

    def _get_first_or_default(self, value, default):
        if isinstance(value, list):
            for v in value:
                if isinstance(v, type(default)):
                    return v
            return default
        return value if value is not None else default

    def _normalize_input(self, input_data, count):
        if input_data is None:
            return [None] * count
        if not isinstance(input_data, list):
            return [input_data] * count
        return input_data + [input_data[-1]] * (count - len(input_data))

    def _get_output_path(self, base_dir, user_path):
        if not user_path or str(user_path).lower() in ["none", "."]:
            return base_dir
        return (
            os.path.join(base_dir, user_path)
            if not os.path.isabs(str(user_path))
            else user_path
        )

    def _generate_filename(
        self,
        prefix,
        suffix,
        padding,
        counter_start,
        counter_end,
        delimiter,
        final_output_path,
        extension,
    ):
        existing_files = [
            f
            for f in os.listdir(final_output_path)
            if f.endswith(f".{extension}") or f.endswith(".txt")
        ]

        if suffix.strip():
            pattern_start = re.compile(
                rf"^(\d+){re.escape(delimiter)}{re.escape(prefix)}{re.escape(delimiter)}{re.escape(suffix)}$"
            )
            pattern_end = re.compile(
                rf"^{re.escape(prefix)}{re.escape(delimiter)}{re.escape(suffix)}{re.escape(delimiter)}(\d+)$"
            )
        else:
            pattern_start = re.compile(
                rf"^(\d+){re.escape(delimiter)}{re.escape(prefix)}$"
            )
            pattern_end = re.compile(
                rf"^{re.escape(prefix)}{re.escape(delimiter)}(\d+)$"
            )

        numbers = []
        for file in existing_files:
            file_base = os.path.splitext(file)[0]
            if counter_start and not counter_end:
                match = pattern_start.match(file_base)
            elif counter_end and not counter_start:
                match = pattern_end.match(file_base)
            elif counter_start and counter_end:
                match_start = pattern_start.match(file_base)
                match_end = pattern_end.match(file_base)
                match = match_start or match_end
                if match_start and match_end:
                    numbers.extend([int(match_start.group(1)), int(match_end.group(1))])
                    continue
            else:
                match = None
            if match:
                numbers.append(int(match.group(1)))

        counter = max(numbers) + 1 if numbers else 1
        parts = []
        if counter_start:
            parts.append(f"{counter:0{padding}d}")
        parts.append(str(prefix))
        if suffix.strip():
            parts.append(str(suffix))
        if counter_end:
            parts.append(f"{counter:0{padding}d}")

        return delimiter.join(parts)

    def _parse_video_container(self, value):
        if isinstance(value, VideoContainer):
            return value
        if value is None:
            return VideoContainer.MP4
        try:
            return VideoContainer(str(value).lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported format: {value}") from exc

    def _parse_video_codec(self, value):
        if isinstance(value, VideoCodec):
            return value
        if value is None:
            return VideoCodec.H264
        try:
            return VideoCodec(str(value).lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported codec: {value}") from exc

    def _update_progress(self, node_id, current, total):
        if node_id:
            PromptServer.instance.send_sync(
                "progress", {"node": node_id, "value": current, "max": total}
            )
