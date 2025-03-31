class SimpleImageTagger:
    CATEGORY = "Batch Process"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "text_template": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"

    def generate_text(self, images, text_template):
        num_images = images.shape[0]
        output_text = ""
        for i in range(num_images):
            output_text += text_template.format(index=i + 1, text="自定义描述") + "\n"
        return (output_text,)
