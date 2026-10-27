import torch

import comfy.sd
import folder_paths

from .lowbit import lowbit_ops


class MossLowBitUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"), )}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "model/loaders"

    def load_unet(self, unet_name):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model_options = {"dtype": torch.bfloat16, "custom_operations": lowbit_ops(torch.bfloat16)}
        return (comfy.sd.load_diffusion_model(unet_path, model_options=model_options),)


NODE_CLASS_MAPPINGS = {"MossLowBitUNETLoader": MossLowBitUNETLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MossLowBitUNETLoader": "MOSS LowBit UNET Loader"}
