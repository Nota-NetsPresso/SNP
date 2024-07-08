import os
import json
import torch

from netspresso import NetsPresso
from netspresso.enums import CompressionMethod, GroupPolicy, LayerNorm, Policy
from netspresso.clients.compressor.v2.schemas import Options


from utils.utils import tofx

COMPRESS_RATIO_PATH="./SNP_compression/compress_ratio"
COMPRESS_RATIO={
    "deit_tiny_patch16_224":"DeiT_t.json",
    "deit_small_patch16_224":"DeiT_s.json",
    "deit_base_patch16_224":"DeiT_b.json"
}

def snp(args, model, inputs):
    netspresso = NetsPresso(email=args.NetsPresso_Email, password=args.NetsPresso_Pwd)
    
    model = tofx(model)
    orig_model_path = os.path.join(args.output_dir, "original_model.pt")

    torch.save(model, orig_model_path)

    # 1. Declare compressor
    compressor = netspresso.compressor_v2()

    # 2. Upload model
    model = compressor.upload_model(
        input_model_path=orig_model_path,
        input_shapes=[{"batch": 1, "channel": 3, "dimension": [224, 224]}],
    )

    # 3. Select compression method
    compression_info = compressor.select_compression_method(
        model_id=model.ai_model_id,
        compression_method=CompressionMethod.PR_SNP,
        options=Options(
            policy=Policy.AVERAGE,
            layer_norm=LayerNorm.TSS_NORM,
            group_policy=GroupPolicy.NONE,
            reshape_channel_axis=-1,
        ),
    )

    # 4. load compress ratio
    with open(os.path.join(COMPRESS_RATIO_PATH, COMPRESS_RATIO[args.model]), "r") as j_file:
        compress_ratio = json.load(j_file)

    for available_layer in compression_info.available_layers:
        available_layer.values = [compress_ratio[available_layer.name]]

    # 5. Compress model
    compressed_model_info = compressor.compress_model(
        compression=compression_info,
        output_dir=args.output_dir,
    )
    return torch.load(compressed_model_info.compressed_model_path)