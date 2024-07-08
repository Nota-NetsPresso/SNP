import os
import json
import torch

from netspresso import NetsPresso
from netspresso.enums import CompressionMethod, GroupPolicy, LayerNorm, Options, Policy

from utils.utils import tofx

COMPRESS_RATIO_PATH="./SNP_compressiont/compress_ratio"
COMPRESS_RATIO={
    "deit_tiny_patch16_224":"DeiT_t.json",
    "deit_small_patch16_224":"DeiT_s.json",
    "deit_base_patch16_224":"DeiT_b.json"
}

def snp(args, model, inputs):
    netspresso = NetsPresso(email=args.NetsPresso_Email, password=args.NetsPresso_Pwd)
    
    model = tofx(model)
    orig_model_path = os.path.join(args.output_dir, "original_model.pt")
    comp_model_path = os.path.join(args.output_dir, "compressed_model.pt")

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
        compression_method=CompressionMethod.PR_L2,
        options=Options(
            policy=Policy.AVERAGE,
            layer_norm=LayerNorm.TSS_NORM,
            group_policy=GroupPolicy.NONE,
            reshape_channel_axis=-1,
        ),
    )

    # 4. Set params for compression(ratio or rank)
    for available_layer in compression_info.available_layers[:5]:
        available_layer.values = [0.2]

    # 5. Compress model
    compressed_model = compressor.compress_model(
        compression=compression_info,
        output_dir=comp_model_path,
    )
    print(compressed_model)
    return compressed_model