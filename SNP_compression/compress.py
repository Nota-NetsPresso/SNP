import os
import torch

from netpresso import NetsPresso
from netspresso.enums import CompressionMethod, GroupPolicy, LayerNorm, Options, Policy

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
        model_id=model.model_id,
        compression_method=CompressionMethod.PR_L2,
        options=Options(
            policy=Policy.BACKWARD,
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
    print("hi")