# type: ignore
"""Convert FP32 ONNX model to FP16 with selective precision control.

Usage:
    # Inspect model ops
    python modules/pose/batch/detection/convert_onnx_to_fp16.py --input models/rtmpose-l_256x192_fp32.onnx --inspect

    # Convert only Conv to FP16, keep everything else FP32
    python modules/pose/batch/detection/convert_onnx_to_fp16.py --input models/rtmpose-l_256x192_fp32.onnx --output models/rtmpose-l_256x192_test.onnx --fp16-ops Conv

    # Add more ops to FP16
    python modules/pose/batch/detection/convert_onnx_to_fp16.py --input models/rtmpose-l_256x192_fp32.onnx --output models/rtmpose-l_256x192_test.onnx --fp16-ops Conv Mul Sigmoid
"""

import argparse
import sys
from pathlib import Path

import onnx
from onnxconverter_common import float16


def inspect_model(input_path: str) -> None:
    """Inspect ONNX model and print operation counts."""
    print(f"\n{'═'*60}")
    print(f"🔍 Model: {input_path}")
    print(f"{'═'*60}\n")
    
    model = onnx.load(input_path)
    
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    
    print("📊 Operations (sorted by count):")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op:25s} : {count}")
    
    print(f"\n📝 Total nodes: {len(model.graph.node)}")
    print("\n💡 Copy op names for --fp16-ops argument")


def convert_selective_fp16(input_path: str, output_path: str, fp16_ops: list) -> bool:
    """Convert only specified op types to FP16, keep everything else FP32.
    
    Uses onnxconverter-common with op_block_list containing all ops EXCEPT the ones we want in FP16.
    """
    print(f"\n{'═'*60}")
    print(f"🔄 Selective FP16 Conversion")
    print(f"{'═'*60}")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  FP16 ops: {fp16_ops}")
    print(f"{'═'*60}\n")

    if not Path(input_path).exists():
        print(f"❌ Input not found: {input_path}")
        return False

    model = onnx.load(input_path)
    
    # Find all unique op types in the model
    all_ops = set(node.op_type for node in model.graph.node)
    print(f"📊 Found {len(all_ops)} unique op types in model")
    
    # Block list = all ops EXCEPT the ones we want in FP16
    op_block_list = [op for op in all_ops if op not in fp16_ops]
    
    print(f"🔧 Converting to FP16: {fp16_ops}")
    print(f"🔒 Keeping in FP32: {len(op_block_list)} op types")

    # Convert using onnxconverter-common
    try:
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            op_block_list=op_block_list
        )
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, output_path)
    
    in_size = Path(input_path).stat().st_size / 1024 / 1024
    out_size = Path(output_path).stat().st_size / 1024 / 1024
    
    print(f"\n✅ Done: {out_size:.1f} MB ({(1-out_size/in_size)*100:.0f}% smaller)")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selective FP16 ONNX conversion')
    parser.add_argument('--input', required=True, help='Input FP32 ONNX file')
    parser.add_argument('--output', help='Output ONNX file')
    parser.add_argument('--inspect', action='store_true', help='Inspect model ops')
    parser.add_argument('--fp16-ops', nargs='+', default=[], help='Ops to convert to FP16 (default: all)')
    parser.add_argument('--all', action='store_true', help='Convert all ops to FP16 (default behavior)')

    args = parser.parse_args()

    if args.inspect:
        inspect_model(args.input)
        sys.exit(0)

    if not args.output:
        print("❌ --output required")
        sys.exit(1)

    # Default to converting all ops if nothing specified
    if not args.fp16_ops and not args.all:
        args.all = True

    if args.all:
        model = onnx.load(args.input)
        all_ops = list(set(node.op_type for node in model.graph.node))
        print(f"🔄 Converting ALL operations to FP16 ({len(all_ops)} op types)")
        success = convert_selective_fp16(args.input, args.output, all_ops)
    else:
        success = convert_selective_fp16(args.input, args.output, args.fp16_ops)

    sys.exit(0 if success else 1)
