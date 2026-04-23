# type: ignore
"""Diagnose FP16 compatibility of ONNX model weights.

Checks which weights would be clipped when converting to FP16 and shows statistics.

Usage:
    python modules/pose/batch/detection/diagnose_fp16_weights.py --onnx models/rtmpose-l_256x192.onnx
"""

import argparse
import numpy as np
import onnx
from pathlib import Path

# FP16 range: -65504 to +65504
FP16_MIN = -65504.0
FP16_MAX = 65504.0


def diagnose_fp16_compatibility(onnx_path: str) -> None:
    """Analyze ONNX model weights for FP16 compatibility."""
    
    print(f"\n{'═'*70}")
    print(f"🔍 FP16 Weight Compatibility Analysis")
    print(f"{'═'*70}")
    print(f"  Model: {onnx_path}")
    print(f"  FP16 range: [{FP16_MIN:,.1f}, {FP16_MAX:,.1f}]")
    print(f"{'═'*70}\n")
    
    if not Path(onnx_path).exists():
        print(f"❌ File not found: {onnx_path}")
        return
    
    model = onnx.load(onnx_path)
    
    total_weights = 0
    total_elements = 0
    problematic_weights = []
    
    print("📊 Analyzing weights...\n")
    
    for initializer in model.graph.initializer:
        # Convert to numpy array
        tensor = onnx.numpy_helper.to_array(initializer)
        
        # Skip non-float tensors
        if not np.issubdtype(tensor.dtype, np.floating):
            continue
        
        total_weights += 1
        total_elements += tensor.size
        
        # Find out-of-range values
        out_of_range = (tensor < FP16_MIN) | (tensor > FP16_MAX)
        num_out_of_range = np.sum(out_of_range)
        
        if num_out_of_range > 0:
            min_val = float(np.min(tensor))
            max_val = float(np.max(tensor))
            mean_val = float(np.mean(tensor))
            std_val = float(np.std(tensor))
            
            # How much would be clipped
            clipped = np.clip(tensor, FP16_MIN, FP16_MAX)
            max_error = float(np.max(np.abs(tensor - clipped)))
            mean_error = float(np.mean(np.abs(tensor - clipped)))
            
            problematic_weights.append({
                'name': initializer.name,
                'shape': tensor.shape,
                'size': tensor.size,
                'num_out_of_range': num_out_of_range,
                'percent_out': (num_out_of_range / tensor.size) * 100,
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'max_error': max_error,
                'mean_error': mean_error
            })
    
    # Print summary
    print(f"📈 Summary:")
    print(f"  Total weight tensors: {total_weights}")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Problematic tensors: {len(problematic_weights)}")
    
    if problematic_weights:
        print(f"\n⚠️  Found {len(problematic_weights)} tensors with out-of-range values:\n")
        print(f"{'='*100}")
        
        for i, weight in enumerate(problematic_weights, 1):
            print(f"\n{i}. {weight['name']}")
            print(f"   Shape: {weight['shape']}")
            print(f"   Out-of-range: {weight['num_out_of_range']:,} / {weight['size']:,} ({weight['percent_out']:.2f}%)")
            print(f"   Value range: [{weight['min']:,.2f}, {weight['max']:,.2f}]")
            print(f"   Mean: {weight['mean']:,.4f}, Std: {weight['std']:,.4f}")
            print(f"   Clipping error: max={weight['max_error']:,.2f}, mean={weight['mean_error']:,.6f}")
        
        print(f"\n{'='*100}")
        
        # Calculate total impact
        total_out_of_range = sum(w['num_out_of_range'] for w in problematic_weights)
        percent_total = (total_out_of_range / total_elements) * 100
        
        print(f"\n💡 Total Impact:")
        print(f"   {total_out_of_range:,} / {total_elements:,} elements affected ({percent_total:.4f}%)")
        
        # Find the worst offender
        worst = max(problematic_weights, key=lambda x: x['max_error'])
        print(f"\n🔴 Worst tensor: {worst['name']}")
        print(f"   Max clipping error: {worst['max_error']:,.2f}")
        print(f"   This weight has values up to {worst['max']:.2f} (FP16 max is {FP16_MAX:.1f})")
        
        # Recommendations
        print(f"\n💭 Recommendations:")
        if percent_total < 0.01:
            print(f"   ✓ Only {percent_total:.4f}% of weights affected - FP16 might be acceptable")
            print(f"   ✓ Test thoroughly with your use case")
        elif percent_total < 0.1:
            print(f"   ⚠️  {percent_total:.4f}% of weights affected - use with caution")
            print(f"   ⚠️  Monitor accuracy carefully")
        else:
            print(f"   ❌ {percent_total:.4f}% of weights affected - FP16 NOT recommended")
            print(f"   ❌ Consider retraining with FP16 or mixed precision")
    else:
        print(f"\n✅ All weights are within FP16 range!")
        print(f"✅ Model should convert to FP16 safely")
    
    print(f"\n{'═'*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose FP16 weight compatibility')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model')
    
    args = parser.parse_args()
    diagnose_fp16_compatibility(args.onnx)
