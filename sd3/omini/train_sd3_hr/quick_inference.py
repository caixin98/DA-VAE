#!/usr/bin/env python3
"""
SD3 HR 快速推理脚本
简化的推理接口，用于快速测试训练好的模型
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from omini.train_sd3_hr.inference_sd3_hr import SD3HRInference


def main():
    """快速推理主函数"""
    parser = argparse.ArgumentParser(description="SD3 HR 快速推理")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="检查点路径")
    parser.add_argument("--lora-checkpoint", type=str, help="LoRA检查点路径")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape, high detail, 4k", help="提示词")
    parser.add_argument("--output-dir", type=str, default="./quick_inference_output", help="输出目录")
    parser.add_argument("--steps", type=int, default=28, help="推理步数")
    parser.add_argument("--guidance", type=float, default=3.5, help="引导尺度")
    parser.add_argument("--size", type=int, default=1024, help="图像尺寸")
    parser.add_argument("--seed", type=int, help="随机种子")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SD3 HR 快速推理")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"检查点: {args.checkpoint}")
    print(f"LoRA检查点: {args.lora_checkpoint}")
    print(f"提示词: {args.prompt}")
    print(f"输出目录: {args.output_dir}")
    print(f"推理步数: {args.steps}")
    print(f"引导尺度: {args.guidance}")
    print(f"图像尺寸: {args.size}x{args.size}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    try:
        # 初始化推理器
        print("\n[1/3] 初始化推理器...")
        inference = SD3HRInference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            lora_checkpoint_path=args.lora_checkpoint,
            device="cuda",
            dtype="bfloat16",
        )
        print("✅ 推理器初始化完成")
        
        # 生成图像
        print(f"\n[2/3] 生成图像...")
        images = inference.generate(
            prompts=[args.prompt],
            output_dir=args.output_dir,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.size,
            width=args.size,
            seed=args.seed,
        )
        print("✅ 图像生成完成")
        
        # 显示结果
        print(f"\n[3/3] 结果:")
        print(f"生成了 {len(images)} 张图像")
        print(f"输出目录: {args.output_dir}")
        
        # 列出生成的文件
        if os.path.exists(args.output_dir):
            files = [f for f in os.listdir(args.output_dir) if f.endswith('.jpg')]
            print(f"生成的文件:")
            for file in files:
                print(f"  - {file}")
        
        print("\n🎉 推理完成!")
        
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()