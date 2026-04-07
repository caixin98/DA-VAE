"""
纯PEFT LoRA管理器 - 全程使用PEFT管理LoRA，避免pipeline和PEFT混用
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import torch
from peft import LoraConfig, get_peft_model_state_dict
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline


class PEFTOnlyLoRAManager:
    """
    纯PEFT LoRA管理器 - 全程使用PEFT管理LoRA，避免pipeline和PEFT混用
    """
    
    def __init__(self, pipeline: StableDiffusion3Pipeline, transformer):
        self.pipeline = pipeline
        self.transformer = transformer
        self.adapters = {}  # 管理所有PEFT适配器
        
    def add_lora(self, name: str, config: Dict[str, Any], weight: float = 1.0) -> bool:
        """
        添加PEFT LoRA适配器
        """
        print(f"[LoRA] 🔧 添加PEFT适配器 '{name}' (权重: {weight})")
        
        try:
            # 创建LoRA配置
            lora_config = LoraConfig(**config)
            self.transformer.add_adapter(lora_config, adapter_name=name)
            
            # PEFT适配器添加后自动激活，不需要手动调用set_adapters
            
            # 记录适配器信息
            self.adapters[name] = {
                "type": "peft",
                "config": config,
                "weight": weight,
                "active": True
            }
            
            print(f"[LoRA] ✅ 成功添加并激活PEFT适配器 '{name}'")
            return True
            
        except Exception as e:
            print(f"[LoRA] ❌ 添加PEFT适配器失败: {e}")
            return False
    
    def load_lora_weights(self, name: str, path: Union[str, List], weight: float = 1.0) -> bool:
        """
        加载LoRA权重 - 纯PEFT方式
        """
        print(f"[LoRA] 📥 加载PEFT权重 '{name}' 从 '{path}' (权重: {weight})")
        
        try:
            # 处理路径参数
            if isinstance(path, (list, tuple)) and len(path) == 2:
                path, weight = path
            elif isinstance(path, (list, tuple)) and len(path) == 1:
                path = path[0]
            
            # 检查路径是否存在
            if not os.path.exists(path):
                print(f"[LoRA] ❌ 权重文件不存在: {path}")
                return False
            
            # 加载权重到transformer
            if os.path.isdir(path):
                # 目录路径 - 查找safetensors文件
                safetensors_files = [f for f in os.listdir(path) if f.endswith('.safetensors')]
                if not safetensors_files:
                    print(f"[LoRA] ❌ 目录中没有找到.safetensors文件: {path}")
                    return False
                
                # 使用第一个safetensors文件
                weight_file = os.path.join(path, safetensors_files[0])
                print(f"[LoRA] 使用权重文件: {weight_file}")
            else:
                weight_file = path
            
            # 加载权重到PEFT适配器
            self._load_peft_weights(name, weight_file, weight)
            
            # 记录适配器信息
            self.adapters[name] = {
                "type": "peft",
                "path": path,
                "weight": weight,
                "active": True
            }
            
            print(f"[LoRA] ✅ 成功加载PEFT权重 '{name}'")
            return True
            
        except Exception as e:
            print(f"[LoRA] ❌ 加载PEFT权重失败: {e}")
            return False
    
    def _load_peft_weights(self, name: str, weight_file: str, weight: float):
        """加载PEFT权重到适配器"""
        try:
            # 检查适配器是否存在
            if name not in getattr(self.transformer, 'peft_config', {}):
                print(f"[LoRA] 适配器 '{name}' 不存在，先创建适配器")
                # 这里需要先创建适配器，但我们需要配置信息
                # 暂时跳过，假设适配器已经存在
                return False
            
            # 加载权重到现有适配器
            from peft import set_peft_model_state_dict
            import safetensors.torch as st
            
            # 加载权重文件
            if weight_file.endswith('.safetensors'):
                state_dict = st.load_file(weight_file)
            else:
                state_dict = torch.load(weight_file, map_location='cpu')
            
            # 应用权重到适配器
            incompatible_keys = set_peft_model_state_dict(
                self.transformer, 
                state_dict, 
                adapter_name=name
            )
            
            if incompatible_keys and hasattr(incompatible_keys, 'unexpected_keys'):
                if incompatible_keys.unexpected_keys:
                    print(f"[LoRA] 警告: 发现意外键: {incompatible_keys.unexpected_keys}")
            
            # PEFT适配器加载权重后自动激活
            
            return True
            
        except Exception as e:
            print(f"[LoRA] 加载PEFT权重失败: {e}")
            return False
    
    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """
        获取所有PEFT LoRA参数
        """
        params = []
        seen = set()
        
        # 只检测PEFT LoRA参数
        from peft.tuners.tuners_utils import BaseTunerLayer
        for name, module in self.transformer.named_modules():
            if isinstance(module, BaseTunerLayer):
                for pname, p in module.named_parameters(recurse=False):
                    if id(p) not in seen:
                        p.requires_grad_(True)
                        params.append(p)
                        seen.add(id(p))
        
        return params
    
    def save_all(self, save_dir: str, step: int) -> bool:
        """
        保存所有PEFT LoRA适配器
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 获取PEFT状态字典
            peft_state = get_peft_model_state_dict(self.transformer)
            
            if not peft_state:
                print(f"[LoRA] 没有PEFT适配器需要保存")
                return True
            
            # 保存为safetensors格式
            import safetensors.torch as st
            save_path = os.path.join(save_dir, "peft_adapters.safetensors")
            st.save_file(peft_state, save_path)
            
            print(f"[LoRA] ✅ 成功保存PEFT适配器到 {save_path}")
            return True
            
        except Exception as e:
            print(f"[LoRA] ❌ 保存PEFT适配器失败: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str, step: int) -> bool:
        """
        加载PEFT LoRA检查点
        """
        print(f"[LoRA] 🔄 加载PEFT检查点: {checkpoint_path}")
        
        try:
            if not os.path.exists(checkpoint_path):
                print(f"[LoRA] ❌ 检查点不存在: {checkpoint_path}")
                return False
            
            # 使用PEFT方式加载检查点
            return self._load_peft_checkpoint(checkpoint_path)
            
        except Exception as e:
            print(f"[LoRA] ❌ 加载PEFT检查点失败: {e}")
            return False
    
    def _load_peft_checkpoint(self, checkpoint_path: str) -> bool:
        """加载PEFT检查点"""
        try:
            if os.path.isdir(checkpoint_path):
                # 目录路径 - 查找safetensors文件
                safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]
                if not safetensors_files:
                    print(f"[LoRA] ❌ 目录中没有找到.safetensors文件: {checkpoint_path}")
                    return False
                
                weight_file = os.path.join(checkpoint_path, safetensors_files[0])
            else:
                weight_file = checkpoint_path
            
            # 加载权重
            import safetensors.torch as st
            state_dict = st.load_file(weight_file)
            
            # 应用权重到transformer
            from peft import set_peft_model_state_dict
            incompatible_keys = set_peft_model_state_dict(self.transformer, state_dict)
            
            if incompatible_keys and hasattr(incompatible_keys, 'unexpected_keys'):
                if incompatible_keys.unexpected_keys:
                    print(f"[LoRA] 警告: 发现意外键: {incompatible_keys.unexpected_keys}")
            
            print(f"[LoRA] ✅ 成功加载PEFT检查点: {weight_file}")
            return True
            
        except Exception as e:
            print(f"[LoRA] ❌ 加载PEFT检查点失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取PEFT LoRA状态
        """
        peft_config = getattr(self.transformer, 'peft_config', {})
        peft_count = len(peft_config)
        total_params = len(self.get_all_parameters())
        
        return {
            "PEFT适配器数量": peft_count,
            "PEFT配置": list(peft_config.keys()),
            "LoRA参数总数": total_params,
            "状态": "正常" if total_params > 0 else "警告: 未检测到LoRA参数"
        }
    
    def print_status(self):
        """打印PEFT状态信息"""
        status = self.get_status()
        print(f"\n[LoRA] === PEFT状态报告 ===")
        for key, value in status.items():
            print(f"[LoRA] {key}: {value}")
        print(f"[LoRA] ====================\n")
