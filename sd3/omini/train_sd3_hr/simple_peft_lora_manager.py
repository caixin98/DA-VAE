"""
超简单PEFT LoRA管理器 - 只使用PEFT的基本功能
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import torch
from peft import LoraConfig, get_peft_model_state_dict
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline


class SimplePEFTLoRAManager:
    """
    超简单PEFT LoRA管理器 - 只使用PEFT的基本功能，不调用不存在的方法
    """
    
    def __init__(self, pipeline: StableDiffusion3Pipeline, transformer):
        self.pipeline = pipeline
        self.transformer = transformer
        self.adapters = {}  # 管理所有PEFT适配器
        
    def add_lora(self, name: str, config: Dict[str, Any], weight: float = 1.0) -> bool:
        """
        添加PEFT LoRA适配器 - 只使用基本的add_adapter
        """
        print(f"[LoRA] 🔧 添加PEFT适配器 '{name}'")
        
        try:
            # 创建LoRA配置
            lora_config = LoraConfig(**config)
            self.transformer.add_adapter(lora_config, adapter_name=name)
            
            # 激活适配器 (使用PEFT的set_adapter方法)
            self.transformer.set_adapter(name)
            
            # 记录适配器信息
            self.adapters[name] = {
                "type": "peft",
                "config": config,
                "weight": weight,
                "active": True
            }
            
            print(f"[LoRA] ✅ 成功添加PEFT适配器 '{name}'")
            return True
            
        except Exception as e:
            print(f"[LoRA] ❌ 添加PEFT适配器失败: {e}")
            return False
    
    def load_lora_weights(self, name: str, path: Union[str, List], weight: float = 1.0) -> bool:
        """
        加载LoRA权重 - 使用PEFT的load_adapter方法
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
            
            # 检查是否已经有LoRA适配器，如果有则删除
            existing_adapters = getattr(self.transformer, 'peft_config', {})
            if existing_adapters:
                print(f"[LoRA] 🗑️ 发现现有适配器，先删除: {list(existing_adapters.keys())}")
                try:
                    self.transformer.delete_adapters(list(existing_adapters.keys()))
                    print(f"[LoRA] ✅ 成功删除现有适配器")
                except Exception as e:
                    print(f"[LoRA] ⚠️ 删除现有适配器失败: {e}")
            
            # 检查路径是文件还是目录
            if os.path.isfile(path):
                # 如果是文件，直接加载
                self.transformer.load_lora_adapter(
                    pretrained_model_name_or_path_or_dict=path,
                    prefix=None,  # 使用None作为prefix
                    hotswap=False,
                    adapter_name=name
                )
            else:
                # 如果是目录，查找LoRA权重文件
                lora_files = []
                for file in os.listdir(path):
                    if file.endswith(('.safetensors', '.bin')):
                        lora_files.append(os.path.join(path, file))
                
                if lora_files:
                    # 使用第一个找到的LoRA文件
                    lora_file = lora_files[0]
                    self.transformer.load_lora_adapter(
                        pretrained_model_name_or_path_or_dict=lora_file,
                        prefix=None,  # 使用None作为prefix
                        hotswap=False,
                        adapter_name=name
                    )
                else:
                    # 如果没有找到LoRA文件，尝试加载整个目录
                    self.transformer.load_lora_adapter(
                        pretrained_model_name_or_path_or_dict=path,
                        prefix=None,  # 使用None作为prefix
                        hotswap=False,
                        adapter_name=name
                    )
            
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
    
    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """
        获取所有PEFT LoRA参数
        """
        params = []
        seen = set()
        
        # 检测LoRA参数 - 直接查找包含lora的模块
        for name, module in self.transformer.named_modules():
            if 'lora' in name.lower() and hasattr(module, 'parameters'):
                for pname, p in module.named_parameters():
                    if id(p) not in seen and p.requires_grad:
                        params.append(p)
                        seen.add(id(p))
        
        return params
    
    def save_all(self, save_dir: str, step: int) -> bool:
        """
        保存所有PEFT LoRA适配器 - 使用PEFT的save_pretrained方法
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 检查是否有PEFT适配器
            peft_config = getattr(self.transformer, 'peft_config', {})
            if not peft_config:
                print(f"[LoRA] 没有PEFT适配器需要保存")
                return True
            
            # 使用Diffusers的save_lora_adapter方法保存适配器
            # 检查save_dir是否已经包含了step信息，避免重复
            if f"lora_step_{step}" in save_dir:
                save_path = save_dir
            else:
                save_path = os.path.join(save_dir, f"lora_step_{step}")
            # 获取当前激活的适配器名称
            try:
                # 尝试调用active_adapters方法
                active_adapters = self.transformer.active_adapters()
            except:
                # 如果方法不存在，尝试获取属性
                active_adapters = getattr(self.transformer, 'active_adapters', [])
                if not active_adapters:
                    active_adapters = getattr(self.transformer, 'active_adapter', None)
                    if active_adapters:
                        active_adapters = [active_adapters]
            
            if active_adapters:
                # 使用第一个激活的适配器名称
                adapter_name = active_adapters[0] if isinstance(active_adapters, list) else active_adapters
                self.transformer.save_lora_adapter(save_path, adapter_name=adapter_name)
            else:
                # 如果没有激活的适配器，尝试使用默认名称
                self.transformer.save_lora_adapter(save_path, adapter_name="default")
            
            return True
            
        except Exception as e:
            print(f"[LoRA] ❌ 保存PEFT适配器失败: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str, step: int) -> bool:
        """
        加载PEFT LoRA检查点 - 处理.safetensors文件
        """
        print(f"[LoRA] 🔄 加载PEFT检查点: {checkpoint_path}")
        
        try:
            # 检查检查点路径是否存在
            if not os.path.exists(checkpoint_path):
                print(f"[LoRA] ❌ 检查点路径不存在: {checkpoint_path}")
                return False
            
            # 处理目录路径，查找.safetensors文件
            if os.path.isdir(checkpoint_path):
                # 查找.safetensors文件
                safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]
                if not safetensors_files:
                    print(f"[LoRA] ❌ 目录中没有找到.safetensors文件: {checkpoint_path}")
                    return False
                
                weight_file = os.path.join(checkpoint_path, safetensors_files[0])
                print(f"[LoRA] 📁 找到权重文件: {weight_file}")
            else:
                weight_file = checkpoint_path
            
            # 使用safetensors加载权重
            import safetensors.torch as st
            state_dict = st.load_file(weight_file)
            
            # 应用权重到transformer
            from peft import set_peft_model_state_dict
            incompatible_keys = set_peft_model_state_dict(self.transformer, state_dict)
            
            if incompatible_keys and hasattr(incompatible_keys, 'unexpected_keys'):
                if incompatible_keys.unexpected_keys:
                    print(f"[LoRA] ⚠️ 警告: 发现意外键: {incompatible_keys.unexpected_keys}")
            
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
        
        # 获取当前激活的适配器
        active_adapters = getattr(self.transformer, 'active_adapters', [])
        if not active_adapters:
            active_adapters = getattr(self.transformer, 'active_adapter', None)
            if active_adapters:
                active_adapters = [active_adapters]
        
        return {
            "PEFT适配器数量": peft_count,
            "PEFT配置": list(peft_config.keys()),
            "当前激活适配器": active_adapters,
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
    
    def merge_into_base(self) -> bool:
        """
        尝试将已加载的LoRA权重合并进基础模型并卸载适配器。
        采用多重回退策略：
          1) 如果pipeline支持 fuse_lora()，优先调用并卸载适配器
          2) 如果transformer支持 fuse_lora()，执行并尝试删除适配器以固化
          3) 如果transformer支持 merge_and_unload()，调用并回写到pipeline.transformer
          4) 否则返回False（不支持合并）
        """
        try:
            # 获取当前适配器列表（如果可用）
            adapters: list = []
            try:
                if hasattr(self.pipeline, "get_list_adapters"):
                    adapters = list(self.pipeline.get_list_adapters() or [])
            except Exception:
                adapters = []
            
            # 路径1：Diffusers pipeline 的 fuse_lora
            if hasattr(self.pipeline, "fuse_lora"):
                try:
                    print("[LoRA] 尝试通过 pipeline.fuse_lora() 合并")
                    self.pipeline.fuse_lora()
                    # 卸载所有适配器，避免后续仍然叠加
                    for name in adapters:
                        try:
                            self.pipeline.unload_lora_weights(name)
                        except Exception:
                            pass
                    print("[LoRA] 🔗 已通过 pipeline.fuse_lora() 合并LoRA并卸载适配器")
                    # 简单校验：查看transformer是否仍然挂载peft_config
                    try:
                        peft_cfg = getattr(self.transformer, "peft_config", {})
                        print(f"[LoRA] 校验：peft_config 剩余条目 = {len(peft_cfg) if isinstance(peft_cfg, dict) else '未知'}")
                    except Exception:
                        pass
                    return True
                except Exception:
                    pass
            
            # 路径2：模型本体的 fuse_lora（如可用），然后删除适配器固化
            if hasattr(self.transformer, "fuse_lora"):
                try:
                    print("[LoRA] 尝试通过 transformer.fuse_lora() 合并")
                    self.transformer.fuse_lora()
                    # 删除适配器使之固化（如可用）
                    try:
                        existing_adapters = getattr(self.transformer, 'peft_config', {})
                        if existing_adapters:
                            self.transformer.delete_adapters(list(existing_adapters.keys()))
                    except Exception:
                        pass
                    try:
                        peft_cfg = getattr(self.transformer, "peft_config", {})
                        print(f"[LoRA] 🔗 已通过 transformer.fuse_lora() 合并；剩余适配器: {list(getattr(peft_cfg, 'keys', lambda: [])()) if isinstance(peft_cfg, dict) else '未知'}")
                    except Exception:
                        print("[LoRA] 🔗 已通过 transformer.fuse_lora() 合并")
                    return True
                except Exception:
                    pass
            
            # 路径3：PEFT的 merge_and_unload（如可用）
            if hasattr(self.transformer, "merge_and_unload"):
                try:
                    print("[LoRA] 尝试通过 transformer.merge_and_unload() 合并")
                    merged = self.transformer.merge_and_unload()
                    if merged is not None:
                        self.transformer = merged
                        try:
                            # 同步回 pipeline
                            if hasattr(self.pipeline, "transformer"):
                                self.pipeline.transformer = self.transformer
                        except Exception:
                            pass
                    # 清理已记录的适配器
                    self.adapters = {}
                    print("[LoRA] 🔗 已通过 transformer.merge_and_unload() 合并LoRA")
                    return True
                except Exception:
                    pass
        except Exception:
            pass
        print("[LoRA] ⚠️ 当前环境不支持自动合并LoRA（保持适配器加载状态）")
        return False