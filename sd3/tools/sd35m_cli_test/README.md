# SD3.5M CLI Test

这里提供两个脚本用于测试 **SD3.5 Medium**（txt2img）：

- `sd35m_infer.py`: 纯命令行推理脚本（支持 prompt/分辨率/steps/CFG/seed/batch/shift 等）
- `run_sd35m.sh`: bash 包装脚本（你直接改顶部变量即可控参）

## 快速开始

1)（可选）赋予可执行权限

```bash
chmod +x tools/sd35m_cli_test/run_sd35m.sh
```

2)直接运行（默认参数）

```bash
tools/sd35m_cli_test/run_sd35m.sh
```

3)通过环境变量覆盖参数（不改脚本）

```bash
PROMPT="a cute cat, photorealistic" WIDTH=768 HEIGHT=768 STEPS=40 SEED=-1 tools/sd35m_cli_test/run_sd35m.sh
```

输出会保存到 `OUTDIR`（默认 `outputs/sd35m/`），同时会写一份 `.json` 记录本次生成的所有参数和耗时。







