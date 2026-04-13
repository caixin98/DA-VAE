# SD3.5M CLI Test

Two scripts for testing **SD3.5 Medium** (txt2img):

- `sd35m_infer.py`: CLI inference script (supports prompt, resolution, steps, CFG, seed, batch, shift, etc.)
- `run_sd35m.sh`: bash wrapper script (modify the variables at the top to control parameters)

## Quick Start

1) (Optional) Make the script executable:

```bash
chmod +x tools/sd35m_cli_test/run_sd35m.sh
```

2) Run with default parameters:

```bash
tools/sd35m_cli_test/run_sd35m.sh
```

3) Override parameters via environment variables:

```bash
PROMPT="a cute cat, photorealistic" WIDTH=768 HEIGHT=768 STEPS=40 SEED=-1 tools/sd35m_cli_test/run_sd35m.sh
```

Output will be saved to `OUTDIR` (default: `outputs/sd35m/`), along with a `.json` file recording all generation parameters and timing.
