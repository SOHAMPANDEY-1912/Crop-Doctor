import json, os, re, subprocess
from pathlib import Path

RUNNER = Path(os.environ.get("LLAMA_RUNNER", Path.home() / "project/executorch/cmake-out/examples/models/llama/llama_main"))
MODEL = Path(os.environ.get("LLAMA_MODEL", Path.home() / "project/models/llama3_2_1b_8da4w_ctx1024.pte"))
TOKENIZER = Path(os.environ.get("LLAMA_TOKENIZER", Path.home() / "project/models/tokenizer.model"))
ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"


def _env(runner):
    """Environment for the runner process: its own folder is on LD_LIBRARY_PATH (needed for libllama_runner.so on the Pi)."""
    env = dict(os.environ)
    d = str(Path(runner).parent)
    env["LD_LIBRARY_PATH"] = d + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def chat_prompt(system: str, user: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>{ASSISTANT_TAG}\n\n"
    )


class LlamaRunner:
    def __init__(self, model=MODEL, tokenizer=TOKENIZER, runner=RUNNER,
                 seq_len=700, temperature=0.2, timeout=300):
        self.cmd = [str(runner), f"--model_path={model}", f"--tokenizer_path={tokenizer}",
                    f"--seq_len={seq_len}", f"--temperature={temperature}"]
        self.timeout = timeout

    def generate(self, prompt: str, max_new_tokens: int = 110):
        seq = min(len(prompt) // 3 + max_new_tokens, 1000)
        cmd = [c for c in self.cmd if not c.startswith("--seq_len=")] + [f"--seq_len={seq}"]
        p = subprocess.run(cmd + [f"--prompt={prompt}"], env=_env(self.cmd[0]),
                           capture_output=True, text=True, timeout=self.timeout)
        out, stats = p.stdout, {}
        m = re.search(r"PyTorchObserver (\{.*\})", out)
        if m:
            stats = json.loads(m.group(1))
            out = out[:m.start()]
        i = out.rfind(ASSISTANT_TAG)            # drop the echoed prompt
        if i != -1:
            out = out[i + len(ASSISTANT_TAG):]
        out = out.replace("<|eot_id|>", "").strip()
        return out, stats


if __name__ == "__main__":
    r = LlamaRunner()
    text, stats = r.generate(chat_prompt("Answer briefly.", "What is the capital of France?"))
    print(text)
    print("prefill tok/s:", stats.get("prefill_token_per_sec"),
          "| decode tok/s:", stats.get("decode_token_per_sec"))
