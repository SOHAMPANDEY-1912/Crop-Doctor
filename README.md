# Crop Doctor (offline)

Offline leaf-disease and pest advisory for farmers (rice and tea, Barak Valley, Assam).
Runs on a Raspberry Pi 5 with a Camera Module 3. No internet needed.

## How it works
- `vision/`: ONNX leaf classifier (20 classes)
- `cards/`: vetted advice cards (the app always shows these)
- `advisory/`: routing, prompt and safety checks for the Llama explanation
- `cropapp/`: Flask web app (port 5000)
- `eval/`: Llama pass-rate test (`llama_passrate.py`)

## Setup (Raspberry Pi OS)
    sudo apt install -y python3-picamera2
    python3 -m venv --system-site-packages ~/venv
    source ~/venv/bin/activate
    pip install -r requirements.txt

Clone this repo to `~/project`; the scripts use that path.

## Run
    source ~/venv/bin/activate
    export LLAMA_RUNNER=$HOME/executorch-deployment/llama_main
    export LLAMA_MODEL=$HOME/executorch-deployment/llama3_2_1b_8da4w_ctx1024.pte
    export LLAMA_TOKENIZER=$HOME/executorch-deployment/tokenizer.model
    cd ~/project/cropapp && python app.py

Open http://127.0.0.1:5000 on the Pi and press Capture.
The Llama files are not in this repo: they come from the ExecuTorch deployment (see `~/executorch-deployment`).

## Safety
Llama text is shown only if it passes checks (no doses or chemicals, grounded in the card,
no repeated sentences, no unsupported cause or danger words). Otherwise only the vetted card is shown.
Advice only: confirm any treatment with your KVK or agriculture officer.

## Evaluation
`eval/llama_passrate.py` runs 12 English prompts; results are in `data/llama_passrate_final.json`.
