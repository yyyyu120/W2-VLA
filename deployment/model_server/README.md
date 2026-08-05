
# Start a policy server

Set the path to a checkpoint produced by your own training run:

```bash
your_ckpt=playground/Pretrained_models/<policy-run>/checkpoints/steps_<N>_pytorch_model.pt
python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --use_bf16
```

## Debug the connection

```bash
python deployment/model_server/debug_server_policy.py
```
