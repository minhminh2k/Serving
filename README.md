# Serving

```

# Exporting and converting the models
docker run -it --gpus all -p 8888:8888 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:yy.mm-py3

# Accelerating VAE with TensorRT (Optional)
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16

# Run with Docker
docker run --gpus=all --shm-size=256m --rm -p8111:8000 -p8112:8001 -p8113:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models
```