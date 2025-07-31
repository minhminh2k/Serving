import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import time
# import onnxruntime as ort

import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# def allocate_buffers(engine, batch_size):
#     inputs, outputs, bindings = [], [], []
#     stream = cuda.Stream()

#     for binding in engine:
#         dtype = trt.nptype(engine.get_binding_dtype(binding))
#         shape = engine.get_binding_shape(binding)
#         shape = tuple([batch_size if s == -1 else s for s in shape])
#         size = np.prod(shape)
#         host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(host_mem.nbytes)
#         bindings.append(int(device_mem))
#         buffer = {"host": host_mem, "device": device_mem}
#         if engine.binding_is_input(binding):
#             inputs.append(buffer)
#         else:
#             outputs.append(buffer)
#     return inputs, outputs, bindings, stream

# def run_inference(context, bindings, inputs, outputs, stream, input_data):
#     # Copy input data to device
#     for i, inp in enumerate(inputs):
#         np.copyto(inp['host'], input_data[i].ravel())
#         cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

#     # Inference
#     context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

#     # Copy output data back
#     for out in outputs:
#         cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

#     stream.synchronize()
#     return [out['host'] for out in outputs]

# def test_trt_engine(engine_path, input_shapes, num_runs=10, compare_onnx_path=None):
#     print(f"Loading TensorRT engine: {engine_path}")
#     with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())

#     context = engine.create_execution_context()

#     batch_size = input_shapes[0][0]
#     context.set_binding_shape(0, input_shapes[0])  # chỉ set shape cho dynamic input

#     inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size)

#     # Generate dummy input data
#     input_data = [np.random.rand(*shape).astype(np.float32) for shape in input_shapes]

#     # Warm-up
#     for _ in range(3):
#         run_inference(context, bindings, inputs, outputs, stream, input_data)

#     # Benchmark
#     times = []
#     for _ in range(num_runs):
#         start = time.time()
#         out = run_inference(context, bindings, inputs, outputs, stream, input_data)
#         times.append(time.time() - start)

#     print(f"\n🔥 Inference time (avg over {num_runs} runs): {np.mean(times) * 1000:.2f} ms")

#     for idx, out in enumerate(out):
#         print(f"Output[{idx}]: shape={out.shape}, dtype={out.dtype}")

#     # Optional: compare with ONNX
#     if compare_onnx_path:
#         print("\n🔍 Verifying against ONNXRuntime output:")
#         ort_session = ort.InferenceSession(compare_onnx_path, providers=["CUDAExecutionProvider"])
#         onnx_inputs = {ort_session.get_inputs()[i].name: input_data[i] for i in range(len(input_data))}
#         onnx_out = ort_session.run(None, onnx_inputs)

#         for i, (trt_out, onnx_o) in enumerate(zip(out, onnx_out)):
#             diff = np.max(np.abs(trt_out.reshape(onnx_o.shape) - onnx_o))
#             print(f"Compare output[{i}]: max abs diff = {diff:.6f}")

def tensorrt_conversion(args: argparse.Namespace):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 2^30
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"{t.name}: {t.shape}")
        
    input_name = "input_ids"

    profile.set_shape(input_name, 
                    min=(1, 77), # batch, seq_len
                    opt=(1, 77),
                    max=(4, 77))

    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine.")
    
    with open("model.plan", "wb") as f:
        f.write(serialized_engine)
    
def tensorrt_conversion_container(args: argparse.Namespace):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error_idx in range(parser.num_errors):
                print(parser.get_error(error_idx))
            raise RuntimeError("Failed to parse ONNX model")

    # Build config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 4 GiB
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    config.set_flag(trt.BuilderFlag.FP16)

    # Khai báo dynamic shape
    profile = builder.create_optimization_profile()
    
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"{inp.name}: {inp.shape}")
    
        profile.set_shape(inp.name,
                            min=(1, 77),
                            opt=(1, 77),
                            max=(8, 77))

    config.add_optimization_profile(profile)

    print("Building serialized engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")

    with open(args.output_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Serialized engine written to {args.output_path}")

def checking_plan(model_path: str):

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(model_path, "rb") as f:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)

    assert engine is not None, "Engine deserialize failed"
    print("TensorRT engine loaded OK!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Conversion Script")
    parser.add_argument("--onnx_model", type=str, default="model.onnx", required=True, help="Path to the ONNX model file")
    parser.add_argument("--output_path", type=str, default="model.plan", help="Path to save the TensorRT engine")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_model):
        raise FileNotFoundError(f"ONNX model file {args.onnx_model} does not exist.")

    # tensorrt_conversion(args)
    
    # checking_plan("/home/duong.quang.minh/project/packtech-innovate/sunec/triton/model_repository_sdxl/text_encoder/1/model.plan")
    
    tensorrt_conversion_container(args)
    
    

# trtexec --loadEngine=text_encoder.plan --shapes=input_ids:1x77 --fp16
# docker run --rm -it --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:23.12-py3
# trtexec --onnx=model.onnx --saveEngine=tensor.plan --minShapes=input_ids:1x77 --optShapes=input_ids:2x77 --maxShapes=input_ids:4x77 --fp16
# docker run -it --gpus all -p 8888:8888 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:23.12-py3

# trtexec --onnx=sdxl/vae_encoder/model.onnx --saveEngine=model.plan --minShapes=sample:1x3x512x512 --optShapes=sample:1x3x512x512 --maxShapes=sample:2x3x1024x1024 --fp16
# trtexec --onnx=sdxl/vae_decoder/model.onnx --saveEngine=model.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:1x4x128x128 --maxShapes=latent_sample:4x4x128x128 --fp16


# trtexec --onnx=sdxl/unet/model.onnx --saveEngine=model.plan --minShapes=sample:1x9x128x128,timestep:1,encoder_hidden_states:1x77x2048,text_embeds:1x1280,time_ids:1x6 --optShapes=sample:1x9x128x128,timestep:1,encoder_hidden_states:1x77x2048,text_embeds:1x1280,time_ids:1x6  --maxShapes=sample:2x9x128x128,timestep:2,encoder_hidden_states:2x77x2048,text_embeds:2x1280,time_ids:2x6  --shapes=sample:1x9x128x128,timestep:1,encoder_hidden_states:1x77x2048,text_embeds:1x1280,time_ids:1x6