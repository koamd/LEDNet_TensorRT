import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union, NewType
import onnx

# Verifies packages are installed.
import tensorrt as trt  # pip install tensorrt
import polygraphy  # pip install polygraphy
import torch

import ctypes
import importlib
import platform
import json


def load_trtllm_plugins(required: bool = True):
    TRT_LLM_PLUGIN_NAMESPACE = "tensorrt_llm"
    on_windows = platform.system() == "Windows"
    winmode = None
    spec = importlib.util.find_spec("tensorrt_llm")
    if not spec:
        err_msg: str = "TensorRT-LLM is not installed in this environment!"
        if required:
            raise RuntimeError(err_msg)
        else:
            # print(err_msg, file=sys.stderr)
            return
    plugin_path = Path(spec.origin).parent.joinpath("libs").joinpath("libnvinfer_plugin_tensorrt_llm.so")
    if on_windows:
        plugin_path = plugin_path.with_suffix(".dll")
        winmode = 0
    lib_dir = Path(sys.executable).parent.parent.joinpath(sys.platlibdir)
    if on_windows:
        libmpi_libs: List[Path] = list(lib_dir.glob("libmpi.dll"))
    else:
        libmpi_libs: List[Path] = list(lib_dir.glob("libmpi.so"))
    if len(libmpi_libs) != 1:
        raise RuntimeError(f"Find 0 or more than 1 'libmpi.so' candidates at path: {lib_dir} - cannot load TRTLLM")
    ctypes.CDLL(str(libmpi_libs[0]), mode=ctypes.RTLD_GLOBAL, winmode=winmode)
    handle = ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL, winmode=winmode)
    try:
        handle.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtLlmPlugins.restype = ctypes.c_bool
    except AttributeError as err:
        raise ImportError("TensorRT-LLM Plugin is unavailable") from err
    handle.initTrtLlmPlugins(None, TRT_LLM_PLUGIN_NAMESPACE.encode("utf-8"))





def _is_tensorrt_llm_node(node: onnx.NodeProto) -> bool:
    for attr in node.attribute:
        if attr.name == "plugin_namespace":
            value = onnx.helper.get_attribute_value(attr)
            if isinstance(value, bytes):
                value = value.decode()
            if value == "tensorrt_llm":
                return True
    return False


def onnx2trt(
        model_path: Union[str, Path],
        int8: Optional[bool] = None,
        fp16: bool = False,
        bf16: bool = False,
        fp8: bool = False,
        tf32: bool = False,
        optimization_level: Literal[0, 1, 2, 3, 4, 5] = 3,
        weight_streaming: bool = False,
        shape_profiles: Optional[List[Dict[str, Dict[Literal["min", "opt", "max"], Sequence[int]]]]] = None,
        workspace_size: Optional[int] = None,
        strongly_typed: bool = False,
        sparse_weights: bool = False,
        max_aux_streams: int = None,
        timing_cache: str = None,
        algorithm_selector: trt.IAlgorithmSelector = None,
        tactic_sources=None,
        strict_type: bool = False,
        refit: bool = False,
        precision_constraint: Literal["obey", "prefer"] = None,
        # visualize: bool = False,
        verbose: bool = False,
        cache_name_hint: Optional[str] = None,
        trtllm_plugins_required: Optional[bool] = None,
        force: bool = False
) -> Tuple[trt.ICudaEngine, Path]:

    if workspace_size is None:
        mem_info: Tuple[int, ...] = torch.cuda.mem_get_info(torch.cuda.current_device())
        workspace_size: int = mem_info[0]
        workspace_size -= int(workspace_size * 0.05)

    model_path = Path(model_path)

    if cache_name_hint is None:
        cache_name_hint = ""

    onnx_model = onnx.load_model(model_path, load_external_data=False)

    _input_names = [x.name for x in onnx_model.graph.input]
    if int8 is None:
        int8 = any(["quantize" in x.op_type.lower() for x in onnx_model.graph.node])

    if trtllm_plugins_required is None:
        trtllm_plugins_required = any([_is_tensorrt_llm_node(x) for x in onnx_model.graph.node])
        if trtllm_plugins_required:
            int8 = False
            fp16 = False
            bf16 = False
            strongly_typed = True

    if trtllm_plugins_required:
        load_trtllm_plugins(required=trtllm_plugins_required)

    if strongly_typed:
        int8 = False
        fp16 = False
        bf16 = False

    del onnx_model

    _unique_name: str = f"{model_path.stem}_trt{trt.__version__}"
    if cache_name_hint != "":
        _unique_name += f"_{cache_name_hint}"
    if int8:
        _unique_name += "_int8"
    if fp16:
        _unique_name += "_fp16"
    if bf16:
        _unique_name += "_bf16"
    if tf32:
        _unique_name += "_tf32"
    if trtllm_plugins_required:
        _unique_name += "_trtllm"
    _unique_name += f"_opt{optimization_level}"
    _unique_name += ".engine"
    engine_path = model_path.parent.joinpath(_unique_name)
    if engine_path.exists() and not force:
        with engine_path.open("rb") as fp:
            engine_bytes = fp.read()
        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)
        engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(engine_bytes)
        # if visualize:
        #     _visualize(engine, engine_path)
        return engine, engine_path

    timing_cache = Path(timing_cache) if timing_cache else None

    _log_level = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger = trt.Logger(_log_level)
    runtime = trt.Runtime(logger)
    builder = trt.Builder(logger)
    builder.max_threads = os.cpu_count()

    _create_network_args = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    # if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys():  # deprecated in recent TRT
    #     # all ONNX models are interpreted as having explicit batch (https://github.com/NVIDIA/TensorRT/issues/358#issuecomment-578447122)
    #     _create_network_args = _create_network_args | 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network: trt.INetworkDefinition = builder.create_network(_create_network_args)

    config: trt.IBuilderConfig = builder.create_builder_config()

    config.builder_optimization_level = optimization_level  # max: 5, min: 0

    if max_aux_streams:
        config.max_aux_streams = max_aux_streams
    config.engine_capability = (
        trt.EngineCapability.STANDARD
    )  # default without targeting safety runtime, supporting GPU and DLA
    # config.engine_capability = trt.EngineCapability.SAFETY # targeting safety runtime, supporting GPU on NVIDIA Drive(R) products
    # config.engine_capability = trt.EngineCapability.DLA_STANDALONE # targeting DLA runtime, supporting DLA

    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html?highlight=precision_constraints#tensorrt.MemoryPoolType
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    # config.avg_timing_iterations = 10
    if workspace_size:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    if sparse_weights:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if weight_streaming:
        config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
    if precision_constraint == "obey":
        # layers execute in specified precisions
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    elif precision_constraint == "prefer":
        # layers execute in specified precisions but allow TRT to fall back to other precisions
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if strict_type:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    if refit:  # refitable model (swapable weights)
        config.set_flag(trt.BuilderFlag.REFIT)
    if algorithm_selector:
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        config.algorithm_selector = algorithm_selector
    if tactic_sources:
        # e.g. 1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS) | 1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
        config.set_tactic_sources(tactic_sources=tactic_sources)
    if (timing_cache is not None) and (timing_cache.exists()):
        with timing_cache.open("rb") as fp:
            cache: trt.ITimingCache = config.create_timing_cache(fp.read())
    else:
        cache: trt.ITimingCache = config.create_timing_cache(b"")
    if timing_cache is not None:
        with cache.serialize() as buffer:  # save cache
            with timing_cache.open("wb") as fp:
                fp.write(buffer)
                fp.flush()
                os.fsync(fp)
    config.set_timing_cache(cache, False)
    # config.progress_monitor = ProgressMonitor()
    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    elif builder.platform_has_tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.INT4)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if bf16:
        config.set_flag(trt.BuilderFlag.BF16)
    if fp8:
        config.set_flag(trt.BuilderFlag.FP8)
    # if visualize:
    #     config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TODO: hw, version compatibility
    # config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)
    # config.set_flag(trt.BuilderFlag.SAFETY_SCOPE)
    # config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
    # config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS

    print(f"config.default_device_type = {config.default_device_type}")
    print(f"config.max_aux_streams = {config.max_aux_streams}")
    print(f"config.plugins_to_serialize = {config.plugins_to_serialize}")
    print(
        "config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) = %d Byte (%.1f GiB)"
        % (
            config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE),
            config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) / (1 << 30),
        )
    )
    # config.set_memory_pool_limit(config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE))
    if shape_profiles:
        for group in shape_profiles:
            assert len(_input_names) == len(group), "mismatch num inputs, num shapes in profile"
            profile = builder.create_optimization_profile()
            for name, shape in group.items():
                opt = shape.get("opt")
                profile.set_shape(name, min=shape.get("min", opt), opt=shape.get("opt", opt), max=shape.get("max", opt))
            config.add_optimization_profile(profile)

    parser = trt.OnnxParser(network, logger)
    for error in range(parser.num_errors):
        print(parser.get_error(error))
    assert parser.parse_from_file(str(model_path)), "ONNX load failed"

    # NOTE: builder.build_engine(network, config) is deprecated
    plan: trt.IHostMemory = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("=== [WARNING] builder.build_serialized_network failed, exit -1 ===")
    engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(plan)
    print(f"TRT Engine uses: {engine.device_memory_size} bytes of Memory")

    # save engine file
    with open(engine_path, "wb") as fp:
        fp.write(engine.serialize())
    # with open(engine_path.parent.joinpath(engine_path.stem+".cache"), "wb") as fp:
    #     fp.write(cache)
    # if visualize:
    #     _visualize(engine, engine_path)

    return engine, engine_path