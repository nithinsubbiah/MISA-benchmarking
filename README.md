# MISA-benchmarking

IREE tools SHA (3ca0a49b424c7d9918a2e73d0c19c308e7d8e6db) 

Run on `smc300x-pla-t25-12` with `HSA_OVERRIDE_GFX_VERSION=9.4.0` environment variable set because MISA kernels don't have an explicit configuration for `gfx942`.

Choosing the top conv kernel in UNet for benchmarking purposes (https://docs.google.com/spreadsheets/d/17Xpaxo9l0kbB_5Jx7j4xfEgtzC1xRtmYbpLSViKFXaI/edit#gid=789226983). In this experiment snapshot the said 
top kernel has the following configuration and tflops (from results.txt)

```
[fwd:40] igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt128x128x32_wt32x32x8_ws1x1_wr2x2_ta1x8x2x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs, p_in:0x7f6507600000,p_wei:0x7f6505600000,p_out:0x7f6504e00000,hi:34,wi:34,n:2,k:1280,c:1280,ho:32,wo:32,stride_h:1,stride_w:1,dilation_h:1,dilation_w:1,pad_h:0,pad_w:0,y:3,x:3,group:1,magic_0:2576980378,magic_1:1,magic_2:1,magic_3:2576980378,magic_4:4,magic_5:30475536,shift_pack_0:134547972,shift_pack_1:16,ks:0,block:256,grid:160,splits:1,karg_size:128,[3], cost:0.183ms, tflops:329.672(41.27%), valid:y
```

# Microbenchmark
Benchmark information for a conv kernel in IREE with MISA kernel integrated. Note that this is a microbenchmark with only conv kernel in MLIR. Below are the IREE compile and run commands:
For MISA kernel: 
```
iree-compile --iree-hal-target-backends=rocm --iree-hal-target-backends=rocm     --iree-rocm-target-chip=gfx940 --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode/ --iree-preprocessing-transform-spec-filename=conv_spec_microkernel.mlir --iree-util-zero-fill-elided-attrs elided_conv.mlir -o conv_misa.vmfb
```

For IREE kernel:
```
iree-compile --iree-hal-target-backends=rocm --iree-hal-target-backends=rocm     --iree-rocm-target-chip=gfx940 --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode/ --iree-codegen-llvmgpu-use-vector-distribution=true --iree-llvmgpu-enable-prefetch --iree-util-zero-fill-elided-attrs elided_conv.mlir -o conv_iree.vmfb
```

Inputs values are generated using `numpy.random.rand`, converted to `numpy.float16`, and fed as `.npy` files to `iree-run-module`.
IREE kernel:
```
nmeganat@smc300x-pla-t25-12:~/MISA-benchmarking$ ./../iree/build_rocm/tools/iree-benchmark-module --device=rocm --module=conv_iree.vmfb --function=forward -
-input=@input1.npy --input=@input2.npy --benchmark_repitions=50
2024-05-07T21:09:15-05:00
Running ./../iree/build_rocm/tools/iree-benchmark-module
Run on (128 X 3799.07 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x64)
  L1 Instruction 32 KiB (x64)
  L2 Unified 1024 KiB (x64)
  L3 Unified 32768 KiB (x16)
Load Average: 125.10, 56.79, 21.95
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
***WARNING*** Library was built as DEBUG. Timings may be affected.
--------------------------------------------------------------------------------------------
Benchmark                                  Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------
BM_forward/process_time/real_time       1.58 ms         1.67 ms          397 items_per_second=633.357/s
```

MISA kernel:
```
nmeganat@smc300x-pla-t25-12:~/MISA-benchmarking$ ./../iree/build_rocm/tools/iree-benchmark-module --device=rocm --module=conv_misa.vmfb --function=forward --input=@input1.npy --input=@input2.npy --benchmark_repitions=50
2024-05-07T21:09:46-05:00
Running ./../iree/build_rocm/tools/iree-benchmark-module
Run on (128 X 3799.07 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x64)
  L1 Instruction 32 KiB (x64)
  L2 Unified 1024 KiB (x64)
  L3 Unified 32768 KiB (x16)
Load Average: 134.08, 65.33, 25.89
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
***WARNING*** Library was built as DEBUG. Timings may be affected.
--------------------------------------------------------------------------------------------
Benchmark                                  Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------
BM_forward/process_time/real_time       1.37 ms         1.85 ms          381 items_per_second=728.367/s
```

There is a slight deviation in the result values which may be due to typecast but overall values are close enough. Artifacts for this experiment can be found on this repo.

#UNet benchmarking

UNet is benchmarked using the following compile and run commands (reference: https://github.com/nod-ai/sdxl-scripts). Note that it uses a different spec than the microkernel because of the different matcher:

With MISA kernel
```
iree-compile --iree-hal-target-backends=rocm --iree-rocm-target-chip=gfx940 \
--iree-rocm-bc-dir=$PWD/bitcode-2024-03-07 --iree-global-opt-propagate-transposes=true --iree-opt-outer-dim-concat=true \
--iree-opt-data-tiling=false --iree-opt-const-eval=false --iree-codegen-llvmgpu-use-vector-distribution --iree-vm-target-truncate-unsupported-floats \
--iree-llvmgpu-enable-prefetch --iree-codegen-gpu-native-math-precision=true --iree-rocm-waves-per-eu=2 \
--iree-flow-enable-aggressive-fusion --iree-global-opt-enable-fuse-horizontal-contractions=true \
--iree-opt-aggressively-propagate-transposes=true --iree-execution-model=async-external \
--iree-hal-dump-executable-configurations-to=configurations/unet --iree-hal-dump-executable-sources-to=sources/unet \
--iree-hal-dump-executable-binaries-to=binaries/unet --iree-hal-dump-executable-benchmarks-to=benchmarks/unet \
--iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics))" \
--iree-codegen-transform-dialect-library=./specs/attention_and_matmul_spec.mlir \
--iree-preprocessing-transform-spec-filename=./../MISA-benchmarking/conv_spec_unet.mlir ./base_ir/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet.mlir -
o ./tmp/unet_spec.vmfb
```
```
nmeganat@smc300x-pla-t25-12:~/sdxl-scripts$ TRACY_NO_EXIT=1 ./../iree/build_rocm/tools/iree-benchmark-module --device=rocm --device_allocator=caching --module=./tmp/unet_spec.vmfb --parameters=model=./scheduled_unet_fp16.irpa --function=main --input=1x4x128x128xf16 --input=1xi64 --input=2x64x2048xf16 --input=2x1280xf16 --input=2x6xf16 --input=1xf16  --benchmark_repetitions=3
2024-05-07T21:51:13-05:00
Running ./../iree/build_rocm/tools/iree-benchmark-module
Run on (128 X 3799.07 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x64)
  L1 Instruction 32 KiB (x64)
  L2 Unified 1024 KiB (x64)
  L3 Unified 32768 KiB (x16)
Load Average: 1.73, 0.70, 3.52
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
***WARNING*** Library was built as DEBUG. Timings may be affected.
------------------------------------------------------------------------------------------------
Benchmark                                      Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------
BM_main/process_time/real_time              84.3 ms          166 ms            8 items_per_second=11.8633/s
BM_main/process_time/real_time              84.6 ms          168 ms            8 items_per_second=11.815/s
BM_main/process_time/real_time              84.5 ms          167 ms            8 items_per_second=11.8311/s
BM_main/process_time/real_time_mean         84.5 ms          167 ms            3 items_per_second=11.8364/s
BM_main/process_time/real_time_median       84.5 ms          167 ms            3 items_per_second=11.8311/s
BM_main/process_time/real_time_stddev      0.175 ms        0.915 ms            3 items_per_second=0.0245839/s
BM_main/process_time/real_time_cv           0.21 %          0.55 %             3 items_per_second=0.21%
```

Without MISA kernel - same compile command as MISA except not using the `conv_spec_unet.mlir`.
```
nmeganat@smc300x-pla-t25-12:~/sdxl-scripts$ TRACY_NO_EXIT=1 ./../iree/build_rocm/tools/iree-benchmark-module --device=rocm --device_allocator=caching --module=./tmp/unet.vmfb --parameters=model=./scheduled_unet_fp16.irpa --function=main --input=1x4x128x128xf16 --input=1xi64 --input=2x64x2048xf16 --input=2x1280x
f16 --input=2x6xf16 --input=1xf16  --benchmark_repetitions=3
2024-05-07T21:56:51-05:00
Running ./../iree/build_rocm/tools/iree-benchmark-module
Run on (128 X 3799.07 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x64)
  L1 Instruction 32 KiB (x64)
  L2 Unified 1024 KiB (x64)
  L3 Unified 32768 KiB (x16)
Load Average: 0.95, 0.90, 2.72
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
***WARNING*** Library was built as DEBUG. Timings may be affected.
------------------------------------------------------------------------------------------------
Benchmark                                      Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------
BM_main/process_time/real_time              84.1 ms          166 ms            8 items_per_second=11.892/s
BM_main/process_time/real_time              84.3 ms          167 ms            8 items_per_second=11.8568/s
BM_main/process_time/real_time              84.2 ms          167 ms            8 items_per_second=11.8724/s
BM_main/process_time/real_time_mean         84.2 ms          167 ms            3 items_per_second=11.8738/s
BM_main/process_time/real_time_median       84.2 ms          167 ms            3 items_per_second=11.8724/s
BM_main/process_time/real_time_stddev      0.125 ms        0.754 ms            3 items_per_second=0.0176346/s
BM_main/process_time/real_time_cv           0.15 %          0.45 %             3 items_per_second=0.15%
```
NOTE: The divergence in results between the two is not insignificant, need to debug why the results vary.
