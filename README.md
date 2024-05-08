# MISA-benchmarking

IREE tools SHA (3ca0a49b424c7d9918a2e73d0c19c308e7d8e6db) 

Run on `smc300x-pla-t25-12` with `HSA_OVERRIDE_GFX_VERSION=9.4.0` environment variable set because MISA kernels don't have an explicit configuration for `gfx942`.

Choosing the top conv kernel in UNet for benchmarking purposes (https://docs.google.com/spreadsheets/d/17Xpaxo9l0kbB_5Jx7j4xfEgtzC1xRtmYbpLSViKFXaI/edit#gid=789226983). In this experiment snapshot the said 
top kernel has the following configuration and tflops (from results.txt)

```
[fwd:40] igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt128x128x32_wt32x32x8_ws1x1_wr2x2_ta1x8x2x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs, p_in:0x7f6507600000,p_wei:0x7f6505600000,p_out:0x7f6504e00000,hi:34,wi:34,n:2,k:1280,c:1280,ho:32,wo:32,stride_h:1,stride_w:1,dilation_h:1,dilation_w:1,pad_h:0,pad_w:0,y:3,x:3,group:1,magic_0:2576980378,magic_1:1,magic_2:1,magic_3:2576980378,magic_4:4,magic_5:30475536,shift_pack_0:134547972,shift_pack_1:16,ks:0,block:256,grid:160,splits:1,karg_size:128,[3], cost:0.183ms, tflops:329.672(41.27%), valid:y
```


