// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx940", ukernels = "none"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

module attributes {transform.with_named_sequence} {

  transform.named_sequence @cast_and_call_dag(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    // transform.print {name = "hi"}
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    // transform.print {name = "hi"}
    transform.yield
  }


  util.func private @conv_entry_point(%arg0: tensor<2x34x34x1280xf16>, %arg1: tensor<3x3x1280x1280xf16>) 
                                          -> tensor<2x32x32x1280xf32> {
    %hi = arith.constant 34 : i32
    %wi = arith.constant 34 : i32
    %n = arith.constant 2 : i32
    %k = arith.constant 1280 : i32
    %c = arith.constant 1280 : i32
    %ho = arith.constant 32 : i32
    %wo = arith.constant 32 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 2576980378 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 2576980378 : i32
    %magic_4 = arith.constant 5 : i32
    %magic_5 = arith.constant 10151360 : i32
    %shift_pack_0 = arith.constant 168102405 : i32
    %shift_pack_1 = arith.constant 32 : i32
    %ks = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt64x64x64_wt16x16x16_ws1x1_wr2x2_ta1x8x2x1_1x8x1x32_tb1x8x2x1_1x8x1x32"(%hi, %wi, %n,
        %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group, 
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks,
        %arg0, %arg1) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        i32, i32, i32, i32, i32, i32, i32, i32, tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) -> tensor<2x32x32x1280xf16>
      count(%device: !hal.device) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        %c80_0 = arith.constant 640 : index
        hal.return %c80_0, %c1_0, %c1_0 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 25, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/home/nmeganat/MISA-benchmarking/igemm_fwd_gtc_gfx940_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x32x32x1280xf16> to tensor<2x32x32x1280xf32>
    util.return %6 : tensor<2x32x32x1280xf32>
  }

  transform.named_sequence @match_conv(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x34x34x1280xf16>, %rhs: tensor<3x3x1280x1280xf16>):
        %cst_31 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x32x32x1280xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%cst_31 : f32) outs(%84 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%87 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op   
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_conv -> @cast_and_call_dag
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.apply_registered_pass "inline" to %module : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
