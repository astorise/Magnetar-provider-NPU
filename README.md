# magnetar-provider-npu

## Purpose

An Intel NPU ("VPU"-class) execution Provider for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime,
implementing Magnetar's `Provider` contract for NPU Devices via the real,
vendor-neutral oneAPI [Level Zero](https://github.com/oneapi-src/level-zero)
API -- the same low-level API Intel's own NPU driver stack exposes
(`ZE_INIT_FLAG_VPU_ONLY` is Level Zero's own real flag for NPU/VPU-class
devices specifically, distinct from its GPU discovery path).

## Status

**Real device-discovery skeleton, not a full Provider.** `NpuProvider::new()`
dynamically loads the real Level Zero loader shared library (`ze_loader.dll`
on Windows, `libze_loader.so.1` on Linux -- no build-time dependency on
either, the same `libloading`-based dynamic-loading posture
[`providers/rocm`](https://github.com/astorise/Magnetar-provider-ROCm)
already establishes for HIP) and calls the real, public
`zeInit(ZE_INIT_FLAG_VPU_ONLY)`/`zeDriverGet` Level Zero API entry points
(transcribed directly from the real, public `ze_api.h` header, not
recalled from memory) to discover whether a usable NPU driver exists. When
it does not -- including on every machine and CI runner this crate has
actually been developed and tested on, none of which has a real Intel NPU
or Level Zero driver installed -- it gracefully reports
`ProviderHealth::Unavailable` and zero Devices, exactly like `CudaProvider`
does for CUDA and `RocmProvider` does for ROCm when no compatible hardware
is present.

**What is real and verified**: the graceful-unavailability path itself.
`NpuProvider::new()` constructs successfully and `is_available()`/
`health()` correctly report "no NPU driver here" on a genuinely
Level-Zero-less machine -- confirmed on this crate's own development
environment and confirmed again by this repository's CI (`ubuntu-latest`,
also genuinely without a Level Zero NPU driver), the same
two-environments-agree verification `providers/rocm`'s own README already
establishes for the equivalent ROCm case.

**What is explicitly not implemented, and why**: no compute Kernels
(`ProviderExecutionApi`), no per-device enumeration beyond a raw driver
count, no Kernel advertisements. Writing real NPU compute kernels and
claiming they are correct without ever running them on real Intel NPU
hardware would be unverifiable guesswork -- exactly the category of
confident-but-untested work this repository's own development practice
avoids everywhere else. This crate stops at the boundary of what can
actually be verified from a Level-Zero-less development environment: real
driver discovery, honestly reporting unavailability when that is the
truth.

## What a future contributor with real Intel NPU hardware would need to do

1. Extend `src/ze_sys.rs` with the real per-driver device enumeration API
   (`zeDeviceGet`) and per-device properties (`zeDeviceGetProperties`),
   building real `magnetar_runtime::device::DeviceDescriptor` values from
   them, mirroring `providers/cuda`'s own `device.rs`.
2. Implement Level Zero's real module/kernel compilation and execution API
   (`zeModuleCreate`, `zeKernelCreate`, `zeCommandListAppendLaunchKernel`)
   and `ProviderExecutionApi`, mirroring `providers/cuda`'s own
   `CudaKernels`/`CudaExecutor`.
3. Verify every ported Kernel against `providers/cpu`'s reference
   implementation on real Intel NPU hardware, the same `provider-compute`
   conformance profile `providers/cuda` already passes for CUDA.

## Governing contract

Implements Magnetar's generic `Provider`/`Device`/`ProviderExecutionApi`
contracts from the main [Magnetar](https://github.com/astorise/Magnetar)
repository's `magnetar-runtime` crate. No dedicated OpenSpec capability
exists yet for this crate specifically; one should be scoped in the main
repository's `openspec/specs/` once real Kernel execution work here
begins, the same way `providers/cuda`'s own `cuda-provider` capability was.

## Relationship to magnetar-runtime

This Provider is loaded and driven by the Runtime's own Provider registry,
never the reverse -- `magnetar-runtime` has zero compile-time dependency
on this crate (the same externalization invariant every Provider/Component/
Format module in this workspace observes). It is pinned into the main
Magnetar repository as a git submodule at `providers/npu`.
