//! Minimal, hand-declared FFI surface for Intel's real oneAPI Level Zero
//! loader API -- only the two stable, well-documented entry points this
//! baseline's device discovery actually calls, matching the real public
//! Level Zero API signatures (`ze_api.h`): `zeInit`, `zeDriverGet`.
//! Level Zero is a real, vendor-neutral low-level API for GPU- and
//! NPU-class ("VPU" in the API's own vocabulary) compute devices,
//! published and maintained by the `oneapi-src/level-zero` project.
//! Real per-driver device enumeration (`zeDeviceGet`) or per-device
//! properties (`zeDeviceGetProperties`) are real future work, not
//! declared here yet -- see this crate's own README for why driver
//! discovery alone is this baseline's whole scope.
//!
//! # Honesty about verification
//!
//! These declarations are transcribed verbatim from the real, public
//! `ze_api.h` header (`oneapi-src/level-zero`, fetched directly rather
//! than recalled from memory) and are exercised for real by
//! [`crate::provider::NpuProvider::new`]'s device-discovery path on every
//! machine this repository's own tests and CI actually run on -- but no
//! machine this crate has been developed or tested on has a real Intel
//! NPU or Level Zero driver installed. The only path genuinely exercised
//! so far is the loader library's absence (dynamic loading fails
//! outright) -- never a real, successful call against real hardware. See
//! this crate's own README for what would be needed to close that gap.

use std::os::raw::c_int;

/// `ze_result_t`'s `ZE_RESULT_SUCCESS` value (`0`) -- the only variant
/// this baseline's device-discovery path distinguishes; any other value
/// is treated uniformly as "discovery failed", matching
/// [`super::provider::NpuProvider`]'s own "no partial credit"
/// graceful-unavailability posture (the same one `providers/rocm`'s
/// `RocmProvider` already establishes for the HIP runtime API).
pub const ZE_RESULT_SUCCESS: c_int = 0;

/// `ze_init_flags_t`'s `ZE_INIT_FLAG_VPU_ONLY` bit (`1 << 1`) -- Level
/// Zero's own real flag restricting initialization/enumeration to
/// NPU-class ("VPU") devices specifically, rather than GPUs. Passing this
/// (instead of `0`, which would also discover GPUs) keeps this crate's
/// device-discovery scope honestly aligned with its own name.
pub const ZE_INIT_FLAG_VPU_ONLY: u32 = 1 << 1;

/// `ze_result_t ZE_APICALL zeInit(ze_init_flags_t flags)` -- real Level
/// Zero API, must be called before any other Level Zero entry point.
pub type ZeInitFn = unsafe extern "C" fn(flags: u32) -> c_int;

/// `ze_result_t ZE_APICALL zeDriverGet(uint32_t* pCount, ze_driver_handle_t* phDrivers)`.
/// Passing a null `phDrivers` (as this baseline does) queries only the
/// real driver count, matching the real API's own documented
/// query-then-allocate calling convention.
pub type ZeDriverGetFn =
    unsafe extern "C" fn(count: *mut u32, drivers: *mut *mut std::ffi::c_void) -> c_int;

/// Real Level Zero loader shared library file names this crate attempts
/// to dynamically load, in order, per platform -- the real, documented
/// install-time artifact names a real Level Zero driver installation
/// provides. No build-time dependency on any of them (`libloading`
/// resolves purely at runtime), matching `providers/rocm`'s own
/// dynamic-loading posture for HIP.
#[cfg(target_os = "windows")]
pub const ZE_LOADER_LIBRARY_CANDIDATES: &[&str] = &["ze_loader.dll"];
#[cfg(target_os = "linux")]
pub const ZE_LOADER_LIBRARY_CANDIDATES: &[&str] = &["libze_loader.so.1", "libze_loader.so"];
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
pub const ZE_LOADER_LIBRARY_CANDIDATES: &[&str] = &[];
