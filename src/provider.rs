//! `NpuProvider`: a real device-discovery skeleton for Intel's NPU
//! ("VPU"-class) devices via the real oneAPI Level Zero loader API,
//! deliberately not yet an optimized, kernel-executing Provider.
//!
//! # Status: driver discovery only, no compute kernels
//!
//! Unlike `providers/cuda`'s `CudaProvider` (a full, real-hardware-
//! verified Provider), this crate implements only the "does a compatible
//! Level Zero NPU driver exist" half of that same pattern --
//! `NpuProvider::new()` always constructs successfully and gracefully
//! reports [`ProviderHealth::Unavailable`] when no Level Zero driver
//! reporting NPU/VPU-class devices is found, exactly like `CudaProvider`
//! does for CUDA and `RocmProvider` does for ROCm, but it advertises zero
//! Kernels and zero real compute capability even when a driver *is*
//! found. Real Kernel execution needs real Intel NPU hardware to write
//! and verify against, which no machine this crate has been developed on
//! has. Implementing that is real, tracked future work for whoever has
//! that hardware -- not attempted here as unverifiable guesswork.
//!
//! # Graceful unavailability
//!
//! `NpuProvider::new()` never fails Runtime initialization for the
//! absence of hardware or the Level Zero loader shared library: it
//! attempts to dynamically load the real loader library (`ze_sys.rs`'s
//! `ZE_LOADER_LIBRARY_CANDIDATES`) and call `zeInit(ZE_INIT_FLAG_VPU_ONLY)`
//! followed by `zeDriverGet`; any failure at any step (library not found,
//! symbol not found, a real Level Zero call reporting an error, or zero
//! drivers found) is treated uniformly as "no usable NPU driver here" --
//! [`ProviderHealth::Unavailable`], not a construction error. This is
//! also what keeps this crate's `cargo test` green on every machine and
//! CI runner this repository currently has (all genuinely without a
//! Level Zero NPU driver): the only thing that can fail is the runtime
//! `dlopen` attempt, which this module already treats as a normal,
//! expected outcome, the same posture `providers/rocm`'s `RocmProvider`
//! established first for HIP.

use magnetar_runtime::affinity::ProviderHealth;
use magnetar_runtime::provider::{Provider, ProviderError, ProviderMetadata, ProviderRegistry};

use crate::ze_sys::{
    ZE_INIT_FLAG_VPU_ONLY, ZE_LOADER_LIBRARY_CANDIDATES, ZE_RESULT_SUCCESS, ZeDriverGetFn, ZeInitFn,
};

const NPU_PROVIDER_NAME: &str = "npu";
const NPU_PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");
const NPU_PROVIDER_VENDOR: &str = "Intel (oneAPI Level Zero)";

pub fn npu_provider_metadata() -> ProviderMetadata {
    ProviderMetadata::new(
        NPU_PROVIDER_NAME,
        NPU_PROVIDER_VERSION,
        NPU_PROVIDER_VENDOR,
        "Device-discovery skeleton for Intel NPU (\"VPU\"-class) devices via \
         oneAPI Level Zero -- no compute Kernels implemented yet (see this \
         crate's own README)",
    )
}

/// The NPU Provider itself. Registers as a built-in Provider, matching
/// `RocmProvider`/`CudaProvider`'s own posture (no dynamic-library
/// Provider ABI in this baseline).
pub struct NpuProvider {
    metadata: ProviderMetadata,
    /// `Some(count)` only when the real Level Zero loader was dynamically
    /// loaded, `zeInit(ZE_INIT_FLAG_VPU_ONLY)` succeeded, and `zeDriverGet`
    /// reported at least one driver. `count` itself is currently unused
    /// beyond "is it positive" -- no per-device descriptor is built yet
    /// (see this module's own doc comment: driver discovery only, not
    /// full Device enumeration).
    driver_count: Option<u32>,
}

impl NpuProvider {
    pub fn new() -> Self {
        Self {
            metadata: npu_provider_metadata(),
            driver_count: Self::discover_driver_count(),
        }
    }

    /// Attempts to dynamically load the real Level Zero loader library
    /// and query its real NPU/VPU-class driver count. Returns `None` for
    /// *any* failure along the way (library not found, symbol not found,
    /// a real Level Zero call reporting an error, or zero drivers) --
    /// this baseline draws no distinction between those cases; they are
    /// all "no usable NPU driver here" to a caller.
    fn discover_driver_count() -> Option<u32> {
        let library = ZE_LOADER_LIBRARY_CANDIDATES
            .iter()
            .find_map(|name| unsafe { libloading::Library::new(*name).ok() })?;

        // SAFETY: `zeInit`/`zeDriverGet` are real, stable, publicly
        // documented Level Zero API entry points (`ze_sys.rs`, transcribed
        // from the real `ze_api.h` header); the symbol lookups below only
        // succeed against a real Level Zero loader library actually
        // exporting them under these exact names, and the function-pointer
        // types declared in `ze_sys.rs` match Level Zero's own real C
        // signatures.
        unsafe {
            let ze_init: libloading::Symbol<ZeInitFn> = library.get(b"zeInit\0").ok()?;
            if ze_init(ZE_INIT_FLAG_VPU_ONLY) != ZE_RESULT_SUCCESS {
                return None;
            }
            let ze_driver_get: libloading::Symbol<ZeDriverGetFn> =
                library.get(b"zeDriverGet\0").ok()?;
            let mut count: u32 = 0;
            if ze_driver_get(&mut count, std::ptr::null_mut()) != ZE_RESULT_SUCCESS {
                return None;
            }
            if count == 0 { None } else { Some(count) }
        }
    }

    /// Whether this Provider found a usable Level Zero NPU/VPU driver.
    pub fn is_available(&self) -> bool {
        self.driver_count.is_some()
    }
}

impl Default for NpuProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for NpuProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        // No Devices to register yet -- this baseline does not build a
        // real `DeviceDescriptor` even when a driver is found (see this
        // module's own "Status" doc comment); `devices()` keeps the
        // `Provider` trait's own empty-`Vec` default.
        Ok(())
    }

    fn health(&self) -> ProviderHealth {
        if self.is_available() {
            ProviderHealth::Available
        } else {
            ProviderHealth::Unavailable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one thing genuinely verified on every machine this crate has
    /// actually run on (this development environment, and CI's own
    /// Level-Zero-less runners): a machine with no real NPU driver
    /// installed gets a Provider that constructs successfully and
    /// honestly reports itself unavailable, never a panic or a false
    /// "available".
    #[test]
    fn reports_unavailable_without_a_real_npu_driver() {
        let provider = NpuProvider::new();
        assert!(
            !provider.is_available(),
            "this test environment has no real Level Zero NPU driver installed"
        );
        assert_eq!(provider.health(), ProviderHealth::Unavailable);
    }

    #[test]
    fn metadata_is_well_formed() {
        let provider = NpuProvider::new();
        let metadata = provider.metadata();
        assert_eq!(metadata.name, NPU_PROVIDER_NAME);
        assert_eq!(metadata.vendor, NPU_PROVIDER_VENDOR);
        assert!(!metadata.description.is_empty());
    }

    #[test]
    fn devices_are_empty_without_a_real_npu_driver() {
        let provider = NpuProvider::new();
        assert!(provider.devices().is_empty());
    }
}
