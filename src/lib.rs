//! Intel NPU ("VPU"-class) execution Provider for the Magnetar local AI
//! Runtime, via the real oneAPI Level Zero API.
//!
//! **Status: real device-discovery skeleton, no compute Kernels yet.**
//! See [`provider::NpuProvider`]'s own module doc comment and this
//! crate's README for what that means and why.

mod provider;
mod ze_sys;

pub use provider::{NpuProvider, npu_provider_metadata};
