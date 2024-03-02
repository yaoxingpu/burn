use crate::element::FloatNdArrayElement;
use crate::NdArrayTensor;
use alloc::string::String;
use burn_common::stub::Mutex;
use burn_tensor::{backend::{Backend, BackendBridge, BackendPrecisionSettings, DoublePrecision}, Tensor};
use core::marker::PhantomData;
use rand::{rngs::StdRng, SeedableRng};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// The device type for the ndarray backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NdArrayDevice {
    /// The CPU device.
    Cpu,
}

impl Default for NdArrayDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

impl<F, TF> BackendBridge<BackendPrecisionSettings<TF, i64>> for NdArray<F>
where
    F: FloatNdArrayElement,
    TF: FloatNdArrayElement,
{
    type InputBackend = Self;
    type TargetBackend = NdArray<TF>;

    fn bridge_float<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D>,
        _settings: BackendPrecisionSettings<TF, i64>,
    ) -> burn_tensor::ops::FloatTensor<Self::TargetBackend, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();
        NdArrayTensor { array }
    }

    fn bridge_int<const D: usize>(
        tensor: burn_tensor::ops::IntTensor<Self, D>,
        _settings: BackendPrecisionSettings<TF, i64>,
    ) -> burn_tensor::ops::IntTensor<Self::TargetBackend, D> {
        let array = tensor.array;
        NdArrayTensor { array }
    }
}

#[allow(dead_code)]
fn allo() {
    let tensor: Tensor<NdArray<f32>, 1> = Tensor::ones([32], &Default::default());
    let tensor_full: Tensor<NdArray<f64>, 1> =  tensor.bridge(DoublePrecision::default());
}

/// Tensor backend that uses the [ndarray](ndarray) crate for executing tensor operations.
///
/// This backend is compatible with CPUs and can be compiled for almost any platform, including
/// `wasm`, `arm`, and `x86`.
#[derive(Clone, Copy, Default, Debug)]
pub struct NdArray<E = f32> {
    phantom: PhantomData<E>,
}

impl<E: FloatNdArrayElement> Backend for NdArray<E> {
    type Device = NdArrayDevice;
    type FullPrecisionElem = f32;
    type FullPrecisionBackend = NdArray<f32>;

    type FloatTensorPrimitive<const D: usize> = NdArrayTensor<E, D>;
    type IntTensorPrimitive<const D: usize> = NdArrayTensor<i64, D>;
    type BoolTensorPrimitive<const D: usize> = NdArrayTensor<bool, D>;

    type FloatElem = E;
    type IntElem = i64;

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        String::from("ndarray")
    }

    fn seed(seed: u64) {
        let rng = StdRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }
}
