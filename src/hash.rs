// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.

//! Hash trait abstraction for use in hash-based polynomial commitment schemes.
#![allow(unused_imports, dead_code)]

use ff::PrimeField;
use rayon::prelude::*;
use sha3::digest::{Digest, HashMarker};
pub use sha3::{
  Keccak256,
  digest::{FixedOutputReset, Output, Update},
};
use std::fmt::Debug;

pub trait Hash:
  'static + Sized + Clone + Debug + Default + FixedOutputReset + Update + HashMarker
{
  fn new() -> Self {
    Self::default()
  }

  fn update_field_element(&mut self, field_element: &impl PrimeField) {
    Digest::update(self, field_element.to_repr());
  }

  fn digest(data: impl AsRef<[u8]>) -> Output<Self> {
    let mut hasher = Self::default();
    hasher.update(data.as_ref());
    hasher.finalize()
  }

  /// Hash many fixed-size chunks in parallel.
  ///
  /// `input` is a flat byte slice of `output.len()` chunks, each `chunk_size` bytes.
  /// Each chunk is hashed independently into the corresponding `output[i]` digest.
  fn hash_many(input: &[u8], chunk_size: usize, output: &mut [[u8; 32]]) {
    debug_assert_eq!(input.len(), chunk_size * output.len());
    input
      .par_chunks(chunk_size)
      .zip(output.par_iter_mut())
      .for_each(|(chunk, out)| {
        let result = Self::digest(chunk);
        debug_assert_eq!(result.len(), 32, "Hash output must be 32 bytes");
        out.copy_from_slice(&result[..32]);
      });
  }
}

impl<T: 'static + Sized + Clone + Debug + Default + FixedOutputReset + Update + HashMarker> Hash
  for T
{
}
