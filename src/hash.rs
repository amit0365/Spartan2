// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.

//! Hash trait abstraction for use in hash-based polynomial commitment schemes.
#![allow(unused_imports, dead_code)]

use ff::PrimeField;
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
}

impl<T: 'static + Sized + Clone + Debug + Default + FixedOutputReset + Update + HashMarker> Hash
  for T
{
}
