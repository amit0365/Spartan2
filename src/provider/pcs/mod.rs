// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module provides implementations of polynomial commitment schemes (PCS).

use std::fmt::Debug;

// helper code for polynomial commitment schemes
pub mod ipa;

// implementations of polynomial commitment schemes
pub mod hyrax_pc;
#[allow(dead_code)]
pub mod whir;

pub mod irs;

/// Trait for a Reed-Solomon encoder over field `F`.
///
/// Used by hash-based PCS (WHIR, Brakedown) to encode polynomial coefficients
/// into codewords for proximity testing. Implementations are expected to use
/// NTT-based encoding with coset decomposition for efficiency.
pub trait ReedSolomon<F>: Debug + Send + Sync {
  /// Returns the next supported order >= `size`, or `None` if too large.
  ///
  /// The result is an NTT-smooth number suitable for `codeword_length`.
  fn next_order(&self, size: usize) -> Option<usize>;

  /// Returns evaluation points at the given indices.
  ///
  /// `message_length`: message length including mask values.
  /// `codeword_length`: must be a supported order >= `message_length`.
  /// `indices`: positions within `[0, codeword_length)`.
  fn evaluation_points(
    &self,
    message_length: usize,
    codeword_length: usize,
    indices: &[usize],
  ) -> Vec<F>;

  /// Compute a masked interleaved Reed-Solomon encoding.
  ///
  /// `messages`: `num_messages` slices of `message_length` elements each.
  /// `masks`: flat `num_messages * mask_length` blinding coefficients.
  /// `codeword_length`: NTT-smooth number >= `message_length + mask_length`.
  ///
  /// Returns `codeword_length * num_messages` evaluations in row-major order.
  fn interleaved_encode(&self, messages: &[&[F]], masks: &[F], codeword_length: usize) -> Vec<F>;
}
