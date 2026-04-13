// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Interleaved Reed-Solomon commitment configuration and soundness analysis.
//! Reference: https://github.com/WizardOfMenlo/whir/
//!
//! Provides [`IrsConfig`] for computing per-round protocol parameters (sample counts,
//! code rate, Johnson slack) targeting a given security level.
use std::{f64::consts::LOG2_10, ops::Neg};

use ff::PrimeField;

use crate::provider::{ntt::NttEngine, pcs::ReedSolomon};

/// Configuration for one round of the interleaved Reed-Solomon commitment protocol.
///
/// Derived from security parameters via [`IrsConfig::new`]; not constructed directly.
/// Stores all values needed for soundness analysis and for driving the prover/verifier.
#[derive(Clone, Debug)]
pub struct IrsConfig {
  /// Number of polynomial coefficients.
  pub vector_size: usize,
  /// Number of RS evaluation points per column, $\lceil \text{message\_length} / \rho \rceil$.
  pub codeword_length: usize,
  /// Number of interleaved codewords; splits the polynomial into this many columns.
  pub interleaving_depth: usize,
  /// Number of random in-domain Merkle query indices per round.
  pub in_domain_samples: usize,
  /// Number of out-of-domain challenge points per round. Zero when `unique_decoding`.
  pub out_domain_samples: usize,
  /// Johnson-bound slack $\eta$. Zero for unique decoding, $\sqrt{\rho}/20$ otherwise.
  pub johnson_slack: f64,
  /// True when operating in the unique-decoding regime ($\eta = 0$, no OOD samples).
  pub unique_decoding: bool,
  /// Field size $\log_2 |\mathbb{F}|$, captured from `F::NUM_BITS` at construction time.
  pub field_size_bits: f64,
}

impl IrsConfig {
  /// Computes IRS parameters targeting `security_target` bits of round-by-round soundness.
  ///
  /// Selects `in_domain_samples`, `out_domain_samples`, and Johnson slack $\eta$. The actual rate may differ slightly from the
  /// requested `rate` due to ceiling of `codeword_length`.
  ///
  /// # Arguments
  /// * `security_target` - Target bits of security per round (after PoW deduction)
  /// * `unique_decoding` - Use unique-decoding bound ($\eta = 0$) instead of list decoding
  /// * `vector_size` - Number of polynomial coefficients
  /// * `interleaving_depth` - Number of interleaved codewords (columns)
  /// * `rate` - Target code rate $\rho \in (0, 1]$
  pub fn new<F: PrimeField>(
    security_target: f64,
    unique_decoding: bool,
    vector_size: usize,
    interleaving_depth: usize,
    rate: f64,
  ) -> Self {
    let message_length = vector_size / interleaving_depth;
    let codeword_length = (message_length as f64 / rate).ceil() as usize;
    let rate = message_length as f64 / codeword_length as f64; // actual rate after rounding

    let johnson_slack = if unique_decoding {
      0.0
    } else {
      rate.sqrt() / 20.0
    };

    let field_size_bits = F::NUM_BITS as f64;

    #[allow(clippy::cast_sign_loss)]
    let out_domain_samples = if unique_decoding {
      0
    } else {
      // Johnson list size bound 1 / (2 η √ρ)
      let list_size = 1. / (2. * johnson_slack * rate.sqrt());

      // The list size error is (L choose 2) * [(d - 1) / |𝔽|]^s
      // See [STIR] lemma 4.5.
      // We want to find s such that the error is less than security_target.
      let l_choose_2 = list_size * (list_size - 1.) / 2.;
      let log_per_sample = field_size_bits - ((vector_size - 1) as f64).log2();
      assert!(log_per_sample > 0.);
      ((security_target + l_choose_2.log2()) / log_per_sample)
        .ceil()
        .max(1.) as usize
    };

    #[allow(clippy::cast_sign_loss)]
    let in_domain_samples = {
      // Query error is (1 - δ)^q, so we compute 1 - δ
      let per_sample = if unique_decoding {
        // Unique decoding bound: δ = (1 - ρ) / 2
        f64::midpoint(1., rate)
      } else {
        // Johnson bound: δ = 1 - √ρ - η
        rate.sqrt() + johnson_slack
      };
      (security_target / (-per_sample.log2())).ceil() as usize
    };
    Self {
      vector_size,
      codeword_length,
      interleaving_depth,
      in_domain_samples,
      out_domain_samples,
      johnson_slack,
      unique_decoding,
      field_size_bits,
    }
  }

  /// Number of coefficients per column.
  pub fn message_length(&self) -> usize {
    assert!(self.vector_size.is_multiple_of(self.interleaving_depth));
    self.vector_size / self.interleaving_depth
  }

  /// Maps Merkle leaf indices to NTT evaluation domain points $\omega^i$.
  pub fn evaluation_points<F: PrimeField>(&self, indices: &[usize]) -> Vec<F> {
    NttEngine::<F>::new_from_cache().evaluation_points(
      self.vector_size / self.interleaving_depth,
      self.codeword_length,
      indices,
    )
  }

  /// Actual code rate $\rho$.
  pub fn rate(&self) -> f64 {
    self.message_length() as f64 / self.codeword_length as f64
  }

  /// Returns true when operating in the unique-decoding regime.
  pub fn unique_decoding(&self) -> bool {
    self.out_domain_samples == 0 && self.johnson_slack == 0.0
  }

  /// Compute a list size bound.
  pub fn list_size(&self) -> f64 {
    if self.unique_decoding() {
      1.
    } else {
      // This is the Johnson bound $1 / (2 η √ρ)$.
      1. / (2. * self.johnson_slack * self.rate().sqrt())
    }
  }

  /// Round-by-round soundness of the out-of-domain samples in bits.
  pub fn rbr_ood_sample(&self) -> f64 {
    let list_size = self.list_size();
    let log_field_size = self.field_size_bits;
    // See [STIR] lemma 4.5.
    let l_choose_2 = list_size * (list_size - 1.) / 2.;
    let log_per_sample = ((self.vector_size - 1) as f64).log2() - log_field_size;
    -l_choose_2.log2() - self.out_domain_samples as f64 * log_per_sample
  }

  /// Round-by-round soundness of the in-domain queries in bits.
  pub fn rbr_queries(&self) -> f64 {
    let per_sample = if self.unique_decoding() {
      // 1 - δ = 1 - (1 + ρ) / 2
      f64::midpoint(1., self.rate())
    } else {
      // 1 - δ = sqrt(ρ) + η
      self.rate().sqrt() + self.johnson_slack
    };
    self.in_domain_samples as f64 * per_sample.log2().neg()
  }

  /// Round-by-round proximity gaps soundness of the fold step, in bits.
  pub fn rbr_soundness_fold_prox_gaps(&self) -> f64 {
    let log_field_size = self.field_size_bits;
    let log_inv_rate = self.rate().log2().neg();
    let log_k = (self.message_length() as f64).log2();
    // See WHIR Theorem 4.8
    // Recall, at each round we are only folding by two at a time
    let error = if self.unique_decoding() {
      log_k + log_inv_rate
    } else {
      let log_eta = self.johnson_slack.log2();
      // Make sure η hits the min bound.
      assert!(log_eta >= -(0.5 * log_inv_rate + LOG2_10 + 1.0) - 1e-6);
      7. * LOG2_10 + 3.5 * log_inv_rate + 2. * log_k
    };
    log_field_size - error
  }
}
