//! WHIR protocol configuration and security parameter computation.
//! Reference: https://github.com/WizardOfMenlo/whir/
//!
//! Provides [`WhirConfig`], computed from user-facing [`WhirParams`] via [`WhirConfig::new`].
//! All per-round PoW budgets and sample counts are derived from the target security level.

use crate::provider::{bits::Bits, pcs::irs::IrsConfig, proof_of_work::PowConfig};
use ff::PrimeField;

/// User-facing WHIR protocol parameters.
#[derive(Clone, Debug)]
pub struct WhirParams {
  /// Target bits of security, e.g. 128.
  pub security_level: usize,
  /// Proof-of-work budget in bits, e.g. 20. Deducted from `security_level` before sampling.
  pub pow_bits: usize,
  /// Folding factor for the first IRS round, e.g. 4 folds by $2^4 = 16$.
  pub initial_folding_factor: usize,
  /// Folding factor for subsequent IRS rounds, e.g. 4.
  pub folding_factor: usize,
  /// $\log_2(1/\rho)$ for the starting code rate, e.g. 1 gives $\rho = 1/2$.
  pub starting_log_inv_rate: usize,
  /// Use unique-decoding bound ($\eta = 0$) instead of list decoding.
  pub unique_decoding: bool,
}

impl Default for WhirParams {
  fn default() -> Self {
    Self {
      security_level: 128,
      pow_bits: 20,
      initial_folding_factor: 4,
      folding_factor: 4,
      starting_log_inv_rate: 1,
      unique_decoding: false,
    }
  }
}

/// Fully computed WHIR protocol configuration.
///
/// Constructed via [`WhirConfig::new`]; not built directly. Stores all per-round
/// IRS, sumcheck, and PoW parameters derived from [`WhirParams`] and the field size.
#[derive(Clone, Debug)]
pub struct WhirConfig {
  /// Original user parameters used to construct this config.
  pub params: WhirParams,
  /// IRS commitment config for the first round.
  pub initial_irs: IrsConfig,
  /// Sumcheck config for the initial folding phase.
  pub initial_sumcheck: WhirSumcheckConfig,
  /// Per-round configs for each subsequent IRS + sumcheck + PoW cycle.
  pub round_configs: Vec<RoundConfig>,
  /// Sumcheck config for the final reduction (no new IRS commitment).
  pub final_sumcheck: WhirSumcheckConfig,
  /// PoW challenge after all round Merkle queries.
  pub final_pow: PowConfig,
}

/// Configuration for one WHIR protocol round.
#[derive(Clone, Debug)]
pub struct RoundConfig {
  /// IRS commitment config for this round's folded polynomial.
  pub irs_config: IrsConfig,
  /// Sumcheck config for this round's folding phase.
  pub sumcheck: WhirSumcheckConfig,
  /// PoW challenge between this round's Merkle queries and the next commit.
  pub pow: PowConfig,
}

/// Configuration for one sumcheck phase (initial, per-round, or final).
#[derive(Clone, Debug)]
pub struct WhirSumcheckConfig {
  /// Number of field elements at the start of this sumcheck phase.
  pub initial_size: usize,
  /// Number of folding rounds. Full reduction requires $\log_2(\text{initial\_size})$.
  pub num_rounds: usize,
  /// Proof-of-work budget applied after each sumcheck round.
  pub round_pow: PowConfig,
}

impl WhirSumcheckConfig {
  /// Number of elements remaining after all folding rounds: `initial_size >> num_rounds`.
  pub fn final_size(&self) -> usize {
    self.initial_size >> self.num_rounds
  }
}

impl WhirConfig {
  /// Computes all protocol parameters targeting `params.security_level` bits of security.
  ///
  /// Derives per-round IRS configs, sumcheck configs, and PoW budgets from the security
  /// target and field size $\log_2 |\mathbb{F}|$. `size` must be a power of two.
  ///
  /// # Arguments
  /// * `size` - Number of polynomial coefficients; must be a power of two
  /// * `params` - User-facing security and folding parameters
  #[allow(clippy::too_many_lines)]
  pub fn new<F: PrimeField>(size: usize, params: &WhirParams) -> Self {
    assert!(
      size.is_power_of_two(),
      "Only powers of two size are supported at the moment."
    );

    let pow = |difficulty: f64| PowConfig::from_difficulty(Bits::new(difficulty));
    let security_level = params.security_level as f64;
    let protocol_security_level = params.security_level.saturating_sub(params.pow_bits) as f64;
    let field_size_bits = F::NUM_BITS as f64;
    let mut log_inv_rate = params.starting_log_inv_rate;
    let mut num_variables = size.trailing_zeros() as usize;

    #[allow(clippy::cast_possible_wrap)]
    let initial_irs_config = IrsConfig::new::<F>(
      protocol_security_level,
      params.unique_decoding,
      size,
      1 << params.initial_folding_factor,
      0.5_f64.powi(params.starting_log_inv_rate as i32),
    );

    // Initial sumcheck round pow bits.
    let starting_folding_pow_bits = {
      let prox_gaps_error = initial_irs_config.rbr_soundness_fold_prox_gaps();
      let log_list_size = initial_irs_config.list_size().log2();
      let sumcheck_error = field_size_bits - log_list_size - 1.;
      let error = prox_gaps_error.min(sumcheck_error);
      (security_level - error).max(0.)
    };

    let mut round_configs = Vec::new();
    let mut round = 0;
    let mut in_domain_samples = initial_irs_config.in_domain_samples;
    let mut query_error = initial_irs_config.rbr_queries();
    num_variables -= params.initial_folding_factor;
    while num_variables >= params.folding_factor {
      // Queries are set w.r.t. to old rate, while the rest to the new rate
      let round_folding_factor = if round == 0 {
        params.initial_folding_factor
      } else {
        params.folding_factor
      };
      let next_rate = log_inv_rate + (round_folding_factor - 1);

      #[allow(clippy::cast_possible_wrap)]
      let irs_config = IrsConfig::new::<F>(
        protocol_security_level,
        params.unique_decoding,
        1 << num_variables,
        1 << params.folding_factor,
        0.5_f64.powi(next_rate as i32),
      );
      let combination_error = {
        let log_list_size = irs_config.list_size().log2();
        let count = irs_config.out_domain_samples + in_domain_samples;
        let log_combination = (count as f64).log2();
        field_size_bits - (log_combination + log_list_size + 1.)
      };
      let pow_bits = 0_f64.max(security_level - (query_error.min(combination_error)));
      let folding_pow_bits = {
        let prox_gaps_error = irs_config.rbr_soundness_fold_prox_gaps();
        let log_list_size = irs_config.list_size().log2();
        let sumcheck_error = field_size_bits - (log_list_size + 1.);
        let error = prox_gaps_error.min(sumcheck_error);
        (security_level - error).max(0.)
      };

      let config = RoundConfig {
        irs_config,
        sumcheck: WhirSumcheckConfig {
          initial_size: 1 << num_variables,
          round_pow: pow(folding_pow_bits),
          num_rounds: params.folding_factor,
        },
        pow: pow(pow_bits),
      };

      round += 1;
      num_variables -= params.folding_factor;
      log_inv_rate = next_rate;
      in_domain_samples = config.irs_config.in_domain_samples;
      query_error = config.irs_config.rbr_queries();
      round_configs.push(config);
    }

    let rbr_error = round_configs.last().map_or_else(
      || initial_irs_config.rbr_queries(),
      |r| r.irs_config.rbr_queries(),
    );
    let final_pow_bits = 0_f64.max(security_level - rbr_error);

    let final_folding_pow_bits = 0_f64.max(security_level - field_size_bits + 1.0);

    Self {
      params: params.clone(),
      initial_irs: initial_irs_config,
      initial_sumcheck: WhirSumcheckConfig {
        initial_size: size,
        round_pow: pow(starting_folding_pow_bits),
        num_rounds: params.initial_folding_factor,
      },
      round_configs,
      final_sumcheck: WhirSumcheckConfig {
        initial_size: 1 << num_variables,
        round_pow: pow(final_folding_pow_bits),
        num_rounds: num_variables,
      },
      final_pow: pow(final_pow_bits),
    }
  }

  /// Returns true when all rounds use the unique-decoding regime.
  pub fn unique_decoding(&self) -> bool {
    self.initial_irs.unique_decoding()
      && self
        .round_configs
        .iter()
        .all(|r| r.irs_config.unique_decoding())
  }

  /// Achieved security level in bits for the given batch dimensions.
  ///
  /// Takes the minimum over all round-by-round soundness terms. Returns 0.0 if
  /// the config is degenerate (e.g. empty protocol).
  ///
  /// # Arguments
  /// * `num_vectors` - Number of committed polynomials (1 for single-poly PCS)
  /// * `num_linear_forms` - Number of evaluation claims (1 for single-point evaluation)
  pub fn security_level<F: PrimeField>(&self, num_vectors: usize, num_linear_forms: usize) -> f64 {
    let field_size_bits = F::NUM_BITS as f64;
    let mut security_level = f64::INFINITY;
    if num_vectors > 1 {
      security_level = security_level.min(field_size_bits - ((num_vectors - 1) as f64).log2());
    }
    if num_linear_forms > 1 {
      security_level = security_level.min(field_size_bits - ((num_linear_forms - 1) as f64).log2());
    }
    let has_initial_constraints = num_linear_forms > 0 || self.initial_irs.out_domain_samples > 0;

    if !self.initial_irs.unique_decoding() {
      security_level = security_level.min(self.initial_irs.rbr_ood_sample());
    }

    // Initial sumcheck error (or the skipped version for LDT).
    let initial_prox_gaps_error = self.initial_irs.rbr_soundness_fold_prox_gaps();
    if has_initial_constraints {
      let log_list_size = self.initial_irs.list_size().log2();
      let initial_sumcheck_error = field_size_bits - (log_list_size + 1.);
      let initial_fold_error = initial_prox_gaps_error.min(initial_sumcheck_error)
        + f64::from(self.initial_sumcheck.round_pow.difficulty());
      security_level = security_level.min(initial_fold_error);
    };

    let mut rbr_queries = self.initial_irs.rbr_queries();
    let mut old_in_domain_samples = self.initial_irs.in_domain_samples;
    for round in &self.round_configs {
      // Query soundness is computed at the old rate, while all fold and OOD terms use the new rate.
      let new_unique_decoding = round.irs_config.unique_decoding();

      if !new_unique_decoding {
        let ood_error = round.irs_config.rbr_ood_sample();
        security_level = security_level.min(ood_error);
      }

      let log_list_size = round.irs_config.list_size().log2();
      let combination_error = {
        let count = round.irs_config.out_domain_samples + old_in_domain_samples;
        let log_combination = (count as f64).log2();
        field_size_bits - (log_combination + log_list_size + 1.)
      };
      let round_query_error =
        rbr_queries.min(combination_error) + f64::from(round.pow.difficulty());
      security_level = security_level.min(round_query_error);

      let prox_gaps_error = round.irs_config.rbr_soundness_fold_prox_gaps();
      let sumcheck_error = field_size_bits - (log_list_size + 1.);
      let round_fold_error =
        prox_gaps_error.min(sumcheck_error) + f64::from(round.sumcheck.round_pow.difficulty());
      security_level = security_level.min(round_fold_error);

      old_in_domain_samples = round.irs_config.in_domain_samples;
      rbr_queries = round.irs_config.rbr_queries();
    }

    let final_query_error = rbr_queries + f64::from(self.final_pow.difficulty());
    security_level = security_level.min(final_query_error);

    if self.final_sumcheck.num_rounds > 0 {
      let final_combination_error =
        field_size_bits - 1. + f64::from(self.final_sumcheck.round_pow.difficulty());
      security_level = security_level.min(final_combination_error);
    }

    if security_level.is_finite() {
      security_level
    } else {
      0.0
    }
  }

  /// Returns true when all PoW difficulties in the config are within `max_bits`.
  pub fn check_max_pow_bits(&self, max_bits: Bits) -> bool {
    if self.initial_sumcheck.round_pow.difficulty() > max_bits {
      return false;
    }
    for round_config in &self.round_configs {
      if round_config.pow.difficulty() > max_bits {
        return false;
      }
      if round_config.sumcheck.round_pow.difficulty() > max_bits {
        return false;
      }
    }
    if self.final_pow.difficulty() > max_bits {
      return false;
    }
    if self.final_sumcheck.round_pow.difficulty() > max_bits {
      return false;
    }
    true
  }

  /// Number of polynomial coefficients in the initial commitment.
  pub const fn initial_size(&self) -> usize {
    self.initial_irs.vector_size
  }

  /// Number of variables in the initial multilinear polynomial.
  pub fn initial_num_variables(&self) -> usize {
    assert!(self.initial_size().is_power_of_two());
    self.initial_size().trailing_zeros() as usize
  }

  /// Number of elements remaining after the final sumcheck reduction.
  pub fn final_size(&self) -> usize {
    self.final_sumcheck.final_size()
  }

  /// Number of IRS + sumcheck + PoW rounds (excludes the initial and final phases).
  pub const fn n_rounds(&self) -> usize {
    self.round_configs.len()
  }
}

impl RoundConfig {
  /// Number of polynomial coefficients at the start of this round.
  pub fn initial_size(&self) -> usize {
    assert_eq!(self.irs_config.vector_size, self.sumcheck.initial_size);
    self.sumcheck.initial_size
  }

  /// Number of elements remaining after this round's sumcheck reduction.
  pub fn final_size(&self) -> usize {
    self.sumcheck.final_size()
  }

  /// Number of multilinear variables at the start of this round.
  pub fn initial_num_variables(&self) -> usize {
    assert!(self.irs_config.vector_size.is_power_of_two());
    self.irs_config.vector_size.ilog2() as usize
  }

  /// Number of multilinear variables remaining after this round's sumcheck reduction.
  pub fn final_num_variables(&self) -> usize {
    self.initial_num_variables() - self.sumcheck.num_rounds
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;

  type F = pallas::Scalar;

  fn default_params() -> WhirParams {
    WhirParams::default()
  }

  #[test]
  fn test_list_decoding_has_ood_samples() {
    let config = WhirConfig::new::<F>(1 << 10, &default_params());
    assert!(config.initial_irs.out_domain_samples > 0);
    assert!(config.initial_irs.johnson_slack > 0.0);
  }

  #[test]
  fn test_unique_decoding_no_ood_samples() {
    let mut params = default_params();
    params.unique_decoding = true;
    let config = WhirConfig::new::<F>(1 << 10, &params);
    assert_eq!(config.initial_irs.out_domain_samples, 0);
    assert_eq!(config.initial_irs.johnson_slack, 0.0);
    assert!(config.unique_decoding());
  }

  #[test]
  fn test_security_level_meets_target() {
    let params = default_params();
    let config = WhirConfig::new::<F>(1 << 10, &params);
    let achieved = config.security_level::<F>(1, 1);
    // achieved security should be within 1 bit of target (rounding in sample counts)
    assert!(
      achieved >= params.security_level as f64 - 1.0,
      "achieved {achieved:.1} bits, target {} bits",
      params.security_level
    );
  }

  #[test]
  fn test_check_pow_bits_within_limits() {
    let mut config = WhirConfig::new::<F>(1 << 10, &default_params());
    config.initial_sumcheck.round_pow = PowConfig::from_difficulty(Bits::new(15.0));
    config.final_pow = PowConfig::from_difficulty(Bits::new(18.0));
    config.final_sumcheck.round_pow = PowConfig::from_difficulty(Bits::new(19.5));
    for r in &mut config.round_configs {
      r.pow = PowConfig::from_difficulty(Bits::new(17.0));
      r.sumcheck.round_pow = PowConfig::from_difficulty(Bits::new(19.0));
    }
    assert!(
      config.check_max_pow_bits(Bits::new(20.0)),
      "all values within limits, check_max_pow_bits should return true"
    );
  }

  #[test]
  fn test_check_pow_bits_initial_sumcheck_exceeds() {
    let mut config = WhirConfig::new::<F>(1 << 10, &default_params());
    config.initial_sumcheck.round_pow = PowConfig::from_difficulty(Bits::new(21.0));
    assert!(
      !config.check_max_pow_bits(Bits::new(20.0)),
      "initial_sumcheck.round_pow exceeds max, check_max_pow_bits should return false"
    );
  }
}
