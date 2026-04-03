//! Protocol for grinding and verifying proof of work.

use crate::{
  errors::SpartanError,
  hash::Hash,
  provider::bits::Bits,
  traits::{Engine, transcript::TranscriptEngineTrait},
};

use ff::PrimeField;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Config {
  pub threshold: u64,
  pub batch_size: usize,
}

pub fn threshold(difficulty: Bits) -> u64 {
  assert!((0.0..=60.0).contains(&difficulty.into()));

  let threshold = (64.0 - f64::from(difficulty)).exp2().ceil();
  #[allow(clippy::cast_sign_loss)]
  if threshold >= u64::MAX as f64 {
    u64::MAX
  } else {
    threshold as u64
  }
}

pub fn difficulty(threshold: u64) -> Bits {
  Bits::from(64.0 - (threshold as f64).log2())
}

impl Config {
  pub const fn none() -> Self {
    Self {
      threshold: u64::MAX,
      batch_size: 64,
    }
  }

  /// Creates a new configuration from a difficulty.
  pub fn from_difficulty(difficulty: Bits) -> Self {
    Self {
      threshold: threshold(difficulty),
      batch_size: 64,
    }
  }

  pub fn difficulty(&self) -> Bits {
    difficulty(self.threshold)
  }

  pub fn prove<E: Engine, H: Hash>(&self, transcript: &mut E::TE) -> Result<u64, SpartanError> {
    if self.threshold == u64::MAX {
      // If the difficulty is zero, do nothing (also produce no transcript)
      return Ok(u64::MAX);
    }

    let challenge = Self::squeeze_challenge::<E>(transcript)?;
    let nonce = self.grind::<H>(&challenge);
    Self::absorb_nonce::<E>(transcript, nonce);
    Ok(nonce)
  }

  pub fn verify<E: Engine, H: Hash>(
    &self,
    transcript: &mut E::TE,
    nonce: u64,
  ) -> Result<(), SpartanError> {
    if self.threshold == u64::MAX {
      return Ok(());
    }

    let challenge = Self::squeeze_challenge::<E>(transcript)?;
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&challenge);
    input[32..40].copy_from_slice(&nonce.to_le_bytes());

    let hash = H::digest(input);
    let value = u64::from_le_bytes(hash[..8].try_into().unwrap());
    if value > self.threshold {
        return Err(SpartanError::ProofVerifyError {
            reason: "proof-of-work verification failed".into(),
        });
    }
    
    Ok(())
  }

  /// Parallel grinding: find the minimum nonce satisfying the threshold.
  fn grind<H: Hash>(&self, challenge: &[u8; 32]) -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    let batch_size = self.batch_size;
    // Split the work across all available threads.
    // Use atomics to find the unique deterministic lowest satisfying nonce.
    let global_min = AtomicU64::new(u64::MAX);
    rayon::broadcast(|ctx| {
      let thread_nonces =
        ((batch_size * ctx.index()) as u64..).step_by(batch_size * ctx.num_threads());
      let mut inputs = vec![[0u8; 64]; batch_size];
      for input in &mut inputs {
        input[..32].copy_from_slice(challenge);
      }
      let mut outputs = vec![[0u8; 32]; batch_size];
      for batch_start in thread_nonces {
        // Stop work if another thread already found a lower valid nonce.
        if batch_start >= global_min.load(Ordering::Relaxed) {
          break;
        }
        let input_len = inputs.len();
        for (input, nonce) in inputs.iter_mut().zip(batch_start..).take(input_len) {
          input[32..40].copy_from_slice(&nonce.to_le_bytes());
        }
        H::hash_many(64, inputs.as_flattened(), &mut outputs);
        let output_len = outputs.len();
        for (output, nonce) in outputs.iter().zip(batch_start..).take(output_len) {
          let value = u64::from_le_bytes(output[..8].try_into().unwrap());
          if value <= self.threshold {
            // We found a solution, store it in the global_min.
            // Use fetch_min to solve race condition with simultaneous solutions.
            global_min.fetch_min(nonce, Ordering::SeqCst);
            break;
          }
        }
      }
    });

    // Return the best found nonce, or fallback check on `u64::MAX`.
    let nonce = global_min.load(Ordering::SeqCst);
    assert!(nonce != u64::MAX, "Proof of Work failed to solve.");
    nonce
  }

  /// Squeeze a scalar from the transcript and return its 32-byte representation.
  fn squeeze_challenge<E: Engine>(transcript: &mut E::TE) -> Result<[u8; 32], SpartanError> {
    let scalar = transcript.squeeze(b"pow_challenge")?;
    let repr = scalar.to_repr();
    let bytes: &[u8] = repr.as_ref();
    bytes.try_into().map_err(|_| SpartanError::InternalError {
      reason: "scalar repr is not 32 bytes".into(),
    })
  }

  /// Absorb the nonce back into the transcript as a scalar.
  fn absorb_nonce<E: Engine>(transcript: &mut E::TE, nonce: u64) {
    let scalar = E::Scalar::from(nonce);
    transcript.absorb(b"pow_nonce", &scalar);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    hash::Keccak256,
    provider::{PallasHyraxEngine, keccak::Keccak256Transcript},
    traits::transcript::TranscriptEngineTrait,
  };
  use proptest::{proptest, strategy::Strategy};

  type E = PallasHyraxEngine;
  type H = Keccak256;

  fn make_transcript() -> Keccak256Transcript<E> {
    Keccak256Transcript::<E>::new(b"pow_test")
  }

  impl Config {
    pub fn arbitrary() -> impl Strategy<Value = Self> {
      // threshold in range [u64::MAX >> 6 ..] ensures low difficulty (fast tests)
      ((u64::MAX >> 6)..).prop_map(|threshold| Self {
        threshold,
        batch_size: 64,
      })
    }
  }

  fn test_config(config: &Config) {
    // Prover
    let mut prover_t = make_transcript();
    let nonce = config.prove::<E, H>(&mut prover_t).unwrap();

    // Verifier
    let mut verifier_t = make_transcript();
    config.verify::<E, H>(&mut verifier_t, nonce).unwrap();
  }

  #[test]
  fn test_pow() {
    proptest!(|(config in Config::arbitrary())| {
        test_config(&config);
    });
  }

  #[test]
  fn test_pow_none_is_noop() {
    let config = Config::none();
    let mut t = make_transcript();
    let nonce = config.prove::<E, H>(&mut t).unwrap();
    assert_eq!(nonce, u64::MAX);

    let mut t2 = make_transcript();
    config.verify::<E, H>(&mut t2, u64::MAX).unwrap();
  }

  #[test]
  fn test_threshold_integer() {
    assert_eq!(threshold(Bits::new(0.0)), u64::MAX);
    assert_eq!(threshold(Bits::new(60.0)), 1 << 4);
    proptest!(|(bits in 1_u64..=60)| {
        assert_eq!(threshold(Bits::new(bits as f64)), 1 << (64 - bits));
    });
  }

  #[test]
  fn test_threshold_fractional() {
    proptest!(|(bits in 0.0..=60.0)| {
        let t = threshold(Bits::new(bits));
        let min = threshold(Bits::new(bits.ceil()));
        let max = threshold(Bits::new(bits.floor()));
        assert!((min..=max).contains(&t));
    });
  }

  #[test]
  fn test_threshold_monotonic() {
    proptest!(|(bits in 0.0..=59.0, delta in 0.0..=1.0)| {
        let low = threshold(Bits::new(bits + delta));
        let high = threshold(Bits::new(bits));
        assert!(low <= high);
    });
  }

  #[test]
  fn test_difficulty_integer() {
    assert_eq!(difficulty(u64::MAX), Bits::new(0.0));
    assert_eq!(difficulty(1 << 4), Bits::new(60.0));
    proptest!(|(bits in 1_u64..=60)| {
        assert_eq!(difficulty(1 << (64 - bits)), Bits::new(bits as f64));
    });
  }

  #[test]
  fn test_difficulty_fractional() {
    proptest!(|(threshold in 16_u64..)| {
        let d = difficulty(threshold);
        let min = difficulty(threshold.checked_next_power_of_two().unwrap_or(u64::MAX));
        let max = Bits::new(f64::from(min) + 1.0);
        assert!((min..=max).contains(&d));
    });
  }

  #[test]
  fn test_difficulty_monotonic() {
    proptest!(|(threshold in 16_u64.., delta: u64)| {
        let high = difficulty(threshold);
        let low = difficulty(threshold.saturating_add(delta));
        assert!(low <= high);
    });
  }
}
