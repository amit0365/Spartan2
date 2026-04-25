# WHIR PCS Native Implementation in Spartan2

## Context

WHIR is a hash-based multilinear polynomial commitment scheme ([eprint 2024/1586](https://eprint.iacr.org/2024/1586)) using Reed-Solomon codes + Merkle trees — no elliptic curves. We implement it natively to avoid `ark-ff`/`ff` incompatibility. Reference impl: `/Users/ak36/Desktop/rust/whir/` (local clone of [WizardOfMenlo/whir](https://github.com/WizardOfMenlo/whir)).

**Why**: Transparent (no trusted setup), hash-based, plausibly post-quantum PCS alternative to Hyrax/IPA.

## Trait Mapping: ark-ff → ff

WHIR uses `ark_ff::FftField`; Spartan2 uses `ff::PrimeField`. The constants map directly:

| ark-ff (WHIR) | ff (Spartan2) | Purpose |
|----------------|---------------|---------|
| `F::TWO_ADICITY` | `F::S` | 2-adicity of the field |
| `F::TWO_ADIC_ROOT_OF_UNITY` | `F::ROOT_OF_UNITY` | Primitive 2^S-th root of unity |
| — | `F::ROOT_OF_UNITY_INV` | Its inverse (for INTT) |
| `F::ZERO`, `F::ONE` | `F::ZERO`, `F::ONE` | Identity elements |
| `field.pow([exp])` | `field.pow([exp])` | Exponentiation |
| `field.inverse()` | `field.invert()` | Field inversion |

Confirmed: halo2curves 0.9.0 implements these via `halo2derive::impl_field!` macro. BN254 Fr has S=28, Pallas Fq has S=32.

## Patterns from Existing PRs

PR #106 (Brakedown) and PR #107 (mKZG) establish the pattern for new PCS implementations:

- PCS file goes in `src/provider/pcs/<name>.rs`
- Commitment struct stores prover-side witness as `Option` + `#[serde(skip)]` fields
- `comm_eval` carries the evaluation value (verifier reads `comm_eval.rows[0]`)
- No-op methods: `blind` = empty, `rerandomize` = clone, `combine` = first
- `Hash` trait in `src/hash.rs` (from Brakedown PR) abstracts Keccak256
- Engine type reuses an existing curve's `GE` (unused by hash-based PCS, just satisfies trait bounds)
- Tests added in `src/spartan.rs::tests` with new engine type

## Key Design Decisions

1. **Native implementation** — no `whir` crate dep
2. **Non-ZK variant first** — ZK can be added later
3. **BN254 scalar field first** — 2-adicity=28, supports FFT up to 2^28
4. **Reuse BN254's `GE` type** — required by Engine trait, unused by WHIR
5. **Skip `FoldingEngineTrait`** — only needed by spartan_zk/neutronnova, not basic spartan
6. **Reuse `Keccak256Transcript`** for Fiat-Shamir
7. **Parameters configurable** — `WhirParams` struct, passed via `WhirPCS` type parameter
8. **Witness stored in Commitment** — `#[serde(skip)]` pattern from Brakedown

## File Structure

```
# Shared PCS infrastructure (reusable by Brakedown, Basefold, Binius, Ligero, etc.)
src/hash.rs                     — Hash trait (already created)
src/provider/pcs/ntt.rs         — NTT engine, transpose, interleaved RS encode (generic over ff::PrimeField)
src/provider/pcs/merkle.rs      — Binary Merkle tree (generic over Hash)
src/provider/pcs/pow.rs         — Proof-of-work grinding (generic over Hash)

# WHIR-specific protocol logic
src/provider/pcs/whir/
  mod.rs                        — pub exports, WhirPCS<E,H> + PCSEngineTrait impl
  config.rs                     — WhirConfig, security parameter computation
  types.rs                      — WhirCommitment, WhirBlind, WhirProof, WhirWitness
  irs.rs                        — Interleaved RS commitment (uses ntt, merkle)
  prover.rs                     — prove()
  verifier.rs                   — verify()
  sumcheck.rs                   — WHIR quadratic sumcheck (uses pow)
  algebra.rs                    — scalar_mul, powers, geometric_accumulate
```

Also modify:
- `src/lib.rs` — already has `mod hash;`
- `src/provider/pcs/mod.rs` — add `pub mod ntt; pub mod merkle; pub mod pow; pub mod whir;`
- `src/provider/mod.rs` — add `Bn254WhirEngine` + `PallasWhirEngine`
- `src/spartan.rs` — add test with WHIR engine

**Note**: `algebra.rs` (dot, fold, compute_sumcheck_polynomial) stays in whir/ for now since the existing `src/polys/` and `src/sumcheck.rs` have their own conventions. Can be promoted to `provider/pcs/` later if other PCS implementations need them.

## Build Order (dependency graph)

```
# Shared PCS infra (src/provider/pcs/ level)
0. src/hash.rs                  (done ✓)
1. src/provider/pcs/ntt.rs      (standalone — port NttEngine using ff::PrimeField::{S, ROOT_OF_UNITY})
2. src/provider/pcs/merkle.rs   (depends on: hash)
3. src/provider/pcs/pow.rs      (depends on: hash)

# WHIR protocol (src/provider/pcs/whir/)
4. algebra.rs        (standalone — scalar_mul, powers, geometric_accumulate) ✓
5. config.rs         (standalone — WhirParams + WhirConfig + security parameter computation)
6. types.rs          (depends on: merkle)
7. sumcheck.rs       (depends on: pow, algebra)
8. irs.rs            (depends on: ntt, merkle, config, types)
9. prover.rs         (depends on: irs, sumcheck, config, types)
10. verifier.rs      (depends on: merkle, sumcheck, config, types)
11. mod.rs           (depends on: all above — PCSEngineTrait impl + Engine registration)
```

## Detailed Component Design

### Step 0: `src/hash.rs` — Hash Trait

PR #106 (Brakedown) adds this but isn't merged. Create it ourselves.

```rust
use ff::PrimeField;
use sha3::digest::{Digest, HashMarker};
pub use sha3::{Keccak256, digest::{FixedOutputReset, Output, Update}};
use std::fmt::Debug;

pub trait Hash: 'static + Sized + Clone + Debug + FixedOutputReset + Default + Update + HashMarker {
  fn new() -> Self { Self::default() }
  fn update_field_element(&mut self, field: &impl PrimeField) {
    Digest::update(self, field.to_repr());
  }
  fn digest(data: impl AsRef<[u8]>) -> Output<Self> {
    let mut hasher = Self::default();
    hasher.update(data.as_ref());
    hasher.finalize()
  }
}
impl<T: 'static + Sized + Clone + Debug + FixedOutputReset + Default + Update + HashMarker> Hash for T {}
```

Reference: PR #106 `src/hash.rs` (30 lines). Add `mod hash;` to `src/lib.rs`.

### Step 1: `ntt.rs` — NTT Engine

Port from `whir/src/algebra/ntt/cooley_tukey.rs`. The WHIR impl uses a sqrt(N) Cooley-Tukey six-step algorithm with cached root tables. We rewrite for `ff::PrimeField` instead of `ark_ff::FftField`.

```rust
use ff::PrimeField;

pub struct NttEngine<F: PrimeField> {
  order: usize,           // 2^S or smaller power-of-2
  omega_order: F,         // primitive root of that order
  roots: RwLock<Vec<F>>,  // cached root table
}

impl<F: PrimeField> NttEngine<F> {
  /// Construct from ff::PrimeField constants.
  /// Maps: F::S -> order = 2^S, F::ROOT_OF_UNITY -> omega_order
  pub fn new_from_primefield() -> Self {
    let order = 1usize << F::S;
    Self::new(order, F::ROOT_OF_UNITY)
  }

  /// Get or create a cached engine (global cache by TypeId, like WHIR does)
  pub fn from_cache() -> Arc<Self>;

  /// Get the N-th root of unity: omega_order^(order/N)
  pub fn root(&self, n: usize) -> F;

  /// Forward NTT (in-place)
  pub fn ntt(&self, values: &mut [F]);

  /// Batch NTT: each chunk of `size` gets NTT'd
  pub fn ntt_batch(&self, values: &mut [F], size: usize);

  /// Inverse NTT (no 1/n scaling): reverse elements [1..], then NTT
  pub fn intt(&self, values: &mut [F]);
  pub fn intt_batch(&self, values: &mut [F], size: usize);
}

/// Transpose a matrix stored row-major: rows×cols → cols×rows
pub fn transpose<F>(data: &mut [F], rows: usize, cols: usize);

/// Interleaved RS encode (top-level entry point, like whir/src/algebra/ntt/mod.rs::ark_ntt)
pub fn interleaved_rs_encode<F: PrimeField>(
  coeffs: &[&[F]], codeword_length: usize, interleaving_depth: usize
) -> Vec<F>;
```

Reference: `whir/src/algebra/ntt/cooley_tukey.rs` (NttEngine, ~400 lines), `whir/src/algebra/ntt/mod.rs` (interleaved_rs_encode), `whir/src/algebra/ntt/transpose.rs`.

**Key difference**: WHIR's `NttEngine::new_from_fftfield()` uses `F::TWO_ADICITY` and `F::TWO_ADIC_ROOT_OF_UNITY`. We use `F::S` and `F::ROOT_OF_UNITY`. Semantics are identical: both define `s` where `p-1 = 2^s * t`, and a primitive 2^s-th root of unity = `generator^t`. `ff` additionally provides `ROOT_OF_UNITY_INV` (free INTT optimization). The `SMALL_SUBGROUP_BASE` / mixed-radix features of ark-ff's `FftField` are not used by WHIR's NTT.

### Step 2: `merkle_tree.rs` — Binary Merkle Tree ✅

**Status: Complete.** File: `src/provider/merkle_tree.rs`. 15 tests passing.

Decoupled port from `whir/src/protocols/merkle_tree.rs`. WHIR's version is tightly coupled to their transcript (`ProverState`/`VerifierState`). We build the tree, then open/verify independently via proof structs. Uses 1-indexed heap layout (`nodes[0]` unused, root at `nodes[1]`, leaves at `nodes[capacity..2*capacity]`).

```rust
pub type Digest = [u8; 32];

pub struct MerkleTree {
  nodes: Vec<Digest>,    // 1-indexed heap: [unused, root, ..internal.., ..leaves..]
  num_leaves: usize,
  capacity: usize,       // num_leaves.next_power_of_two()
}

pub struct MerkleProof { pub siblings: Vec<Digest>, pub leaf_index: usize }
pub struct BatchMerkleProof { pub sibling_hashes: Vec<Digest>, pub leaf_indices: Vec<usize> }

impl MerkleTree {
  pub fn new<H: Hash>(leaves: Vec<Digest>) -> Self;  // H::hash_many for parallel bottom-up
  pub fn root(&self) -> Digest;
  pub fn open(&self, indices: &[usize]) -> Vec<MerkleProof>;           // per-leaf (Brakedown-style)
  pub fn batch_open(&self, indices: &[usize]) -> BatchMerkleProof;     // sibling merging (WHIR-style)
}

pub fn verify_proof<H: Hash>(root: &Digest, leaf: &Digest, proof: &MerkleProof) -> bool;
pub fn batch_verify<H: Hash>(root, num_leaves, indices, leaf_hashes, proof) -> Result<(), SpartanError>;
pub fn hash_row<F: PrimeField, H: Hash>(row: &[F]) -> Digest;
```

Key decisions:
- `batch_open`/`batch_verify` use WHIR's sibling-merge optimization (`a ^ 1 == b` skips sibling hash). General technique (STARK 2018, arkworks `MultiPath`), serves both WHIR and Brakedown.
- `batch_open` uses absolute node indices (natural for heap layout). `batch_verify` uses level-relative indices (matches WHIR). Both produce identical sibling hash order.
- `batch_verify` returns `Result<(), SpartanError>` — no `unwrap()` panics on malicious proofs.
- `verify_proof` stays `-> bool` (no untrusted iteration).

### Step 3: `pow.rs` — Proof-of-Work

Port from `whir/src/protocols/proof_of_work.rs`. Uses threshold-based difficulty: `H(challenge || nonce)` interpreted as u64 must be `<= threshold`. Difficulty d bits means threshold = 2^(64-d).

```rust
use crate::hash::Hash;

pub struct PowConfig {
  pub threshold: u64,  // u64::MAX means no PoW (difficulty 0)
}

impl PowConfig {
  pub fn from_difficulty_bits(bits: f64) -> Self {
    let threshold = (64.0 - bits).exp2().ceil() as u64;
    Self { threshold }
  }

  /// Grind: squeeze challenge from transcript, find nonce where H(challenge || nonce) <= threshold.
  /// Absorb nonce back into transcript.
  pub fn prove<E: Engine>(&self, transcript: &mut E::TE) -> u64;

  /// Verify: squeeze same challenge, check H(challenge || nonce) <= threshold.
  pub fn verify<E: Engine>(&self, transcript: &mut E::TE, nonce: u64) -> Result<(), SpartanError>;
}
```

Reference: `whir/src/protocols/proof_of_work.rs` (prove ~60 lines, verify ~20 lines). The WHIR impl also has parallel grinding via `rayon::broadcast` — include this.

### Step 4: `algebra.rs` — Field Algebra Utilities ✅

**Status: Complete.** File: `src/provider/pcs/whir/algebra.rs`.

Pure field operations used by the WHIR protocol. No `compute_sumcheck_polynomial` or `fold` needed — WHIR sumcheck reuses Spartan's `prove_quad` + `CompressedUniPoly` directly.

```rust
/// Multiply every element of `vector` by `weight` in place.
pub fn scalar_mul<F: Field>(vector: &mut [F], weight: F);

/// Geometric sequence: [1, base, base^2, ..., base^(length-1)].
pub fn powers<F: Field>(base: F, length: usize) -> Vec<F>;

/// Batch geometric series accumulation (parallel via rayon + workload_size threshold).
/// accumulator[i] += sum_j scalars[j] * points[j]^i
pub fn geometric_accumulate<F: Field>(accumulator: &mut [F], scalars: Vec<F>, points: &[F]);
```

Key decisions:
- `dot` and `univariate_evaluate` removed — callers import `inner_product` and `UniPoly` from Spartan directly (no wrapper indirection).
- `geometric_accumulate` uses `workload_size::<F>()` threshold (same pattern as `ntt.rs`) for rayon::join recursion.
- `powers` replaces WHIR's `PowPolynomial::split_evals` for geometric sequences.

### Step 5: `config.rs` — Protocol Configuration

Port from `whir/src/protocols/whir/config.rs`. This computes security-driven parameters: sample counts, PoW budgets, per-round configs based on Johnson bounds and list decoding analysis.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirParams {
  pub security_level: usize,          // e.g. 128
  pub pow_bits: usize,                // PoW budget, e.g. 20
  pub initial_folding_factor: usize,  // e.g. 4 (fold by 2^4 = 16 first round)
  pub folding_factor: usize,          // e.g. 4 (fold by 2^4 = 16 subsequent rounds)
  pub starting_log_inv_rate: usize,   // e.g. 1 (rate = 1/2)
  pub unique_decoding: bool,          // unique vs list decoding
}

impl Default for WhirParams {
  fn default() -> Self {
    Self {
      security_level: 128, pow_bits: 20,
      initial_folding_factor: 4, folding_factor: 4,
      starting_log_inv_rate: 1, unique_decoding: false,
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirConfig {
  pub params: WhirParams,
  pub initial_irs: IrsConfig,
  pub initial_sumcheck: WhirSumcheckConfig,
  pub round_configs: Vec<RoundConfig>,
  pub final_sumcheck: WhirSumcheckConfig,
  pub final_pow: PowConfig,
}

pub struct RoundConfig {
  pub irs: IrsConfig,
  pub sumcheck: WhirSumcheckConfig,
  pub pow: PowConfig,
}

pub struct IrsConfig {
  pub vector_size: usize,
  pub codeword_length: usize,      // vector_size / rate
  pub interleaving_depth: usize,   // 2^folding_factor
  pub in_domain_samples: usize,
  pub out_domain_samples: usize,
}

impl WhirConfig {
  /// Compute all security parameters. Port of whir/src/protocols/whir/config.rs::Config::new().
  pub fn new(num_variables: usize, params: WhirParams) -> Self;
}
```

**Note on parameter passing**: `WhirPCS` is parameterized by a `WhirParamsProvider` trait (or const generics) so different security levels can be selected at compile time. Alternatively, `WhirConfig` is stored as `CommitmentKey` and constructed in `setup()` with default params. For configurability, we add `WhirPCS::setup_with_params(label, n, params)` as an associated function (not part of PCSEngineTrait).

Reference: `whir/src/protocols/whir/config.rs` (Config::new — ~150 lines of security analysis), `whir/src/protocols/irs_commit.rs` (IRS config — sample count computation).

### Step 6: `types.rs` — Core Types

```rust
/// Commitment = Merkle root + OOD evaluations + prover-side witness (skipped in serde)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirCommitment<E: Engine, H: Hash> {
  pub root: merkle::Digest,
  pub ood_evals: Vec<E::Scalar>,
  #[serde(skip)] pub witness: Option<WhirWitness<E::Scalar>>,
  #[serde(skip)] pub _h: PhantomData<fn() -> H>,
}

/// No blinding in non-ZK WHIR
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirBlind;

/// Prover-side witness (RS-encoded data + Merkle tree)
pub struct WhirWitness<F> {
  pub encoded_data: Vec<F>,   // flat interleaved RS codewords
  pub merkle_tree: MerkleTree,
}

/// Evaluation proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirProof<E: Engine> {
  pub round_data: Vec<WhirRoundProof<E>>,
  pub final_polynomial: Vec<E::Scalar>,
  pub final_sumcheck: Vec<(E::Scalar, E::Scalar)>,
  pub final_pow_nonce: u64,
}

pub struct WhirRoundProof<E: Engine> {
  pub commitment: (merkle::Digest, Vec<E::Scalar>),  // (root, ood_evals)
  pub sumcheck_polys: Vec<(E::Scalar, E::Scalar)>,   // (c0, c2) per round
  pub pow_nonce: u64,
  pub merkle_proofs: Vec<MerkleProof>,
  pub opened_rows: Vec<Vec<E::Scalar>>,
}
```

### Step 7: `sumcheck.rs` — WHIR Quadratic Sumcheck

Port from `whir/src/protocols/sumcheck.rs`. Uses `algebra::compute_sumcheck_polynomial` and `algebra::fold`.

```rust
pub struct WhirSumcheckConfig {
  pub initial_size: usize,
  pub num_rounds: usize,
  pub round_pow: PowConfig,
}

impl WhirSumcheckConfig {
  /// Prove: <a, b> = sum. Folds a and b in-place. Returns folding randomness.
  /// Per round: send (c0, c2), PoW, receive challenge r, fold.
  pub fn prove<E: Engine>(
    &self, transcript: &mut E::TE,
    a: &mut Vec<E::Scalar>, b: &mut Vec<E::Scalar>, sum: &mut E::Scalar,
  ) -> Result<Vec<E::Scalar>, SpartanError>;

  /// Verify: check (c0, c2) per round, PoW, derive challenges. Returns randomness.
  pub fn verify<E: Engine>(
    &self, transcript: &mut E::TE, sum: &mut E::Scalar,
    proof: &[(E::Scalar, E::Scalar)], pow_nonces: &[u64],
  ) -> Result<Vec<E::Scalar>, SpartanError>;
}
```

Reference: `whir/src/protocols/sumcheck.rs` (Config::prove, Config::verify — 160 lines total).

### Step 8: `irs.rs` — Interleaved Reed-Solomon Commitment

Port from `whir/src/protocols/irs_commit.rs`. Uses NTT for encoding, Merkle tree for commitment, OOD evaluations for list decoding soundness.

```rust
/// IRS commit: RS encode the polynomial, build Merkle tree, compute OOD evals.
pub fn irs_commit<E: Engine, H: Hash>(
  config: &IrsConfig, polynomial: &[E::Scalar], transcript: &mut E::TE,
) -> Result<(WhirCommitment<E, H>, WhirWitness<E::Scalar>), SpartanError>;
```

Algorithm (matching `whir/src/protocols/irs_commit.rs::Config::commit`):
1. Reshape polynomial into `interleaving_depth` blocks of size `vector_size / interleaving_depth`
2. Zero-pad each block to `codeword_length`
3. `ntt_batch` all blocks (coeffs → evals over multiplicative subgroup)
4. Transpose to row-major: `codeword_length` rows × `interleaving_depth` columns
5. Hash each row → Merkle leaf digests
6. Build Merkle tree, absorb root into transcript
7. Squeeze `out_domain_samples` OOD challenge points from transcript
8. Evaluate polynomial at OOD points using Horner's method
9. Absorb OOD evaluations into transcript

**Evaluation domain**: The NTT evaluates at `{ω^0, ω^1, ..., ω^{N-1}}` where `ω = NttEngine::root(codeword_length)`. When verifying in-domain queries at index `j`, the evaluation point is `ω^j`.

```rust
/// Open at in-domain challenge indices.
pub fn irs_open<F: PrimeField>(witness: &WhirWitness<F>, indices: &[usize]) -> Vec<(Vec<F>, MerkleProof)>;

/// Evaluate polynomial at out-of-domain points via Horner's method.
pub fn evaluate_ood<F: PrimeField>(poly: &[F], points: &[F]) -> Vec<F>;
```

Reference: `whir/src/protocols/irs_commit.rs` (Config::commit, open, verify — ~500 lines).

### Step 9: `prover.rs` — WHIR Prove

Port from `whir/src/protocols/whir/prover.rs`.

```rust
pub fn whir_prove<E: Engine, H: Hash>(
  config: &WhirConfig, transcript: &mut E::TE,
  polynomial: &[E::Scalar], commitment: &WhirCommitment<E, H>,
  point: &[E::Scalar], eval: &E::Scalar,
) -> Result<WhirProof<E>, SpartanError>;
```

### Step 10: `verifier.rs` — WHIR Verify

Port from `whir/src/protocols/whir/verifier.rs`.

```rust
pub fn whir_verify<E: Engine, H: Hash>(
  config: &WhirConfig, transcript: &mut E::TE,
  commitment: &WhirCommitment<E, H>,
  point: &[E::Scalar], eval: &E::Scalar,
  proof: &WhirProof<E>,
) -> Result<(), SpartanError>;
```

### Step 11: `mod.rs` — PCSEngineTrait + Engine Registration

```rust
pub struct WhirPCS<E: Engine, H: Hash>(PhantomData<(E, H)>);

impl<E, H> PCSEngineTrait<E> for WhirPCS<E, H>
where E: Engine, E::GE: DlogGroupExt, H: Hash + Send + Sync
{
  type CommitmentKey = WhirConfig;
  type VerifierKey = WhirConfig;
  type Commitment = WhirCommitment<E, H>;
  type Blind = WhirBlind;
  type EvaluationArgument = WhirProof<E>;

  fn setup(label, n, _width) {
    let num_vars = n.next_power_of_two().trailing_zeros() as usize;
    let config = WhirConfig::new(num_vars, WhirParams::default());
    (config.clone(), config)
  }

  fn commit(ck, v, _r, _is_small) {
    // Create a temporary transcript for the commit phase
    let mut transcript = E::TE::new(b"whir_commit");
    let (comm, _witness) = irs_commit::<E, H>(&ck.initial_irs, v, &mut transcript)?;
    Ok(comm) // witness stored inside comm.witness via #[serde(skip)]
  }

  fn prove(ck, _ck_eval, transcript, comm, poly, _blind, point, comm_eval, _blind_eval) {
    // Recompute eval from polynomial (Brakedown pattern: prover has full poly)
    let eval = {
      let chis = EqPolynomial::evals_from_points(point);
      dot(&poly, &chis)
    };
    // Re-derive witness if not cached in commitment
    let comm_with_witness = if comm.witness.is_some() {
      Cow::Borrowed(comm)
    } else {
      let mut t = E::TE::new(b"whir_commit");
      let (c, _) = irs_commit::<E, H>(&ck.initial_irs, poly, &mut t)?;
      Cow::Owned(c)
    };
    whir_prove::<E, H>(ck, transcript, poly, &comm_with_witness, point, &eval)
  }

  fn verify(vk, _ck_eval, transcript, comm, point, comm_eval, arg) {
    // Extract eval from comm_eval (Brakedown pattern)
    // comm_eval.ood_evals[0] carries the claimed evaluation
    let eval = comm_eval.ood_evals.get(0).ok_or(SpartanError::InvalidPCS {
      reason: "Missing eval in comm_eval".into(),
    })?;
    whir_verify::<E, H>(vk, transcript, comm, point, eval, arg)
  }

  // No-ops (following Brakedown pattern):
  fn blind(_ck, _n) -> WhirBlind { WhirBlind }
  fn check_commitment(..) -> Ok(())
  fn rerandomize_commitment(.., comm, ..) -> Ok(comm.clone())
  fn combine_commitments(comms) -> Ok(comms[0].clone())
  fn combine_blinds(..) -> Ok(WhirBlind)
}

// In src/provider/mod.rs:
pub struct Bn254WhirEngine;
impl Engine for Bn254WhirEngine {
  type Base = bn254_types::Base;
  type Scalar = bn254_types::Scalar;
  type GE = bn254_types::Point;       // unused by WHIR, satisfies trait
  type TE = Keccak256Transcript<Self>;
  type PCS = WhirPCS<Self, Keccak256>;
}

pub struct PallasWhirEngine;
impl Engine for PallasWhirEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type TE = Keccak256Transcript<Self>;
  type PCS = WhirPCS<Self, Keccak256>;
}
```

## Testing Strategy

1. **Unit tests** (in each file):
   - `ntt`: `INTT(NTT(x)) == x`, known small examples, batch NTT
   - `merkle`: build + open + verify roundtrip, bad proof rejection
   - `pow`: prove + verify roundtrip, difficulty 0 = no-op
   - `algebra`: compute_sumcheck_polynomial matches naive, fold correctness
   - `sumcheck`: honest prover verifies, bad sum rejected
   - `irs`: commit + open + verify Merkle proofs, OOD eval correctness

2. **PCS integration test** (in `mod.rs`):
   - Full commit + prove + verify roundtrip for random multilinear polynomials
   - Multiple sizes (2^8, 2^12, 2^16)

3. **Spartan SNARK test** (in `src/spartan.rs::tests`):
   - Add `Bn254WhirEngine` to existing `test_snark` test

## Reference Files (WHIR repo → Spartan2 port)

| WHIR source | Spartan2 target | Notes |
|-------------|-----------------|-------|
| `whir/src/algebra/ntt/cooley_tukey.rs` | `whir/ntt.rs` | Replace `FftField` with `PrimeField` |
| `whir/src/algebra/ntt/mod.rs` | `whir/ntt.rs` | `interleaved_rs_encode` |
| `whir/src/algebra/ntt/transpose.rs` | `whir/ntt.rs` | Transpose utility |
| `whir/src/algebra/sumcheck.rs` | `whir/algebra.rs` | `compute_sumcheck_polynomial`, `fold` |
| `whir/src/protocols/merkle_tree.rs` | `whir/merkle.rs` | Decouple from spongefish transcript |
| `whir/src/protocols/proof_of_work.rs` | `whir/pow.rs` | Use Keccak256 instead of engine registry |
| `whir/src/protocols/sumcheck.rs` | `whir/sumcheck.rs` | Use Spartan2 transcript |
| `whir/src/protocols/irs_commit.rs` | `whir/irs.rs` | Core IRS protocol |
| `whir/src/protocols/whir/config.rs` | `whir/config.rs` | Security parameter computation |
| `whir/src/protocols/whir/prover.rs` | `whir/prover.rs` | Main prove algorithm |
| `whir/src/protocols/whir/verifier.rs` | `whir/verifier.rs` | Main verify algorithm |

## Critical Spartan2 Files to Reference

- [src/traits/pcs.rs](src/traits/pcs.rs) — PCSEngineTrait (L31-L141)
- [src/traits/mod.rs](src/traits/mod.rs) — Engine trait (L37-L60)
- [src/provider/pcs/hyrax_pc.rs](src/provider/pcs/hyrax_pc.rs) — reference PCS impl
- [src/provider/mod.rs](src/provider/mod.rs) — engine registration
- [src/provider/keccak.rs](src/provider/keccak.rs) — transcript
- [src/polys/eq.rs](src/polys/eq.rs) — EqPolynomial (reuse for MLE eval)
- [src/errors.rs](src/errors.rs) — SpartanError
- PR #106 `src/provider/pcs/brakedown.rs` — closest reference (hash-based PCS pattern)


## Github workflow
cargo build --examples --benches --verbose
cargo build --no-default-features --target wasm32-unknown-unknown
cargo test --release --verbose
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
typos
