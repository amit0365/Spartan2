---
name: WHIR PCS Implementation
description: Full context for implementing WHIR polynomial commitment scheme natively in Spartan2. Covers architecture decisions, trait mappings, file layout, and current progress.
type: project
---

## Goal
Implement the WHIR polynomial commitment scheme (eprint 2024/1586) natively in Spartan2. WHIR is a hash-based, transparent (no trusted setup), plausibly post-quantum multilinear PCS using Reed-Solomon codes + Merkle trees.

**Why native**: The reference implementation (local at `/Users/ak36/Desktop/rust/whir/`, GitHub: WizardOfMenlo/whir) uses `ark-ff` which is incompatible with Spartan2's `ff`/`halo2curves` field ecosystem. Bridging traits would be painful, so we port the protocol.

## Architecture Decisions

1. **Non-ZK variant first** -- ZK variant adds blinding polynomials, can be layered on later
2. **BN254 scalar field first** -- 2-adicity=28, supports FFT up to 2^28. Pallas (S=32) also viable.
3. **Reuse BN254's `GE` type** in Engine -- required by `Engine` trait but unused by WHIR (hash-based, no curves)
4. **Skip `FoldingEngineTrait`** initially -- only needed by `spartan_zk.rs`/`neutronnova_zk.rs`, not basic `spartan.rs`
5. **Use `Keccak256Transcript`** for Fiat-Shamir (already in codebase)
6. **Witness stored in Commitment** via `Option` + `#[serde(skip)]` (Brakedown PR #106 pattern)
7. **Eval passing**: prover recomputes eval via `dot(poly, EqPolynomial::evals(point))`; verifier extracts from `comm_eval`
8. **Parameters configurable** via `WhirParams` struct with sensible defaults

## Trait Mapping: ark-ff -> ff

| ark-ff (WHIR) | ff (Spartan2) | Same semantics? |
|----------------|---------------|-----------------|
| `FftField` | `PrimeField` | Yes for NTT use |
| `F::TWO_ADICITY` | `F::S` | Yes: s where p-1 = 2^s * t |
| `F::TWO_ADIC_ROOT_OF_UNITY` | `F::ROOT_OF_UNITY` | Yes: primitive 2^s-th root |
| `F::ZERO`, `F::ONE` | `F::ZERO`, `F::ONE` | Identical |
| `field.pow([exp])` | `field.pow([exp])` | Identical |
| `field.inverse()` | `field.invert()` | Same (returns CtOption in ff) |
| `field.square_in_place()` | `field.square()` | ff returns new value, doesn't mutate |
| -- | `F::ROOT_OF_UNITY_INV` | ff bonus: precomputed inverse for INTT |

`SMALL_SUBGROUP_BASE` / mixed-radix features of ark-ff's FftField are NOT used by WHIR's NTT.

## File Layout

```
# Shared PCS infrastructure (src/provider/pcs/ level, alongside ipa.rs)
src/hash.rs                     -- Hash trait (DONE)
src/provider/pcs/ntt.rs         -- NTT engine, transpose, interleaved RS encode
src/provider/pcs/merkle.rs      -- Binary Merkle tree (generic over Hash)
src/provider/pcs/pow.rs         -- Proof-of-work grinding (generic over Hash)

# WHIR-specific protocol
src/provider/pcs/whir/
  mod.rs        -- WhirPCS<E,H> + PCSEngineTrait impl + Engine registration
  config.rs     -- WhirConfig, WhirParams, security parameter computation
  types.rs      -- WhirCommitment, WhirBlind, WhirProof, WhirWitness
  irs.rs        -- Interleaved RS commitment (uses ntt, merkle)
  prover.rs     -- prove()
  verifier.rs   -- verify()
  sumcheck.rs   -- WHIR quadratic sumcheck (uses pow, algebra)
  algebra.rs    -- compute_sumcheck_polynomial, fold, dot
```

## Build Order

```
0. src/hash.rs                  (DONE)
1. src/provider/pcs/ntt.rs      (IN PROGRESS -- user is porting from whir repo)
2. src/provider/pcs/merkle.rs
3. src/provider/pcs/pow.rs
4. whir/algebra.rs
5. whir/config.rs
6. whir/types.rs
7. whir/sumcheck.rs
8. whir/irs.rs
9. whir/prover.rs
10. whir/verifier.rs
11. whir/mod.rs (PCSEngineTrait glue + Engine types)
```

## Implementation Approach

User is **copying from the local WHIR repo** (`/Users/ak36/Desktop/rust/whir/`) and adapting. Mechanical changes:
- `ark_ff::FftField` -> `ff::PrimeField`
- `ark_ff::Field` -> `ff::Field`
- `F::TWO_ADICITY` -> `F::S`
- `F::TWO_ADIC_ROOT_OF_UNITY` -> `F::ROOT_OF_UNITY`
- `#[cfg(feature = "parallel")]` -> use rayon directly (Spartan2 always has rayon)
- `crate::utils::workload_size` -> define local const
- Remove `ark_std`, `static_assertions`, `zerocopy`, `spongefish` deps
- Replace WHIR's transcript (`ProverState`/`VerifierState`) with Spartan2's `Keccak256Transcript`

## Key WHIR Source Files -> Spartan2 Targets

| WHIR source | Target | Notes |
|-------------|--------|-------|
| `whir/src/algebra/ntt/cooley_tukey.rs` | `pcs/ntt.rs` | NttEngine, sqrt(N) Cooley-Tukey |
| `whir/src/algebra/ntt/mod.rs` | `pcs/ntt.rs` | interleaved_rs_encode |
| `whir/src/algebra/ntt/transpose.rs` | `pcs/ntt.rs` | Matrix transpose |
| `whir/src/algebra/ntt/utils.rs` | `pcs/ntt.rs` | sqrt_factor, lcm, gcd |
| `whir/src/algebra/ntt/matrix.rs` | `pcs/ntt.rs` | MatrixMut helper |
| `whir/src/algebra/sumcheck.rs` | `whir/algebra.rs` | compute_sumcheck_polynomial, fold |
| `whir/src/protocols/merkle_tree.rs` | `pcs/merkle.rs` | Decouple from spongefish |
| `whir/src/protocols/proof_of_work.rs` | `pcs/pow.rs` | Threshold-based grinding |
| `whir/src/protocols/sumcheck.rs` | `whir/sumcheck.rs` | Quadratic sumcheck + PoW |
| `whir/src/protocols/irs_commit.rs` | `whir/irs.rs` | IRS encoding pipeline |
| `whir/src/protocols/whir/config.rs` | `whir/config.rs` | Security parameter computation |
| `whir/src/protocols/whir/prover.rs` | `whir/prover.rs` | Multi-round prove |
| `whir/src/protocols/whir/verifier.rs` | `whir/verifier.rs` | Multi-round verify |

## PCSEngineTrait Mapping

Following Brakedown PR #106 patterns:
- `CommitmentKey` = `WhirConfig`
- `VerifierKey` = `WhirConfig`
- `Commitment` = `WhirCommitment<E, H>` (root + OOD evals + optional witness)
- `Blind` = `WhirBlind` (unit-like, no blinding)
- `EvaluationArgument` = `WhirProof<E>`
- No-ops: `blind()`, `rerandomize_commitment()`, `check_commitment()`, `combine_commitments()`, `combine_blinds()`

## Existing PRs to Reference

- **PR #106**: Brakedown PCS -- closest pattern for hash-based PCS integration (Merkle trees, no-op blinds, witness in commitment)
- **PR #107**: mKZG PCS -- another PCS integration example

## Current Progress

- [x] `src/hash.rs` created with Hash trait + blanket impl + imports
- [x] `src/lib.rs` updated with `mod hash;`
- [x] `src/provider/mod.rs` updated with `mod ntt;` (user added)
- [ ] `src/provider/ntt.rs` -- user is actively porting NttEngine from whir repo
- [ ] Everything else

## Full Plan File

Detailed plan with all type signatures, algorithms, and design rationale is at:
`/Users/ak36/.claude/plans/federated-stirring-coral.md`
