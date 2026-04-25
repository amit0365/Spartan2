# PR: Fused fold-and-compute for sumcheck

## Motivation

`prove_quad` currently makes **3 passes** over the polynomial data per round:
1. `compute_eval_points_quad(A, B)` — read A, B to compute `(eval_0, eval_2)`
2. `poly_A.bind_poly_var_top(r)` — read+write A to fold
3. `poly_B.bind_poly_var_top(r)` — read+write B to fold

For large polynomials (> L3 cache), each pass evicts and reloads the data. Fusing into **1 pass** saves ~2 full memory scans per round.

Reference: `leanMultisig/crates/backend/sumcheck/src/product_computation.rs` implements `fold_and_compute_product_sumcheck_polynomial` — the same optimization for Binius-style sumcheck.

WHIR's `whir/src/algebra/sumcheck.rs` has a TODO for this: `// TODO: Replace with a single pass implementation.`

## Design

### New function: `fold_and_compute_eval_points_quad`

Location: `src/sumcheck.rs`, private method on `SumcheckProof<E>`.

```rust
/// Fold poly_A and poly_B by r, then compute eval points on the folded result.
/// Single pass: reads each element once, writes folded values back, accumulates (eval_0, eval_2).
fn fold_and_compute_eval_points_quad(
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    r: &E::Scalar,
) -> (E::Scalar, E::Scalar)
```

**Access pattern per index `i`** (4 reads, 2 writes):
```
Before fold: poly has 2n elements = [lo | hi]
After fold:  poly has n elements

We need eval points on the folded n elements, split as [lo' | hi'] where each half has n/2 = quarter.

For each i in 0..quarter:
  Read:  A[i], A[quarter+i], A[n+i], A[n+quarter+i]
  Fold:  a_lo = A[i] + r*(A[n+i] - A[i])           — folded lo half
         a_hi = A[quarter+i] + r*(A[n+quarter+i] - A[quarter+i])  — folded hi half
  Write: A[i] = a_lo, A[quarter+i] = a_hi
  Accum: eval_0 += a_lo * b_lo
         eval_2 += (a_hi - a_lo) * (b_hi - b_lo)
```

Uses `DelayedReduction` accumulators (same as current `compute_eval_points_quad`).

### Modified `prove_quad` loop

```
Round 0:
  (eval_0, eval_2) = compute_eval_points_quad(A, B)      // no prior r
  transcript absorb/squeeze → r_0

Rounds 1..n-1:
  (eval_0, eval_2) = fold_and_compute_eval_points_quad(A, B, &r_prev)  // fused
  transcript absorb/squeeze → r_i

After last round:
  poly_A.bind_poly_var_top(&r_last)    // final fold only
  poly_B.bind_poly_var_top(&r_last)
```

### What doesn't change

- `compute_eval_points_quad` — kept for round 0 (no fold needed)
- `bind_poly_var_top` — kept for final round (no compute needed)
- `SumcheckProof::verify` — untouched (verifier doesn't fold)
- `prove_quad` signature and return type — identical
- All existing tests pass without modification

## Files changed

| File | Change |
|------|--------|
| `src/sumcheck.rs` | Add `fold_and_compute_eval_points_quad`, restructure `prove_quad` loop |

No new files, no new deps, no API changes.

## Testing

1. **Correctness**: existing `test_snark` / sumcheck tests cover `prove_quad` — output must be bit-identical
2. **Add unit test**: verify `fold_and_compute_eval_points_quad(A, B, r)` produces same result as sequential `bind+compute`
3. **Benchmark**: `cargo bench` on `prove_quad` with 2^16, 2^20, 2^24 polynomials — expect ~1.5-2x speedup on large sizes

## Risks

- **Stride-4 access** (`i, quarter+i, n+i, n+quarter+i`) is less cache-friendly than sequential. Hardware prefetchers handle regular strides well, but performance should be validated on the target sizes.
- **Parallelism**: `par_iter` over `0..quarter` with mutable writes to `poly_A[i]` and `poly_A[quarter+i]` — no aliasing since `i < quarter`, so this is safe with `split_at_mut` partitioning. Needs careful implementation to satisfy the borrow checker.
- **`prove_cubic`**: same optimization applies to `compute_eval_points_cubic_with_additive_term` + 4 `bind_poly_var_top` calls. Out of scope for this PR but same pattern.

## Scope

This PR is purely an optimization. No behavioral changes, no new features. Can be reviewed/merged independently of the WHIR PCS work.
