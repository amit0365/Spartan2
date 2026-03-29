//! Binary Merkle tree for hash-based polynomial commitment schemes.
//!
//! Uses a flat array layout where internal nodes are stored above leaves.
//! Generic over the hash function via `crate::hash::Hash`.
// Ported from https://github.com/WizardOfMenlo/whir

use std::mem::swap;

use crate::errors::SpartanError;
use crate::hash::Hash;
use ff::PrimeField;
use sha3::digest::Digest as DigestTrait;

/// A 32-byte hash digest (Keccak256 output size).
pub type Digest = [u8; 32];

/// Binary Merkle tree with a flat array layout.
///
/// Node layout (1-indexed, stored 0-indexed):
/// ```text
///        [1]           <- root
///      /     \
///    [2]     [3]       <- internal
///   /  \    /  \
///  [4] [5] [6] [7]    <- leaves (padded to next power of 2)
/// ```
///
/// nodes[0] is unused (wastes 32 bytes) but simplifies navigation:
/// parent = i/2, children = 2i/2i+1, sibling = i^1 — no offset arithmetic needed.
/// Root is at nodes[1], leaves at nodes[capacity..2*capacity].
pub struct MerkleTree {
  /// All nodes: `[unused, root, ..internal.., ..leaves..]`. Length = `2 * capacity`.
  nodes: Vec<Digest>,
  num_leaves: usize,
  capacity: usize,
}

/// Merkle proof: sibling hashes along the path from leaf to root.
pub struct MerkleProof {
  pub siblings: Vec<Digest>,
  pub leaf_index: usize,
}

/// Batched Merkle proof: sibling hashes for multiple leaves with sibling merging.
///
/// When two queried indices are siblings (`a ^ 1 == b`), neither sibling hash
/// is included — the verifier already has both leaves. Only unpaired nodes
/// contribute hashes, so proof size < `k * log(n)` for k opened leaves.
///
/// `leaf_indices` must be sorted and deduplicated (enforced by `batch_open`).
pub struct BatchMerkleProof {
  /// Sibling hashes for unpaired nodes, in bottom-up traversal order.
  pub sibling_hashes: Vec<Digest>,
  /// Sorted, deduplicated leaf indices that were opened.
  pub leaf_indices: Vec<usize>,
}

impl MerkleTree {
  /// Build a Merkle tree from leaf digests.
  ///
  /// Pads leaves to the next power of 2 with zero digests,
  /// then hashes pairs bottom-up: `H(left || right)`.
  /// Hashes level-by-level bottom-up using `H::hash_many` for parallelism.
  pub fn new<H: Hash>(leaves: Vec<Digest>) -> Self {
    let num_leaves = leaves.len();
    let capacity = num_leaves.next_power_of_two().max(1);
    let mut nodes = vec![[0u8; 32]; 2 * capacity];

    // Copy leaves into position; remaining slots stay as zero-padded.
    nodes[capacity..capacity + num_leaves].copy_from_slice(&leaves);

    // Build each level bottom-up. At level l, nodes span [level_start..level_end).
    // Siblings are interleaved pairs in the child level: child_start..child_end.
    let mut level_size = capacity;
    while level_size > 1 {
      let child_start = level_size; // children live at [level_size..2*level_size]
      let parent_start = level_size / 2; // parents live at [level_size/2..level_size]
      let (upper, current_level) = nodes.split_at_mut(child_start);
      let input: &[u8] = current_level[..level_size].as_flattened();
      H::hash_many(
        input,
        64,
        &mut upper[parent_start..parent_start + level_size / 2],
      );
      level_size /= 2;
    }

    MerkleTree {
      nodes,
      num_leaves,
      capacity,
    }
  }

  /// Returns the root digest.
  pub fn root(&self) -> Digest {
    self.nodes[1]
  }

  /// Open the tree at the given leaf indices. Returns one proof per index.
  ///
  /// Each proof contains the sibling hashes from leaf to root.
  /// Use `batch_open` for multi-index openings with sibling merging.
  pub fn open(&self, indices: &[usize]) -> Vec<MerkleProof> {
    indices
      .iter()
      .map(|&leaf_index| {
        let mut siblings = Vec::new();
        let mut idx = self.capacity + leaf_index;
        while idx > 1 {
          siblings.push(self.nodes[idx ^ 1]); //sibling for this idx
          idx >>= 1; //move up to parent
        }

        MerkleProof {
          siblings,
          leaf_index,
        }
      })
      .collect()
  }

  /// Batch open multiple leaves with sibling merging.
  ///
  /// Sorts and deduplicates indices, then walks the tree bottom-up.
  /// When two queried indices are siblings (`a ^ 1 == b`), no sibling hash
  /// is emitted for either — both are already known by the verifier.
  /// Proof size: `O(k + log n)` in the best case vs `O(k * log n)` naive.
  ///
  /// Used by WHIR's FRI rounds and any hash-based scheme querying many positions.
  pub fn batch_open(&self, indices: &[usize]) -> BatchMerkleProof {
    if indices.is_empty() {
      return BatchMerkleProof {
        sibling_hashes: vec![],
        leaf_indices: vec![],
      };
    }

    let mut indices = indices.to_vec();
    indices.sort_unstable();
    indices.dedup();

    let mut current_indices: Vec<_> = indices.iter().map(|&i| self.capacity + i).collect();
    let mut sibling_hashes = Vec::new();
    while current_indices[0] > 1 {
      let mut iter = current_indices.iter().copied().peekable();
      let mut next_indices = Vec::new();
      loop {
        match (iter.next(), iter.peek()) {
          (Some(cur), Some(&nxt)) if nxt == cur ^ 1 => {
            // cur and nxt are siblings — merge, no hash needed
            next_indices.push(cur >> 1);
            iter.next();
          }
          (Some(cur), _) => {
            // isolated node — need sibling from tree
            sibling_hashes.push(self.nodes[cur ^ 1]);
            next_indices.push(cur >> 1);
          }
          (None, _) => break,
        }
      }

      current_indices = next_indices;
    }

    BatchMerkleProof {
      sibling_hashes,
      leaf_indices: indices,
    }
  }
}

/// Verify a Merkle proof against a claimed root.
///
/// Recomputes the root by hashing the leaf with each sibling bottom-up,
/// then checks equality with the provided root.
pub fn verify_proof<H: Hash>(root: &Digest, leaf: &Digest, proof: &MerkleProof) -> bool {
  let mut current = *leaf;
  let capacity = 1 << proof.siblings.len();
  let mut idx = capacity + proof.leaf_index;
  for sibling in &proof.siblings {
    current = if idx & 1 == 0 {
      let result = H::digest([current, *sibling].concat());
      let mut d = [0u8; 32];
      d.copy_from_slice(&result[..32]);
      d
    } else {
      let result = H::digest([*sibling, current].concat());
      let mut d = [0u8; 32];
      d.copy_from_slice(&result[..32]);
      d
    };

    idx >>= 1;
  }

  current == *root
}

/// Verify a batched Merkle proof against a claimed root.
///
/// Mirrors `batch_open` exactly: when two queried indices are siblings,
/// both hashes are already known — no sibling is consumed from the proof.
/// Otherwise, reads the next sibling from `proof.sibling_hashes`.
///
/// `num_leaves` is used for bounds checking (all indices must be < num_leaves).
/// `indices` and `leaf_hashes` may be unsorted and may contain duplicates
/// (duplicates with inconsistent hashes are rejected).
pub fn batch_verify<H: Hash>(
  root: &Digest,
  num_leaves: usize,
  indices: &[usize],
  leaf_hashes: &[Digest],
  proof: &BatchMerkleProof,
) -> Result<(), SpartanError> {
  // Validate indices.
  if indices.len() != leaf_hashes.len() {
    return Err(SpartanError::InvalidPCS {
      reason: "indices and leaf_hashes length mismatch".into(),
    });
  }
  if !indices.iter().all(|i| *i < num_leaves) {
    return Err(SpartanError::InvalidPCS {
      reason: "leaf index out of bounds".into(),
    });
  }
  if indices.is_empty() {
    return Ok(());
  }

  // Sort indices and leaf hashes.
  let mut layer: Vec<_> = indices
    .iter()
    .copied()
    .zip(leaf_hashes.iter().copied())
    .collect();
  layer.sort_unstable_by_key(|(i, _)| *i);

  // Check duplicate leaf consistency and deduplicate.
  for i in 1..layer.len() {
    if layer[i - 1].0 == layer[i].0 && layer[i - 1].1 != layer[i].1 {
      return Err(SpartanError::InvalidPCS {
        reason: "duplicate index with inconsistent hash".into(),
      });
    }
  }
  layer.dedup_by_key(|(i, _)| *i);

  let capacity = num_leaves.next_power_of_two();
  let depth = capacity.ilog2() as usize;
  let mut current_indices: Vec<_> = layer.iter().map(|(i, _)| *i).collect();
  let mut current_hashes: Vec<_> = layer.iter().map(|(_, h)| *h).collect();
  let mut hint_iter = proof.sibling_hashes.iter();
  let mut next_indices = Vec::with_capacity(layer.len());
  let mut input_hashes: Vec<Digest> = Vec::with_capacity(layer.len() * 2);
  let mut next_hashes = Vec::with_capacity(layer.len());

  for _ in 0..depth {
    next_indices.clear();
    input_hashes.clear();
    next_hashes.clear();

    let mut indices_iter = current_indices.iter().copied().peekable();
    let mut hashes_iter = current_hashes.iter().copied();
    loop {
      match (indices_iter.next(), indices_iter.peek()) {
        (Some(a), Some(&b)) if b == a ^ 1 => {
          input_hashes.push(hashes_iter.next().unwrap());
          input_hashes.push(hashes_iter.next().unwrap());
          next_indices.push(a >> 1);
          indices_iter.next();
        }
        (Some(a), _) => {
          let sibling_hash = hint_iter.next().ok_or(SpartanError::InvalidPCS {
            reason: "insufficient sibling hashes in proof".into(),
          })?;
          if a & 1 == 0 {
            input_hashes.push(hashes_iter.next().unwrap());
            input_hashes.push(*sibling_hash);
          } else {
            input_hashes.push(*sibling_hash);
            input_hashes.push(hashes_iter.next().unwrap());
          }
          next_indices.push(a >> 1);
        }
        (None, _) => break,
      }
    }

    next_hashes.resize(input_hashes.len() / 2, [0u8; 32]);
    H::hash_many(input_hashes.as_flattened(), 64, &mut next_hashes);
    swap(&mut current_indices, &mut next_indices);
    swap(&mut current_hashes, &mut next_hashes);
  }

  if current_indices == [0] && current_hashes[0] == *root {
    Ok(())
  } else {
    Err(SpartanError::InvalidPCS {
      reason: "root mismatch".into(),
    })
  }
}

/// Hash a row of field elements into a single leaf digest.
///
/// Serializes each element via `to_repr()` and feeds all bytes into the hasher.
pub fn hash_row<F: PrimeField, H: Hash>(row: &[F]) -> Digest {
  let mut hasher = H::new();
  for element in row {
    hasher.update_field_element(element);
  }
  let result = DigestTrait::finalize_reset(&mut hasher);
  let mut digest = [0u8; 32];
  digest.copy_from_slice(&result[..32]);
  digest
}

#[cfg(test)]
mod tests {
  use super::*;
  use sha3::Keccak256;

  #[test]
  fn test_hash_row_deterministic() {
    use halo2curves::bn256::Fr;
    let row = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
    let d1 = hash_row::<Fr, Keccak256>(&row);
    let d2 = hash_row::<Fr, Keccak256>(&row);
    assert_eq!(d1, d2);
  }

  #[test]
  fn test_hash_row_different_inputs() {
    use halo2curves::bn256::Fr;
    let row_a = vec![Fr::from(1u64), Fr::from(2u64)];
    let row_b = vec![Fr::from(3u64), Fr::from(4u64)];
    assert_ne!(
      hash_row::<Fr, Keccak256>(&row_a),
      hash_row::<Fr, Keccak256>(&row_b)
    );
  }

  /// Helper: create leaves as [i as u8; 32] for i in 0..n (like WHIR's test pattern).
  fn make_leaves(n: usize) -> Vec<Digest> {
    (0..n).map(|i| [i as u8; 32]).collect()
  }

  #[test]
  fn test_tree_build_and_root() {
    let leaves = make_leaves(8);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    assert_ne!(tree.root(), [0u8; 32]);

    // Same leaves → same root.
    let tree2 = MerkleTree::new::<Keccak256>(leaves);
    assert_eq!(tree.root(), tree2.root());
  }

  #[test]
  fn test_tree_different_leaves_different_root() {
    let tree_a = MerkleTree::new::<Keccak256>(make_leaves(8));
    let mut leaves_b = make_leaves(8);
    leaves_b[3] = [0xFF; 32];
    let tree_b = MerkleTree::new::<Keccak256>(leaves_b);
    assert_ne!(tree_a.root(), tree_b.root());
  }

  #[test]
  fn test_open_and_verify_single() {
    let leaves = make_leaves(256);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();

    for &idx in &[0, 13, 42, 127, 255] {
      let proofs = tree.open(&[idx]);
      assert!(verify_proof::<Keccak256>(&root, &leaves[idx], &proofs[0]));
    }
  }

  #[test]
  fn test_verify_rejects_wrong_leaf() {
    let leaves = make_leaves(16);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();
    let proofs = tree.open(&[5]);

    let wrong_leaf = [0xFF; 32];
    assert!(!verify_proof::<Keccak256>(&root, &wrong_leaf, &proofs[0]));
  }

  #[test]
  fn test_verify_rejects_wrong_root() {
    let leaves = make_leaves(16);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let proofs = tree.open(&[5]);

    let wrong_root = [0xFF; 32];
    assert!(!verify_proof::<Keccak256>(
      &wrong_root,
      &leaves[5],
      &proofs[0]
    ));
  }

  #[test]
  fn test_non_power_of_two_leaves() {
    // 5 leaves → padded to 8 internally.
    let leaves = make_leaves(5);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();

    let proofs = tree.open(&[0, 4]);
    assert!(verify_proof::<Keccak256>(&root, &leaves[0], &proofs[0]));
    assert!(verify_proof::<Keccak256>(&root, &leaves[4], &proofs[1]));
  }

  #[test]
  fn test_batch_open_and_verify() {
    let leaves = make_leaves(256);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();

    let indices = vec![13, 42, 100, 255];
    let leaf_hashes: Vec<Digest> = indices.iter().map(|&i| leaves[i]).collect();
    let proof = tree.batch_open(&indices);

    batch_verify::<Keccak256>(&root, 256, &indices, &leaf_hashes, &proof).unwrap();
  }

  #[test]
  fn test_batch_verify_sibling_merge() {
    // Open two siblings (indices 0 and 1 share parent) — should have fewer hashes than 2 naive paths.
    let leaves = make_leaves(8);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();

    let batch_proof = tree.batch_open(&[0, 1]);
    let naive_proofs = tree.open(&[0, 1]);

    // Batch proof should have strictly fewer sibling hashes than two full paths combined.
    let naive_total: usize = naive_proofs.iter().map(|p| p.siblings.len()).sum();
    assert!(batch_proof.sibling_hashes.len() < naive_total);

    let leaf_hashes = vec![leaves[0], leaves[1]];
    batch_verify::<Keccak256>(&root, 8, &[0, 1], &leaf_hashes, &batch_proof).unwrap();
  }

  #[test]
  fn test_batch_verify_unsorted_and_duplicates() {
    let leaves = make_leaves(16);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();

    // Unsorted, with duplicate index — should still verify.
    let indices = vec![10, 3, 10, 7];
    let leaf_hashes = vec![leaves[10], leaves[3], leaves[10], leaves[7]];
    let proof = tree.batch_open(&[3, 7, 10]);

    batch_verify::<Keccak256>(&root, 16, &indices, &leaf_hashes, &proof).unwrap();
  }

  #[test]
  fn test_batch_verify_rejects_inconsistent_duplicate() {
    let leaves = make_leaves(16);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();

    // Same index, different hash → should fail.
    let indices = vec![5, 5];
    let leaf_hashes = vec![leaves[5], [0xFF; 32]];
    let proof = tree.batch_open(&[5]);

    assert!(batch_verify::<Keccak256>(&root, 16, &indices, &leaf_hashes, &proof).is_err());
  }

  #[test]
  fn test_batch_verify_rejects_wrong_leaf() {
    let leaves = make_leaves(16);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();

    let indices = vec![3, 7];
    let leaf_hashes = vec![leaves[3], [0xFF; 32]]; // wrong hash for index 7
    let proof = tree.batch_open(&[3, 7]);

    assert!(batch_verify::<Keccak256>(&root, 16, &indices, &leaf_hashes, &proof).is_err());
  }

  #[test]
  fn test_batch_empty() {
    let leaves = make_leaves(8);
    let tree = MerkleTree::new::<Keccak256>(leaves);

    let proof = tree.batch_open(&[]);
    assert!(proof.sibling_hashes.is_empty());
    assert!(proof.leaf_indices.is_empty());

    batch_verify::<Keccak256>(&tree.root(), 8, &[], &[], &proof).unwrap();
  }

  #[test]
  fn test_single_leaf_tree() {
    let leaves = make_leaves(1);
    let tree = MerkleTree::new::<Keccak256>(leaves.clone());
    let root = tree.root();

    // Single-leaf tree: root == leaf hash (no siblings).
    let proofs = tree.open(&[0]);
    assert!(proofs[0].siblings.is_empty());
    assert_eq!(root, leaves[0]);
  }
}
