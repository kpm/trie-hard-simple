// Copyright 2024 Cloudflare, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg_attr(not(doctest), doc = include_str!("../README.md"))]
#![deny(
    missing_docs,
    missing_debug_implementations,
    unreachable_pub,
    rustdoc::broken_intra_doc_links,
    unsafe_code
)]
#![warn(rust_2018_idioms)]

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    ops::RangeFrom,
};

/// Structure representing masks for each byte using `u8`.
#[derive(Debug, Clone)]
#[repr(transparent)]
struct MasksByByte([u8; 256]);

impl Default for MasksByByte {
    fn default() -> Self {
        Self([0u8; 256])
    }
}

impl MasksByByte {
    /// Creates a new `MasksByByte` based on the used bytes.
    fn new(used_bytes: BTreeSet<u8>) -> Self {
        let mut mask = 1u8;
        let mut byte_masks = [0u8; 256];

        for c in used_bytes.into_iter() {
            byte_masks[c as usize] = mask;
            // Prevent overflow by resetting mask if it exceeds 8 bits
            mask = mask.checked_shl(1).unwrap_or(1);
        }

        Self(byte_masks)
    }
}

/// Inner representation of a trie-hard trie using `u8` for storage.
#[derive(Debug, Clone, Default)]
pub struct TrieHardSized<'a, T> {
    masks: MasksByByte,
    nodes: Vec<TrieState<'a, T>>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct StateSpec<'a> {
    prefix: &'a [u8],
    index: usize,
}

#[derive(Debug, Clone)]
struct SearchNode {
    mask: u8,
    edge_start: usize,
}

#[derive(Debug, Clone)]
enum TrieState<'a, T> {
    Leaf(&'a [u8], T),
    Search(SearchNode),
    SearchOrLeaf(&'a [u8], T, SearchNode),
}

impl<'a, T> TrieHardSized<'a, T> {
    /// Constructs a new `TrieHardSized` with given masks and values.
    fn new(masks: MasksByByte, values: Vec<(&'a [u8], T)>) -> Self {
        let sorted = values
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect::<BTreeMap<_, _>>();

        let mut nodes = Vec::new();
        let mut next_index = 1;

        let root_state_spec = StateSpec {
            prefix: &[],
            index: 0,
        };

        let mut spec_queue = VecDeque::new();
        spec_queue.push_back(root_state_spec);

        while let Some(spec) = spec_queue.pop_front() {
            debug_assert_eq!(spec.index, nodes.len());
            let (state, next_specs) = TrieState::new(
                spec,
                next_index,
                &masks.0,
                &sorted,
            );

            next_index += next_specs.len();
            spec_queue.extend(next_specs);
            nodes.push(state);
        }

        TrieHardSized { nodes, masks }
    }

    /// Retrieves the value associated with the given key.
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> Option<T> {
        self.get_from_bytes(key.as_ref())
    }

    /// Retrieves the value associated with the given byte slice key.
    pub fn get_from_bytes(&self, key: &[u8]) -> Option<T> {
        let mut state = self.nodes.get(0)?;

        for (i, &c) in key.iter().enumerate() {
            let next_state_opt = match state {
                TrieState::Leaf(k, value) => {
                    return (k.len() == key.len() && k[i..] == key[i..])
                        .then_some(*value)
                }
                TrieState::Search(search)
                | TrieState::SearchOrLeaf(_, _, search) => {
                    search.evaluate(c, self)
                }
            };

            if let Some(next_state_index) = next_state_opt {
                state = &self.nodes[next_state_index];
            } else {
                return None;
            }
        }

        match state {
            TrieState::Leaf(k, value)
            | TrieState::SearchOrLeaf(k, value, _) => {
                (k.len() == key.len()).then_some(*value)
            }
            _ => None,
        }
    }

    /// Creates an iterator over the entire trie in key order.
    pub fn iter(&self) -> TrieIter<'a, T> {
        TrieIter {
            iter: self.iter_inner(),
        }
    }

    /// Creates an iterator over the trie starting with the given prefix.
    pub fn prefix_search<K: AsRef<[u8]>>(&self, prefix: K) -> TrieIter<'a, T> {
        TrieIter {
            iter: self.prefix_search_inner(prefix.as_ref()),
        }
    }

    /// Internal method to initialize iteration.
    fn iter_inner(&self) -> TrieIterSized<'a, T> {
        TrieIterSized {
            stack: vec![TrieNodeIter::default()],
            trie: self,
        }
    }

    /// Internal method to initialize prefix search iteration.
    fn prefix_search_inner(&self, prefix: &[u8]) -> TrieIterSized<'a, T> {
        let mut node_index = 0;
        let Some(mut state) = self.nodes.get(node_index) else {
            return TrieIterSized::empty(self);
        };

        for (i, &c) in prefix.iter().enumerate() {
            let next_state_opt = match state {
                TrieState::Leaf(k, _) => {
                    if k.len() == prefix.len() && k[..] == prefix[..] {
                        return TrieIterSized::new(self, node_index);
                    } else {
                        return TrieIterSized::empty(self);
                    }
                }
                TrieState::Search(search)
                | TrieState::SearchOrLeaf(_, _, search) => {
                    search.evaluate(c, self)
                }
            };

            if let Some(next_state_index) = next_state_opt {
                node_index = next_state_index;
                state = &self.nodes[next_state_index];
            } else {
                return TrieIterSized::empty(self);
            }
        }

        TrieIterSized::new(self, node_index)
    }
}

impl<'a, T> TrieState<'a, T> {
    /// Constructs a new `TrieState` and returns any subsequent state specifications.
    fn new(
        spec: StateSpec<'a>,
        edge_start: usize,
        byte_masks: &[u8; 256],
        sorted: &BTreeMap<&'a [u8], T>,
    ) -> (Self, Vec<StateSpec<'a>>) {
        let StateSpec { prefix, .. } = spec;

        let prefix_len = prefix.len();
        let next_prefix_len = prefix_len + 1;

        let mut prefix_match = None;
        let mut children_seen = 0;
        let mut last_seen = None;

        let next_states_paired = sorted
            .range(RangeFrom { start: prefix })
            .take_while(|(key, _)| key.starts_with(prefix))
            .filter_map(|(key, val)| {
                children_seen += 1;
                last_seen = Some((key, *val));

                if *key == prefix {
                    prefix_match = Some((key, *val));
                    None
                } else {
                    // The byte at `prefix_len` must exist
                    let &next_c = key.get(prefix_len).unwrap();
                    let next_prefix = &key[..next_prefix_len];

                    Some((
                        next_c,
                        StateSpec {
                            prefix: next_prefix,
                            index: 0, // Placeholder, will be updated later
                        },
                    ))
                }
            })
            .collect::<BTreeMap<_, _>>()
            .into_iter()
            .collect::<Vec<_>>();

        // Safety: `last_seen` is guaranteed to be `Some` here.
        let (last_k, last_v) = last_seen.unwrap();

        if children_seen == 1 {
            return (TrieState::Leaf(last_k, last_v), vec![]);
        }

        if next_states_paired.is_empty() {
            return (TrieState::Leaf(last_k, last_v), vec![]);
        }

        let mut mask = 0u8;
        let mut next_state_specs = Vec::new();

        for (i, (c, mut next_state)) in next_states_paired.into_iter().enumerate() {
            let next_node = edge_start + i;
            next_state.index = next_node;
            mask |= byte_masks[c as usize];
            next_state_specs.push(next_state);
        }

        let search_node = SearchNode { mask, edge_start };
        let state = match prefix_match {
            Some((key, value)) => TrieState::SearchOrLeaf(key, value, search_node),
            None => TrieState::Search(search_node),
        };

        (state, next_state_specs)
    }
}

impl SearchNode {
    /// Evaluates the next state based on the current character.
    fn evaluate(&self, c: u8, trie: &TrieHardSized<'_, _>) -> Option<usize> {
        let c_mask = trie.masks.0[c as usize];
        let mask_res = self.mask & c_mask;
        (mask_res > 0).then(|| {
            let smaller_bits = mask_res - 1;
            let smaller_bits_mask = smaller_bits & self.mask;
            let index_offset = smaller_bits_mask.count_ones() as usize;
            self.edge_start + index_offset
        })
    }
}

/// Structure used for iterating over the contents of the trie.
#[derive(Debug)]
pub struct TrieIter<'a, T> {
    iter: TrieIterSized<'a, T>,
}

/// Iterator for a trie-hard trie built on `u8`.
#[derive(Debug)]
pub struct TrieIterSized<'a, T> {
    stack: Vec<TrieNodeIter>,
    trie: &'a TrieHardSized<'a, T>,
}

impl<'a, T> TrieIterSized<'a, T> {
    /// Creates an empty iterator.
    fn empty(trie: &'a TrieHardSized<'a, T>) -> Self {
        Self {
            stack: Vec::new(),
            trie,
        }
    }

    /// Creates a new iterator starting from a specific node index.
    fn new(trie: &'a TrieHardSized<'a, T>, node_index: usize) -> Self {
        Self {
            stack: vec![TrieNodeIter {
                node_index,
                stage: TrieNodeIterStage::Inner,
            }],
            trie,
        }
    }
}

#[derive(Debug, Default)]
struct TrieNodeIter {
    node_index: usize,
    stage: TrieNodeIterStage,
}

#[derive(Debug, PartialEq, Eq)]
enum TrieNodeIterStage {
    Inner,
    Child(usize, usize),
}

impl<'a, T> Iterator for TrieIter<'a, T>
where
    T: Copy,
{
    type Item = (&'a [u8], T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, T> Iterator for TrieIterSized<'a, T>
where
    T: Copy,
{
    type Item = (&'a [u8], T);

    fn next(&mut self) -> Option<Self::Item> {
        use TrieState as TState;
        use TrieNodeIterStage as SStage;

        while let Some(mut node_iter) = self.stack.pop() {
            let node = self.trie.nodes.get(node_iter.node_index)?;

            match (&node, node_iter.stage) {
                (TState::Leaf(key, value), SStage::Inner) => {
                    return Some((*key, *value));
                }
                (TState::SearchOrLeaf(key, value, search), SStage::Inner) => {
                    self.stack.push(TrieNodeIter {
                        node_index: node_iter.node_index,
                        stage: SStage::Child(0, search.mask.count_ones() as usize),
                    });
                    self.stack.push(TrieNodeIter {
                        node_index: search.edge_start,
                        stage: SStage::Inner,
                    });
                    return Some((*key, *value));
                }
                (TState::Search(search), SStage::Inner) => {
                    self.stack.push(TrieNodeIter {
                        node_index: node_iter.node_index,
                        stage: SStage::Child(0, search.mask.count_ones() as usize),
                    });
                    self.stack.push(TrieNodeIter {
                        node_index: search.edge_start,
                        stage: SStage::Inner,
                    });
                }
                (
                    TState::SearchOrLeaf(_, _, search) | TState::Search(search),
                    SStage::Child(mut child, child_count),
                ) => {
                    child += 1;
                    if child < child_count {
                        self.stack.push(TrieNodeIter {
                            node_index: node_iter.node_index,
                            stage: SStage::Child(child, child_count),
                        });
                        self.stack.push(TrieNodeIter {
                            node_index: search.edge_start + child,
                            stage: SStage::Inner,
                        });
                    }
                }
                _ => unreachable!(),
            }
        }

        None
    }
}

/// Enumeration of all the possible sizes of trie-hard tries.
/// Now simplified to use only `u8`.
#[derive(Debug, Clone, Default)]
pub struct TrieHard<'a, T> {
    inner: TrieHardSized<'a, T>,
}

impl<'a, T> FromIterator<&'a T> for TrieHard<'a, &'a T>
where
    T: 'a + AsRef<[u8]> + Copy,
{
    fn from_iter<I: IntoIterator<Item = &'a T>>(values: I) -> Self {
        let values = values
            .into_iter()
            .map(|v| (v.as_ref(), *v))
            .collect::<Vec<_>>();

        TrieHard::new(values)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_trivial() {
        let empty: Vec<&str> = vec![];
        let empty_trie = empty.iter().collect::<TrieHard<'_, _>>();

        assert_eq!(None, empty_trie.get("anything"));
    }

    #[rstest]
    #[case("", Some(""))]
    #[case("a", Some("a"))]
    #[case("ab", Some("ab"))]
    #[case("abc", None)]
    #[case("aac", Some("aac"))]
    #[case("aa", None)]
    #[case("aab", None)]
    #[case("adddd", Some("adddd"))]
    fn test_small_get(#[case] key: &str, #[case] expected: Option<&str>) {
        let trie = ["", "a", "ab", "aac", "adddd", "addde"]
            .into_iter()
            .collect::<TrieHard<'_, _>>();
        assert_eq!(expected, trie.get(key));
    }

    #[test]
    fn test_skip_to_leaf() {
        let trie = ["a", "aa", "aaa"].into_iter().collect::<TrieHard<'_, _>>();

        assert_eq!(trie.get("aa"), Some("aa"))
    }

    #[rstest]
    #[case(8)]
    fn test_sizes(#[case] bits: usize) {
        // Only u8 is supported, so bits should be up to 8
        let range = 0..bits;
        let bytes = range.map(|b| [b as u8]).collect::<Vec<_>>();
        let trie = bytes.iter().collect::<TrieHard<'_, _>>();

        match trie {
            TrieHard { .. } => (),
            _ => panic!("Mismatched trie sizes"),
        }
    }

    #[rstest]
    #[case(include_str!("../data/1984.txt"))]
    #[case(include_str!("../data/sun-rising.txt"))]
    fn test_full_text(#[case] text: &str) {
        let words: Vec<&str> =
            text.split(|c: char| c.is_whitespace()).collect();
        let trie: TrieHard<'_, _> = words.iter().collect();

        let unique_words = words
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        for word in &unique_words {
            assert!(trie.get(word).is_some())
        }

        assert_eq!(
            unique_words,
            trie.iter().map(|(_, v)| v).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_unicode() {
        let trie: TrieHard<'_, _> = ["bär", "bären"].into_iter().collect();

        assert_eq!(trie.get("bär"), Some("bär"));
        assert_eq!(trie.get("bä"), None);
        assert_eq!(trie.get("bären"), Some("bären"));
        assert_eq!(trie.get("bärën"), None);
    }

    #[rstest]
    #[case(&[], &[])]
    #[case(&[""], &[""])]
    #[case(&["aaa", "a", ""], &["", "a", "aaa"])]
    #[case(&["aaa", "a", ""], &["", "a", "aaa"])]
    #[case(&["", "a", "ab", "aac", "adddd", "addde"], &["", "a", "aac", "ab", "adddd", "addde"])]
    fn test_iter(#[case] input: &[&str], #[case] output: &[&str]) {
        let trie = input.iter().collect::<TrieHard<'_, _>>();
        let emitted = trie.iter().map(|(_, v)| v).collect::<Vec<_>>();
        assert_eq!(emitted, output);
    }

    #[rstest]
    #[case(&[], "", &[])]
    #[case(&[""], "", &[""])]
    #[case(&["aaa", "a", ""], "", &["", "a", "aaa"])]
    #[case(&["aaa", "a", ""], "a", &["a", "aaa"])]
    #[case(&["aaa", "a", ""], "aa", &["aaa"])]
    #[case(&["aaa", "a", ""], "aab", &[])]
    #[case(&["aaa", "a", ""], "aaa", &["aaa"])]
    #[case(&["aaa", "a", ""], "b", &[])]
    #[case(&["dad", "ant", "and", "dot", "do"], "d", &["dad", "do", "dot"])]
    fn test_prefix_search(
        #[case] input: &[&str],
        #[case] prefix: &str,
        #[case] output: &[&str],
    ) {
        let trie = input.iter().collect::<TrieHard<'_, _>>();
        let emitted = trie
            .prefix_search(prefix)
            .map(|(_, v)| v)
            .collect::<Vec<_>>();
        assert_eq!(emitted, output);
    }
}
