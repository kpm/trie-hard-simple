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

mod u256;

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    ops::RangeFrom,
};

use u256::U256;

/// Inner representation of a trie-hard trie using `u8` for masks.
#[derive(Debug, Clone)]
#[repr(transparent)]
struct MasksByByte([u8; 256]);

impl Default for MasksByByte {
    fn default() -> Self {
        Self([0u8; 256])
    }
}

impl MasksByByte {
    fn new(used_bytes: BTreeSet<u8>) -> Self {
        let mut mask = 1u8;
        let mut byte_masks = [0u8; 256];

        for c in used_bytes.into_iter() {
            byte_masks[c as usize] = mask;
            mask = mask.wrapping_shl(1);
            if mask == 0 {
                panic!("Exceeded u8 mask capacity");
            }
        }

        Self(byte_masks)
    }
}

/// Inner representation of a trie-hard trie using `u8` for masks.
#[derive(Debug, Clone)]
pub struct TrieHard<'a, T> {
    masks: MasksByByte,
    nodes: Vec<TrieState<'a, T>>,
}

impl<'a, T> Default for TrieHard<'a, T> {
    fn default() -> Self {
        Self {
            masks: MasksByByte::default(),
            nodes: Default::default(),
        }
    }
}

impl<'a, T> TrieHard<'a, T>
where
    T: 'a + Copy,
{
    /// Create an instance of a trie-hard trie with the given keys and values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use trie_hard::TrieHard;
    /// let trie = TrieHard::new(vec![
    ///     (b"and", 0),
    ///     (b"ant", 1),
    ///     (b"dad", 2),
    ///     (b"do", 3),
    ///     (b"dot", 4)
    /// ]);
    ///
    /// assert_eq!(trie.get("dad"), Some(2));
    /// assert_eq!(trie.get("do"), Some(3));
    /// assert_eq!(trie.get("don't"), None);
    /// ```
    pub fn new(values: Vec<(&'a [u8], T)>) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let used_bytes = values
            .iter()
            .flat_map(|(k, _)| k.iter())
            .cloned()
            .collect::<BTreeSet<_>>();

        let masks = MasksByByte::new(used_bytes);

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

        Self { masks, nodes }
    }

    /// Get the value stored for the given key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use trie_hard::TrieHard;
    /// let trie = ["and", "ant", "dad", "do", "dot"]
    ///     .into_iter()
    ///     .collect::<TrieHard<'_, _>>();
    ///
    /// assert_eq!(trie.get("dad"), Some("dad"));
    /// assert_eq!(trie.get("do"), Some("do"));
    /// assert_eq!(trie.get("don't"), None);
    /// ```
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> Option<T> {
        self.get_from_bytes(key.as_ref())
    }

    /// Get the value stored for the given byte-slice key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use trie_hard::TrieHard;
    /// let trie = ["and", "ant", "dad", "do", "dot"]
    ///     .into_iter()
    ///     .collect::<TrieHard<'_, _>>();
    ///
    /// assert_eq!(trie.get_from_bytes(b"dad"), Some("dad"));
    /// assert_eq!(trie.get_from_bytes(b"do"), Some("do"));
    /// assert_eq!(trie.get_from_bytes(b"don't"), None);
    /// ```
    pub fn get_from_bytes(&self, key: &[u8]) -> Option<T> {
        let mut state = self.nodes.get(0)?;

        for (i, c) in key.iter().enumerate() {
            let next_state_opt = match state {
                TrieState::Leaf(k, value) => {
                    return (k.len() == key.len() && k[i..] == key[i..]).then_some(*value)
                }
                TrieState::Search(search) | TrieState::SearchOrLeaf(_, _, search) => {
                    search.evaluate(*c, self)
                }
            };

            if let Some(next_state_index) = next_state_opt {
                state = &self.nodes[next_state_index];
            } else {
                return None;
            }
        }

        match state {
            TrieState::Leaf(k, value) | TrieState::SearchOrLeaf(k, value, _) => {
                (k.len() == key.len()).then_some(*value)
            }
            _ => None,
        }
    }

    /// Create an iterator over the entire trie. Emitted items will be ordered by their keys.
    ///
    /// # Examples
    ///
    /// ```
    /// # use trie_hard::TrieHard;
    /// let trie = ["dad", "ant", "and", "dot", "do"]
    ///     .into_iter()
    ///     .collect::<TrieHard<'_, _>>();
    ///
    /// let mut iter = trie.iter();
    /// assert_eq!(iter.next(), Some((b"and" as &[u8], "and")));
    /// assert_eq!(iter.next(), Some((b"ant", "ant")));
    /// assert_eq!(iter.next(), Some((b"dad", "dad")));
    /// assert_eq!(iter.next(), Some((b"do", "do")));
    /// assert_eq!(iter.next(), Some((b"dot", "dot")));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> TrieIter<'_, 'a, T> {
        TrieIter::new(self.iter_nodes())
    }

    /// Create an iterator over the portion of the trie starting with the given prefix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use trie_hard::TrieHard;
    /// let trie = ["dad", "ant", "and", "dot", "do"]
    ///     .into_iter()
    ///     .collect::<TrieHard<'_, _>>();
    ///
    /// let mut iter = trie.prefix_search("d");
    /// assert_eq!(iter.next(), Some((b"dad", "dad")));
    /// assert_eq!(iter.next(), Some((b"do", "do")));
    /// assert_eq!(iter.next(), Some((b"dot", "dot")));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn prefix_search<K: AsRef<[u8]>>(&self, prefix: K) -> TrieIter<'_, 'a, T> {
        let key = prefix.as_ref();
        let mut node_index = 0;
        let Some(mut state) = self.nodes.get(node_index) else {
            return TrieIter::empty();
        };

        for (i, c) in key.iter().enumerate() {
            let next_state_opt = match state {
                TrieState::Leaf(k, _) => {
                    if k.len() == key.len() && k[i..] == key[i..] {
                        return TrieIter::new(Some(node_index));
                    } else {
                        return TrieIter::empty();
                    }
                }
                TrieState::Search(search) | TrieState::SearchOrLeaf(_, _, search) => {
                    search.evaluate(*c, self)
                }
            };

            if let Some(next_state_index) = next_state_opt {
                node_index = next_state_index;
                state = &self.nodes[next_state_index];
            } else {
                return TrieIter::empty();
            }
        }

        TrieIter::new(Some(node_index))
    }

    fn iter_nodes(&self) -> TrieIterNodes<'_, 'a, T> {
        TrieIterNodes {
            stack: vec![TrieNodeIter::default()],
            trie: self,
        }
    }
}

/// Structure used for iterating over the contents of the trie.
#[derive(Debug)]
pub enum TrieIter<'b, 'a, T> {
    /// Iterator variant for the entire trie or a prefix search.
    Inner(TrieIterNodes<'b, 'a, T>),
}

impl<'b, 'a, T> TrieIter<'b, 'a, T>
where
    T: Copy,
{
    fn new(node_index: Option<usize>, trie: &'b TrieHard<'a, T>) -> Self {
        match node_index {
            Some(idx) => TrieIter::Inner(TrieIterNodes::new(trie, idx)),
            None => TrieIter::empty(),
        }
    }

    fn empty() -> Self {
        TrieIter::Inner(TrieIterNodes::empty())
    }
}

impl<'b, 'a, T> Iterator for TrieIter<'b, 'a, T>
where
    T: Copy,
{
    type Item = (&'a [u8], T);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TrieIter::Inner(iter) => iter.next(),
        }
    }
}

/// Structure for iterating over a trie.
#[derive(Debug)]
struct TrieIterNodes<'b, 'a, T> {
    stack: Vec<TrieNodeIter>,
    trie: &'b TrieHard<'a, T>,
}

impl<'b, 'a, T> TrieIterNodes<'b, 'a, T>
where
    T: Copy,
{
    fn new(trie: &'b TrieHard<'a, T>, node_index: usize) -> Self {
        Self {
            stack: vec![TrieNodeIter {
                node_index,
                stage: TrieNodeIterStage::Inner,
            }],
            trie,
        }
    }

    fn empty() -> Self {
        Self {
            stack: Vec::new(),
            trie: panic!("Empty iterator should not access trie"),
        }
    }
}

impl<'b, 'a, T> Iterator for TrieIterNodes<'b, 'a, T>
where
    T: Copy,
{
    type Item = (&'a [u8], T);

    fn next(&mut self) -> Option<Self::Item> {
        use TrieState::*;
        use TrieNodeIterStage::*;

        while let Some(mut node_iter) = self.stack.pop() {
            let node = self.trie.nodes.get(node_iter.node_index)?;

            match (node, node_iter.stage) {
                (Leaf(key, value), Inner) => return Some((*key, *value)),
                (SearchOrLeaf(key, value, search), Inner) => {
                    // Push the search node children onto the stack
                    let children_count = search.mask.count_ones() as usize;
                    self.stack.push(TrieNodeIter {
                        node_index: search.edge_start,
                        stage: Child(0, children_count),
                    });
                    // Emit the leaf part
                    return Some((*key, *value));
                }
                (Search(search), Inner) => {
                    let children_count = search.mask.count_ones() as usize;
                    self.stack.push(TrieNodeIter {
                        node_index: search.edge_start,
                        stage: Child(0, children_count),
                    });
                }
                (SearchOrLeaf(_, _, search), Child(child, count)) |
                (Search(search), Child(child, count)) => {
                    if *child < *count {
                        self.stack.push(TrieNodeIter {
                            node_index: node_iter.node_index,
                            stage: Child(child + 1, *count),
                        });
                        self.stack.push(TrieNodeIter {
                            node_index: search.edge_start + child,
                            stage: Inner,
                        });
                    }
                }
                _ => {}
            }
        }

        None
    }
}

/// Iterator stage for traversal.
#[derive(Debug, PartialEq, Eq)]
enum TrieNodeIterStage {
    /// Initial traversal stage.
    Inner,
    /// Traversing child nodes.
    Child(usize, usize),
}

impl Default for TrieNodeIterStage {
    fn default() -> Self {
        TrieNodeIterStage::Inner
    }
}

/// Iterator state for a node.
#[derive(Debug)]
struct TrieNodeIter {
    node_index: usize,
    stage: TrieNodeIterStage,
}

/// Represents the state of a trie node.
#[derive(Debug, Clone)]
enum TrieState<'a, T> {
    /// A leaf node containing a key and its associated value.
    Leaf(&'a [u8], T),
    /// A search node containing mask and edge_start information.
    Search(SearchNode),
    /// A search node that may also contain a leaf.
    SearchOrLeaf(&'a [u8], T, SearchNode),
}

impl<'a, T> TrieState<'a, T>
where
    T: Copy,
{
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
                    let next_c = key.get(prefix_len).unwrap();
                    let next_prefix = &key[..next_prefix_len];

                    Some((
                        *next_c,
                        StateSpec {
                            prefix: next_prefix,
                            index: 0,
                        },
                    ))
                }
            })
            .collect::<BTreeMap<_, _>>()
            .into_iter()
            .collect::<Vec<_>>();

        // Safety: last_seen will be present because we saw at least one
        //         entry must be present for this function to be called
        let (last_k, last_v) = last_seen.unwrap();

        if children_seen == 1 {
            return (TrieState::Leaf(last_k, last_v), vec![]);
        }

        // No next_states means we hit a leaf node
        if next_states_paired.is_empty() {
            return (TrieState::Leaf(last_k, last_v), vec![]);
        }

        let mut mask = 0u8;

        // Update the index for the next state now that we have ordered by
        // the children
        let next_state_specs = next_states_paired
            .into_iter()
            .enumerate()
            .map(|(i, (c, mut next_state))| {
                let next_node = edge_start + i;
                next_state.index = next_node;
                mask |= byte_masks[c as usize];
                next_state
            })
            .collect::<Vec<_>>();

        let search_node = SearchNode { mask, edge_start };
        let state = match prefix_match {
            Some((key, value)) => TrieState::SearchOrLeaf(key, value, search_node),
            None => TrieState::Search(search_node),
        };

        (state, next_state_specs)
    }
}

/// Specification for trie node creation.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct StateSpec<'a> {
    prefix: &'a [u8],
    index: usize,
}

/// Represents a search node in the trie.
#[derive(Debug, Clone)]
struct SearchNode {
    mask: u8,
    edge_start: usize,
}

impl SearchNode {
    fn evaluate(&self, c: u8, trie: &TrieHard<'_, _>) -> Option<usize> {
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

        assert_eq!(Some("aa"), trie.get("aa"))
    }

    #[rstest]
    #[case(include_str!("../data/1984.txt"))]
    #[case(include_str!("../data/sun-rising.txt"))]
    fn test_full_text(#[case] text: &str) {
        let words: Vec<&str> =
            text.split(|c: char| c.is_whitespace()).collect();
        let trie: TrieHard<'_, _> = words.iter().copied().collect();

        let unique_words = words
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        for word in &unique_words {
            assert_eq!(Some(*word), trie.get(word));
        }

        assert_eq!(
            unique_words,
            trie.iter().map(|(_, v)| v).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_unicode() {
        let trie: TrieHard<'_, _> = ["bär", "bären"].into_iter().collect();

        assert_eq!(Some("bär"), trie.get("bär"));
        assert_eq!(None, trie.get("bä"));
        assert_eq!(Some("bären"), trie.get("bären"));
        assert_eq!(None, trie.get("bärën"));
    }

    #[rstest]
    #[case(&[], &[])]
    #[case(&[""], &[""])]
    #[case(&["aaa", "a", ""], &["", "a", "aaa"])]
    #[case(&["aaa", "a", ""], &["", "a", "aaa"])]
    #[case(&["", "a", "ab", "aac", "adddd", "addde"], &["", "a", "aac", "ab", "adddd", "addde"])]
    fn test_iter(#[case] input: &[&str], #[case] output: &[&str]) {
        let trie = input.iter().copied().collect::<TrieHard<'_, _>>();
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
        let trie = input.iter().copied().collect::<TrieHard<'_, _>>();
        let emitted = trie
            .prefix_search(prefix)
            .map(|(_, v)| v)
            .collect::<Vec<_>>();
        assert_eq!(emitted, output);
    }
}

