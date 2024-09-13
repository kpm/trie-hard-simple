#![cfg_attr(not(doctest), doc = include_str!("../README.md"))]
#![deny(
//    missing_docs,
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

#[derive(Debug, Clone)]
#[repr(transparent)]
struct MasksByByte([u8; 256]);

impl Default for MasksByByte {
    fn default() -> Self {
        Self([0; 256])
    }
}

impl MasksByByte {
    fn new(used_bytes: BTreeSet<u8>) -> Self {
        let mut mask = 1u8;
        let mut byte_masks = [0u8; 256];

        for c in used_bytes {
            byte_masks[c as usize] = mask;
            mask <<= 1;
        }

        Self(byte_masks)
    }
}

#[derive(Debug, Clone)]
struct SearchNode {
    mask: u8,
    edge_start: usize,
}

impl SearchNode {
    fn evaluate<T>(&self, c: u8, trie: &TrieHardSized<'_, T>) -> Option<usize> {
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

#[derive(Debug, Clone)]
enum TrieState<'a, T> {
    Leaf(&'a [u8], T),
    Search(SearchNode),
    SearchOrLeaf(&'a [u8], T, SearchNode),
}

#[derive(Debug, Clone)]
pub struct TrieHardSized<'a, T> {
    masks: MasksByByte,
    nodes: Vec<TrieState<'a, T>>,
}

impl<'a, T> Default for TrieHardSized<'a, T> {
    fn default() -> Self {
        Self {
            masks: MasksByByte::default(),
            nodes: Vec::new(),
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct StateSpec<'a> {
    prefix: &'a [u8],
    index: usize,
}

/// Inner representation of a trie-hard trie using `u8` size.
#[derive(Debug, Clone)]
pub struct TrieHard<'a, T>(TrieHardSized<'a, T>);

impl<'a, T> Default for TrieHard<'a, T> {
    fn default() -> Self {
        TrieHard(TrieHardSized::default())
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
    /// assert!(trie.get("don't").is_none());
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

        Self(TrieHardSized::new(masks, values))
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
    /// assert!(trie.get("dad".to_owned()).is_some());
    /// assert!(trie.get(b"do").is_some());
    /// assert!(trie.get(b"don't".to_vec()).is_none());
    /// ```
    pub fn get<K: AsRef<[u8]>>(&self, raw_key: K) -> Option<T> {
        self.0.get(raw_key)
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
    /// assert!(trie.get_from_bytes(b"dad").is_some());
    /// assert!(trie.get_from_bytes(b"do").is_some());
    /// assert!(trie.get_from_bytes(b"don't").is_none());
    /// ```
    pub fn get_from_bytes(&self, key: &[u8]) -> Option<T> {
        self.0.get_from_bytes(key)
    }

    /// Create an iterator over the entire trie.
    ///
    /// # Examples
    ///
    /// ```
    /// # use trie_hard::TrieHard;
    /// let trie = ["dad", "ant", "and", "dot", "do"]
    ///     .into_iter()
    ///     .collect::<TrieHard<'_, _>>();
    ///
    /// assert_eq!(
    ///     trie.iter().map(|(_, v)| v).collect::<Vec<_>>(),
    ///     ["and", "ant", "dad", "do", "dot"]
    /// );
    /// ```
    pub fn iter(&self) -> TrieIter<'_, 'a, T> {
        TrieIter::new(&self.0)
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
    /// assert_eq!(
    ///     trie.prefix_search("d").map(|(_, v)| v).collect::<Vec<_>>(),
    ///     ["dad", "do", "dot"]
    /// );
    /// ```
    pub fn prefix_search<K: AsRef<[u8]>>(
        &self,
        prefix: K,
    ) -> TrieIter<'_, 'a, T> {
        TrieIter::new_prefix(&self.0, prefix.as_ref())
    }
}

#[derive(Debug, Default)]
struct TrieNodeIter {
    node_index: usize,
    stage: TrieNodeIterStage,
}

#[derive(Debug, Default)]
enum TrieNodeIterStage {
    #[default]
    Inner,
    Child(usize, usize),
}

/// Structure used for iterating over the contents of the trie.
#[derive(Debug)]
pub struct TrieIter<'b, 'a, T> {
    stack: Vec<TrieNodeIter>,
    trie: &'b TrieHardSized<'a, T>,
}

impl<'b, 'a, T> TrieIter<'b, 'a, T>
where
    T: Copy,
{
    fn new(trie: &'b TrieHardSized<'a, T>) -> Self {
        Self {
            stack: vec![TrieNodeIter::default()],
            trie,
        }
    }

    fn new_prefix(trie: &'b TrieHardSized<'a, T>, prefix: &[u8]) -> Self {
        let mut node_index = 0;
        let Some(mut state) = trie.nodes.get(node_index) else {
            return Self {
                stack: vec![],
                trie,
            };
        };

        for (i, &c) in prefix.iter().enumerate() {
            let next_state_opt = match state {
                TrieState::Leaf(k, _) => {
                    if k.len() == prefix.len() && k[i..] == prefix[i..] {
                        return Self::new_at(trie, node_index);
                    } else {
                        return Self {
                            stack: vec![],
                            trie,
                        };
                    }
                }
                TrieState::Search(search) | TrieState::SearchOrLeaf(_, _, search) => {
                    search.evaluate(c, trie)
                }
            };

            if let Some(next_state_index) = next_state_opt {
                node_index = next_state_index;
                state = &trie.nodes[next_state_index];
            } else {
                return Self {
                    stack: vec![],
                    trie,
                };
            }
        }

        Self::new_at(trie, node_index)
    }

    fn new_at(trie: &'b TrieHardSized<'a, T>, node_index: usize) -> Self {
        Self {
            stack: vec![TrieNodeIter {
                node_index,
                stage: Default::default(),
            }],
            trie,
        }
    }
}

impl<'b, 'a, T> Iterator for TrieIter<'b, 'a, T>
where
    T: Copy,
{
    type Item = (&'a [u8], T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, node_index, stage)) = self.stack.pop().and_then(
            |TrieNodeIter { node_index, stage }| {
                self.trie.nodes.get(node_index).map(|node| (node, node_index, stage))
            },
        ) {
            match (node, stage) {
                (TrieState::Leaf(key, value), TrieNodeIterStage::Inner) => {
                    return Some((*key, *value));
                }
                (
                    TrieState::SearchOrLeaf(key, value, search),
                    TrieNodeIterStage::Inner,
                ) => {
                    self.stack.push(TrieNodeIter {
                        node_index,
                        stage: TrieNodeIterStage::Child(
                            0,
                            search.mask.count_ones() as usize,
                        ),
                    });
                    self.stack.push(TrieNodeIter {
                        node_index: search.edge_start,
                        stage: Default::default(),
                    });
                    return Some((*key, *value));
                }
                (TrieState::Search(search), TrieNodeIterStage::Inner) => {
                    self.stack.push(TrieNodeIter {
                        node_index,
                        stage: TrieNodeIterStage::Child(
                            0,
                            search.mask.count_ones() as usize,
                        ),
                    });
                    self.stack.push(TrieNodeIter {
                        node_index: search.edge_start,
                        stage: Default::default(),
                    });
                }
                (
                    TrieState::Search(search) | TrieState::SearchOrLeaf(_, _, search),
                    TrieNodeIterStage::Child(mut child, child_count),
                ) => {
                    child += 1;
                    if child < child_count {
                        self.stack.push(TrieNodeIter {
                            node_index,
                            stage: TrieNodeIterStage::Child(child, child_count),
                        });
                        self.stack.push(TrieNodeIter {
                            node_index: search.edge_start + child,
                            stage: Default::default(),
                        });
                    }
                }
                _ => unreachable!(),
            }
        }

        None
    }
}

impl<'a, T> FromIterator<&'a T> for TrieHard<'a, &'a T>
where
    T: 'a + AsRef<[u8]> + ?Sized,
{
    fn from_iter<I: IntoIterator<Item = &'a T>>(values: I) -> Self {
        let values = values
            .into_iter()
            .map(|v| (v.as_ref(), v))
            .collect::<Vec<_>>();

        Self::new(values)
    }
}

impl<'a, T> TrieHardSized<'a, T>
where
    T: Copy,
{
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
            assert_eq!(spec.index, nodes.len());
            let (state, next_specs) =
                TrieState::new(spec, next_index, &masks.0, &sorted);

            next_index += next_specs.len();
            spec_queue.extend(next_specs);
            nodes.push(state);
        }

        TrieHardSized { masks, nodes }
    }

    /// Get the value stored for the given key.
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> Option<T> {
        self.get_from_bytes(key.as_ref())
    }

    /// Get the value stored for the given byte-slice key.
    pub fn get_from_bytes(&self, key: &[u8]) -> Option<T> {
        let mut state = self.nodes.get(0)?;

        for (i, &c) in key.iter().enumerate() {
            let next_state_opt = match state {
                TrieState::Leaf(k, value) => {
                    return (k.len() == key.len() && k[i..] == key[i..])
                        .then_some(*value)
                }
                TrieState::Search(search) | TrieState::SearchOrLeaf(_, _, search) => {
                    search.evaluate(c, self)
                }
            };

            if let Some(next_state_index) = next_state_opt {
                state = &self.nodes[next_state_index];
            } else {
                return None;
            }
        }

        if let TrieState::Leaf(k, value) | TrieState::SearchOrLeaf(k, value, _) = state {
            (k.len() == key.len()).then_some(*value)
        } else {
            None
        }
    }
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
            .filter_map(|(key, &val)| {
                children_seen += 1;
                last_seen = Some((key, val));

                if *key == prefix {
                    prefix_match = Some((key, val));
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

        let (last_k, last_v) = last_seen.unwrap();

        if children_seen == 1 {
            return (TrieState::Leaf(last_k, last_v), vec![]);
        }

        if next_states_paired.is_empty() {
            return (TrieState::Leaf(last_k, last_v), vec![]);
        }

        let mut mask = 0u8;

        let next_state_specs = next_states_paired
            .into_iter()
            .enumerate()
            .map(|(i, (c, mut next_state))| {
                let next_node = edge_start + i;
                next_state.index = next_node;
                mask |= byte_masks[c as usize];
                next_state
            })
            .collect();

        let search_node = SearchNode { mask, edge_start };

        let state = match prefix_match {
            Some((key, value)) => TrieState::SearchOrLeaf(key, value, search_node),
            None => TrieState::Search(search_node),
        };

        (state, next_state_specs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

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
