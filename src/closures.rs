use std::{collections::{HashSet, HashMap}};

pub fn expand_terms_using_closure(
        terms:&HashSet<String> ,
        term_closure_map: &HashMap<String, HashSet<String>>
    ) -> HashSet<String> {
    /* Expand terms by inclusing ancestors in the set. */
    let mut expanded_set = HashSet::<String>::new();
    for item in terms.iter() {
        expanded_set.extend(term_closure_map.get(item).unwrap().clone());
    }
    expanded_set
}