use std::collections::{HashMap, HashSet};

pub fn expand_terms_using_closure(
    terms: &HashSet<String>,
    term_closure_map: &HashMap<String, HashSet<String>>,
) -> HashSet<String> {
    /* Expand terms by inclusing ancestors in the set. */
    let mut expanded_set = HashSet::<String>::new();
    for item in terms.iter() {
        expanded_set.extend(term_closure_map.get(item).unwrap().clone());
    }
    expanded_set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_terms_using_closure() {
        let input_set: HashSet<String> =
            HashSet::from([String::from("apple"), String::from("banana")]);

        let closure_map: HashMap<String, HashSet<String>> = [
            (
                String::from("apple"),
                HashSet::from([String::from("apple"), String::from("fruit")]),
            ),
            (
                String::from("banana"),
                HashSet::from([
                    String::from("banana"),
                    String::from("fruit"),
                    String::from("tropical"),
                ]),
            ),
        ]
        .iter()
        .cloned()
        .collect();

        let result: HashSet<String> = expand_terms_using_closure(&input_set, &closure_map);
        println!("{result:?}");
        assert_eq!(result.len(), 4);
    }
}
