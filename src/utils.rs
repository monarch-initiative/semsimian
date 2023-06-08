use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};

type Predicate = String;
type TermID = String;
type PredicateSetKey = String;

pub fn predicate_set_to_key(predicates: &Option<HashSet<Predicate>>) -> PredicateSetKey {
    let mut result = String::new();

    if predicates.is_none() {
        result.push_str("_all");
    } else {
        let mut vec_of_predicates: Vec<String> = predicates
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.to_string())
            .collect();
        vec_of_predicates.sort();

        for predicate in vec_of_predicates {
            result.push('+');
            result.push_str(&predicate);
        }
    }
    println!("Returning key: {}", result); // for debugging

    result
}

pub fn convert_set_to_hashmap(set1: &HashSet<String>) -> HashMap<i32, String> {
    let mut result = HashMap::new();
    for (idx, item) in set1.iter().enumerate() {
        result.insert(idx as i32 + 1, String::from(item));
    }
    result
}

pub fn numericize_sets(
    set1: &HashSet<String>,
    set2: &HashSet<String>,
) -> (HashSet<i32>, HashSet<i32>, HashMap<i32, String>) {
    let mut union_set = set1.clone();
    union_set.extend(set2.clone());
    let union_set_hashmap = convert_set_to_hashmap(&union_set);
    let mut num_set1 = HashSet::new();
    let mut num_set2 = HashSet::new();

    for (k, v) in union_set_hashmap.iter() {
        if set1.contains(v) {
            num_set1.insert(*k);
        }
        if set2.contains(v) {
            num_set2.insert(*k);
        }
    }
    (num_set1, num_set2, union_set_hashmap)
}

pub fn _stringify_sets_using_map(
    set1: &HashSet<i32>,
    set2: &HashSet<i32>,
    map: &HashMap<i32, String>,
) -> (HashSet<String>, HashSet<String>) {
    let mut str_set1 = HashSet::new();
    let mut str_set2 = HashSet::new();

    for (k, v) in map.iter() {
        if set1.contains(k) {
            str_set1.insert(v.clone());
        }
        if set2.contains(k) {
            str_set2.insert(v.clone());
        }
    }
    (str_set1, str_set2)
}

pub fn convert_list_of_tuples_to_hashmap(
    list_of_tuples: &Vec<(TermID, PredicateSetKey, TermID)>,
    predicates: &Option<HashSet<String>>,
) -> (
    HashMap<String, HashMap<String, HashSet<String>>>,
    HashMap<String, HashMap<String, f64>>,
) {
    let mut closure_map: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();
    let mut freq_map: HashMap<String, usize> = HashMap::new();
    let mut ic_map: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut total_count = 0;

    let predicate_set_key: PredicateSetKey = predicate_set_to_key(predicates);

    let progress_bar = generate_progress_bar_of_length_and_message(
        list_of_tuples.len() as u64,
        "Building closure and IC map:",
    );

    for (s, p, o) in list_of_tuples.iter() {
        if predicates.is_some() && !predicates.as_ref().unwrap().contains(p) {
            continue;
        }

        *freq_map.entry(s.clone()).or_insert(0) += 1;
        total_count += 1;

        *freq_map.entry(o.clone()).or_insert(0) += 1;
        total_count += 1;

        closure_map
            .entry(predicate_set_key.clone())
            .or_insert_with(HashMap::new)
            .entry(s.clone())
            .or_insert_with(HashSet::new)
            .insert(o.clone());

        progress_bar.inc(1);
    }

    progress_bar.finish();

    for (k, v) in freq_map.iter() {
        ic_map
            .entry(predicate_set_key.clone())
            .or_insert_with(HashMap::new)
            .insert(k.clone(), -(*v as f64 / total_count as f64).log2());
    }

    (closure_map, ic_map)
}

pub fn expand_term_using_closure(
    term: &str,
    closure_table: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    predicates: &Option<HashSet<Predicate>>,
) -> HashSet<TermID> {
    let mut ancestors: HashSet<String> = HashSet::new();
    let mut this_predicate_set_key = predicate_set_to_key(predicates);
    if this_predicate_set_key == "_all" {
        let closure_table_keys: Vec<String> = closure_table.keys().cloned().collect();
        this_predicate_set_key = closure_table_keys.join("+");
    }

    for (closure_predicate_key, closure_map) in closure_table.iter() {
        if *closure_predicate_key == this_predicate_set_key {
            if let Some(ancestors_for_predicates) = closure_map.get(term) {
                ancestors.extend(ancestors_for_predicates.clone());
            }
        }
    }
    ancestors
}

pub fn generate_progress_bar_of_length_and_message(length: u64, message: &str) -> ProgressBar {
    let progress_bar = ProgressBar::new(length);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "[{{elapsed_precise}}] {} {{bar:40.cyan/blue}} {{percent}}%",
                message
            ))
            .unwrap(),
    );
    progress_bar
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]

    fn test_convert_set_to_hashmap() {
        let set: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("mango"),
            String::from("grapes"),
        ]);

        assert_eq!(set.len(), convert_set_to_hashmap(&set).len());
    }

    #[test]
    fn test_numericize_set() {
        let set1: HashSet<String> = HashSet::from([
            String::from("grapes"),
            String::from("blueberry"),
            String::from("fruit"),
            String::from("blackberry"),
        ]);
        let set2: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("fruit"),
            String::from("tropical"),
        ]);

        let (num_set1, num_set2, _) = numericize_sets(&set1, &set2);

        assert_eq!(set1.len(), num_set1.len());
        assert_eq!(set2.len(), num_set2.len());
    }

    #[test]
    fn test_stringify_sets_using_map() {
        let set1: HashSet<i32> = HashSet::from([1, 2, 3, 4, 5]);
        let set2: HashSet<i32> = HashSet::from([3, 4, 5, 6, 7]);
        let map = HashMap::from([
            (1_i32, String::from("apple")),
            (2_i32, String::from("banana")),
            (3_i32, String::from("orange")),
            (4_i32, String::from("blueberry")),
            (5_i32, String::from("blackberry")),
            (6_i32, String::from("grapes")),
            (7_i32, String::from("fruits")),
        ]);
        let (str_set1, str_set2) = _stringify_sets_using_map(&set1, &set2, &map);
        assert_eq!(set1.len(), str_set1.len());
        assert_eq!(set2.len(), str_set2.len());
    }

    #[test]
    fn test_str_to_int_to_back() {
        let set1: HashSet<String> = HashSet::from([
            String::from("grapes"),
            String::from("blueberry"),
            String::from("fruit"),
            String::from("blackberry"),
        ]);
        let set2: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("fruit"),
            String::from("tropical"),
        ]);

        let (num_set1, num_set2, map) = numericize_sets(&set1, &set2);
        let (str_set1, str_set2) = _stringify_sets_using_map(&num_set1, &num_set2, &map);

        assert_eq!(set1, str_set1);
        assert_eq!(set2, str_set2);
    }

    #[test]
    fn test_convert_list_of_tuples_to_hashmap() {
        let list_of_tuples: Vec<(String, String, String)> = vec![
            (
                String::from("ABCD:123"),
                String::from("is_a"),
                String::from("BCDE:234"),
            ),
            (
                String::from("ABCD:123"),
                String::from("part_of"),
                String::from("ABCDE:1234"),
            ),
            (
                String::from("XYZ:123"),
                String::from("is_a"),
                String::from("WXY:234"),
            ),
            (
                String::from("XYZ:123"),
                String::from("part_of"),
                String::from("WXYZ:1234"),
            ),
        ];

        // test closure map for is_a predicates
        let expected_closure_map_is_a: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::from([(
                String::from("+is_a"),
                HashMap::from([
                    (
                        String::from("ABCD:123"),
                        [String::from("BCDE:234")]
                            .iter()
                            .cloned()
                            .collect::<HashSet<_>>(),
                    ),
                    (
                        String::from("XYZ:123"),
                        [String::from("WXY:234")]
                            .iter()
                            .cloned()
                            .collect::<HashSet<_>>(),
                    ),
                ]),
            )]);

        let predicates_is_a: Option<HashSet<Predicate>> =
            Some(["is_a"].iter().map(|&s| s.to_string()).collect());
        let (closure_map_is_a, _) =
            convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates_is_a);
        assert_eq!(expected_closure_map_is_a, closure_map_is_a);

        // test closure_map for is_a + part_of predicates
        let expected_closure_map_is_a_plus_part_of: HashMap<
            PredicateSetKey,
            HashMap<TermID, HashSet<TermID>>,
        > = HashMap::from([(
            String::from("+is_a+part_of"),
            HashMap::from([
                (
                    String::from("ABCD:123"),
                    [String::from("BCDE:234"), String::from("ABCDE:1234")]
                        .iter()
                        .cloned()
                        .collect::<HashSet<TermID>>(),
                ),
                (
                    String::from("XYZ:123"),
                    [String::from("WXY:234"), String::from("WXYZ:1234")]
                        .iter()
                        .cloned()
                        .collect::<HashSet<TermID>>(),
                ),
            ]),
        )]);

        let predicates_is_a_plus_part_of: Option<HashSet<Predicate>> =
            Some(["is_a", "part_of"].iter().map(|&s| s.to_string()).collect());
        let (closure_map_is_a_plus_part_of, ic_map) =
            convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates_is_a_plus_part_of);
        assert_eq!(
            expected_closure_map_is_a_plus_part_of,
            closure_map_is_a_plus_part_of
        );

        let expected_ic_map_is_a_plus_part_of: HashMap<PredicateSetKey, HashMap<TermID, f64>> = {
            let mut expected: HashMap<TermID, f64> = HashMap::new();
            let total_count = 8;

            expected.insert(String::from("ABCD:123"), -(2.0 / total_count as f64).log2());
            expected.insert(String::from("BCDE:234"), -(1.0 / total_count as f64).log2());
            expected.insert(
                String::from("ABCDE:1234"),
                -(1.0 / total_count as f64).log2(),
            );
            expected.insert(String::from("XYZ:123"), -(2.0 / total_count as f64).log2());
            expected.insert(String::from("WXY:234"), -(1.0 / total_count as f64).log2());
            expected.insert(
                String::from("WXYZ:1234"),
                -(1.0 / total_count as f64).log2(),
            );

            let mut expected_ic_map_is_a_plus_part_of: HashMap<
                PredicateSetKey,
                HashMap<TermID, f64>,
            > = HashMap::new();
            expected_ic_map_is_a_plus_part_of.insert(String::from("+is_a+part_of"), expected);
            expected_ic_map_is_a_plus_part_of
        };

        assert_eq!(ic_map, expected_ic_map_is_a_plus_part_of);

        // Test closure map for None predicates
        let _expected_closure_map_none: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::from([(
                String::from("+is_a+part_of"),
                HashMap::from([
                    (
                        String::from("ABCD:123"),
                        HashSet::from(
                            [String::from("BCDE:234"), String::from("ABCDE:1234")]
                                .iter()
                                .cloned()
                                .collect::<HashSet<TermID>>(),
                        ),
                    ),
                    (
                        String::from("XYZ:123"),
                        HashSet::from(
                            [String::from("WXY:234"), String::from("WXYZ:1234")]
                                .iter()
                                .cloned()
                                .collect::<HashSet<TermID>>(),
                        ),
                    ),
                ]),
            )]);

        let predicates_none: Option<HashSet<Predicate>> = None;
        println!("Passing predicates: {:?}", predicates_none); // for debugging

        let (closure_map_none, _) =
            convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates_none);
        println!("Received closure map: {:?}", closure_map_none); // for debugging

        // when no predicates are specified predicates will be set to _all to cover all relations
        assert!(closure_map_none.contains_key("_all"));
    }

    #[test]
    fn test_predicate_set_to_string() {
        let predicates_is_a: Option<HashSet<Predicate>> =
            Some(["is_a"].iter().map(|&s| s.to_string()).collect());
        let predicates_is_a_part_of: Option<HashSet<Predicate>> =
            Some(["is_a", "part_of"].iter().map(|&s| s.to_string()).collect());
        let predicates_part_of_is_a: Option<HashSet<Predicate>> =
            Some(["part_of", "is_a"].iter().map(|&s| s.to_string()).collect());
        let predicates_empty: Option<HashSet<Predicate>> = None;

        assert_eq!(predicate_set_to_key(&predicates_is_a), "+is_a");
        assert_eq!(
            predicate_set_to_key(&predicates_is_a_part_of),
            "+is_a+part_of"
        );
        assert_eq!(
            predicate_set_to_key(&predicates_part_of_is_a),
            "+is_a+part_of"
        );
        assert_eq!(predicate_set_to_key(&predicates_empty), "_all");
    }

    #[test]
    fn test_expand_term_using_closure() {
        let mut closure_table: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::new();
        let mut map: HashMap<PredicateSetKey, HashSet<TermID>> = HashMap::new();
        let mut set: HashSet<TermID> = HashSet::new();
        set.insert(String::from("CARO:0000000"));
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("CARO:0000000"), set);
        closure_table.insert(String::from("+subClassOf"), map.clone());

        let mut set: HashSet<TermID> = HashSet::new();
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("BFO:0000002"), set);
        closure_table.insert(String::from("+subClassOf"), map.clone());

        let mut set: HashSet<TermID> = HashSet::new();
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("BFO:0000003"), set);
        closure_table.insert(String::from("+subClassOf"), map);

        let term = String::from("CARO:0000000");
        let predicates: Option<HashSet<Predicate>> =
            Some(HashSet::from(["subClassOf".to_string()]));
        let result_1 = expand_term_using_closure(&term, &closure_table, &predicates);
        let result_2 = expand_term_using_closure(&term, &closure_table, &None);

        let expected_result = HashSet::from([
            "BFO:0000002".to_string(),
            "BFO:0000003".to_string(),
            "CARO:0000000".to_string(),
        ]);
        assert_eq!(result_1, expected_result);
        assert_eq!(result_2, expected_result);
    }
}
