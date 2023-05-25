use std::collections::{HashMap, HashSet};

// type Predicate = String;
// type TermID = String;
// type PredicateSetKey = String;

pub fn predicate_set_to_key<'a>(predicates: &'a Option<HashSet<&'a str>>) -> &'static str {
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
            result.push_str("+");
            result.push_str(&predicate);
        }
    }
    Box::leak(result.into_boxed_str())
}

pub fn convert_set_to_hashmap<'a>(set1: HashSet<&'a str>) -> HashMap<i32, &'a str> {
    let mut result = HashMap::new();
    for (idx, item) in set1.iter().enumerate() {
        result.insert(idx as i32 + 1, *item);
    }
    result
}

pub fn numericize_sets<'a>(
    set1: &'a HashSet<&'a str>,
    set2: &'a HashSet<&'a str>,
) -> (HashSet<i32>, HashSet<i32>, HashMap<i32, &'a str>) {
    let mut union_set = set1.clone();
    union_set.extend(set2.clone());
    let union_set_hashmap = convert_set_to_hashmap(union_set);
    let mut num_set1 = HashSet::new();
    let mut num_set2 = HashSet::new();

    for (k, v) in union_set_hashmap.iter() {
        if set1.contains(*v) {
            num_set1.insert(*k);
        }
        if set2.contains(*v) {
            num_set2.insert(*k);
        }
    }
    (num_set1, num_set2, union_set_hashmap)
}

pub fn _stringify_sets_using_map<'a>(
    set1: &HashSet<i32>,
    set2: &HashSet<i32>,
    map: &'a HashMap<i32, &'a str>,
) -> (HashSet<&'a str>, HashSet<&'a str>) {
    let mut str_set1 = HashSet::new();
    let mut str_set2 = HashSet::new();

    for (k, v) in map.iter() {
        if set1.contains(k) {
            str_set1.insert(*v);
        }
        if set2.contains(k) {
            str_set2.insert(*v);
        }
    }
    (str_set1, str_set2)
}

pub fn convert_list_of_tuples_to_hashmap<'a>(
    list_of_tuples: &'a Vec<(&'a str, &'a str, &'a str)>,
    predicates: &'a Option<HashSet<&'a str>>,
) -> (
    HashMap<&'a str, HashMap<&'a str, HashSet<&'a str>>>,
    HashMap<&'a str, HashMap<&'a str, f64>>,
) {
    let mut closure_map: HashMap<&'a str, HashMap<&'a str, HashSet<&'a str>>> = HashMap::new();
    let mut freq_map: HashMap<&'a str, usize> = HashMap::new();
    let mut ic_map: HashMap<&'a str, HashMap<&'a str, f64>> = HashMap::new();
    let mut total_count = 0;
    // let empty_string = "".to_string();

    let predicate_set_key = predicate_set_to_key(predicates);

    // fn get_term_frequencies(
    //     term: &String,
    //     predicate: &str,
    //     // subject_map: &mut HashMap<String, HashMap<String, HashSet<String>>>,
    //     subject_map: &mut HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    //     freq_map: &mut HashMap<String, usize>,
    //     empty_string: &String,
    // ) {
    //     let mut ancestor = term;
    //     while let Some(predicate_map) = subject_map.get(ancestor) {
    //         *freq_map.entry(ancestor.clone()).or_insert(0) += 1;
    //         //
    //         ancestor = predicate_map.get(predicate).and_then(|set| set.iter().next()).unwrap_or(empty_string);
    //     }
    // }

    for (s, p, o) in list_of_tuples {
        if predicates.is_some() && !predicates.as_ref().unwrap().contains(&p.clone()) {
            continue;
        }
        *freq_map.entry(s.clone()).or_insert(0) += 1;
        total_count += 1;
        *freq_map.entry(o.clone()).or_insert(0) += 1;
        total_count += 1;

        // get_term_frequencies(&s, &p, &mut subject_map, &mut freq_map, &empty_string);
        // get_term_frequencies(&o, &p, &mut subject_map, &mut freq_map, &empty_string);

        closure_map
            .entry(&predicate_set_key)
            .or_insert_with(HashMap::new)
            .entry(s.clone())
            .or_insert_with(HashSet::new)
            .insert(o.clone());
    }

    for (k, v) in freq_map.iter() {
        ic_map
            .entry(&predicate_set_key)
            .or_insert_with(HashMap::new)
            .insert(k.clone(), -(*v as f64 / total_count as f64).log2());
    }

    (closure_map, ic_map)
}

pub fn expand_term_using_closure<'a>(
    term: &'a str,
    closure_table: &'a HashMap<&'a str, HashMap<&'a str, HashSet<&'a str>>>,
    predicates: &'a Option<HashSet<&'a str>>,
) -> HashSet<&'a str> {
    let mut ancestors: HashSet<&str> = HashSet::new();
    let this_predicate_set_key = predicate_set_to_key(predicates);

    for (closure_predicate_key, closure_map) in closure_table.iter() {
        if *closure_predicate_key == this_predicate_set_key {
            if let Some(ancestors_for_predicates) = closure_map.get(term) {
                ancestors.extend(ancestors_for_predicates);
            }
        }
    }
    ancestors
}

pub fn convert_map_of_map_of_set<'a>(
    original: &'a HashMap<String, HashMap<String, HashSet<String>>>,
) -> HashMap<&'a str, HashMap<&'a str, HashSet<&'a str>>> {
    let mut new_map = HashMap::new();

    for (key, inner_map) in original.into_iter() {
        let mut new_inner_map = HashMap::new();

        for (inner_key, inner_set) in inner_map.into_iter() {
            let new_inner_set: HashSet<&str> = inner_set.iter().map(|s| s.as_str()).collect();
            new_inner_map.insert(inner_key.as_str(), new_inner_set);
        }

        new_map.insert(key.as_str(), new_inner_map);
    }

    new_map
}

pub fn convert_map_of_map<'a>(
    original: &'a HashMap<String, HashMap<String, f64>>,
) -> HashMap<&'a str, HashMap<&'a str, f64>> {
    let mut new_map = HashMap::new();

    for (key, inner_map) in original.iter() {
        let mut new_inner_map = HashMap::new();

        for (inner_key, value) in inner_map.iter() {
            // Convert the inner key to a &str reference
            let inner_key_ref: &str = inner_key.as_str();

            // Insert the &str reference as the new key in the new inner map
            new_inner_map.insert(inner_key_ref, *value);
        }

        // Convert the outer key to a &str reference
        let key_ref: &str = key.as_str();

        // Insert the new inner map with &str references as keys into the new map
        new_map.insert(key_ref, new_inner_map);
    }

    new_map
}

pub fn convert_vector_of_string_object_to_references<'a>(
    vector: Vec<(String, String, String)>,
) -> Vec<(&'a str, &'a str, &'a str)> {
    let mut result = Vec::with_capacity(vector.len());
    for tuple in vector {
        // We can create string slice references '&' from owned strings 'String'
        // by using the '&' operator followed by the variable name.
        // For example, '&tuple.0' will give us an immutable reference to the first element of the tuple.
        let reference_tuple: (&str, &str, &str) = (&tuple.0, &tuple.1, &tuple.2);
        result.push(reference_tuple);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]

    fn test_convert_set_to_hashmap() {
        let set: HashSet<&str> = HashSet::from(["apple", "banana", "mango", "grapes"]);

        assert_eq!(set.len(), convert_set_to_hashmap(set).len());
    }

    #[test]
    fn test_numericize_set() {
        let set1: HashSet<&str> = HashSet::from(["grapes", "blueberry", "fruit", "blackberry"]);
        let set2: HashSet<&str> = HashSet::from(["apple", "banana", "fruit", "tropical"]);

        let (num_set1, num_set2, _) = numericize_sets(&set1, &set2);

        assert_eq!(set1.len(), num_set1.len());
        assert_eq!(set2.len(), num_set2.len());
    }

    #[test]
    fn test_stringify_sets_using_map() {
        let set1: HashSet<i32> = HashSet::from([1, 2, 3, 4, 5]);
        let set2: HashSet<i32> = HashSet::from([3, 4, 5, 6, 7]);
        let map = HashMap::from([
            (1 as i32, "apple"),
            (2 as i32, "banana"),
            (3 as i32, "orange"),
            (4 as i32, "blueberry"),
            (5 as i32, "blackberry"),
            (6 as i32, "grapes"),
            (7 as i32, "fruits"),
        ]);
        let (str_set1, str_set2) = _stringify_sets_using_map(&set1, &set2, &map);
        assert_eq!(set1.len(), str_set1.len());
        assert_eq!(set2.len(), str_set2.len());
    }

    #[test]
    fn test_str_to_int_to_back() {
        let set1: HashSet<&str> = HashSet::from(["grapes", "blueberry", "fruit", "blackberry"]);
        let set2: HashSet<&str> = HashSet::from(["apple", "banana", "fruit", "tropical"]);

        let (num_set1, num_set2, map) = numericize_sets(&set1, &set2);
        let (str_set1, str_set2) = _stringify_sets_using_map(&num_set1, &num_set2, &map);

        assert_eq!(set1, str_set1);
        assert_eq!(set2, str_set2);
    }

    #[test]
    fn test_convert_list_of_tuples_to_hashmap() {
        let list_of_tuples: Vec<(&str, &str, &str)> = vec![
            ("ABCD:123", "is_a", "BCDE:234"),
            ("ABCD:123", "part_of", "ABCDE:1234"),
            ("XYZ:123", "is_a", "WXY:234"),
            ("XYZ:123", "part_of", "WXYZ:1234"),
        ];

        // test closure map for is_a predicates
        let expected_closure_map_is_a: HashMap<&str, HashMap<&str, HashSet<&str>>> =
            HashMap::from([(
                "+is_a",
                HashMap::from([
                    (
                        "ABCD:123",
                        ["BCDE:234"].iter().cloned().collect::<HashSet<_>>(),
                    ),
                    (
                        "XYZ:123",
                        ["WXY:234"].iter().cloned().collect::<HashSet<_>>(),
                    ),
                ]),
            )]);

        let predicates_is_a: Option<HashSet<&str>> = Some(["is_a"].iter().map(|&s| s).collect());
        let (closure_map_is_a, _) =
            convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates_is_a);
        assert_eq!(expected_closure_map_is_a, closure_map_is_a);

        // test closure_map for is_a + part_of predicates
        let expected_closure_map_is_a_plus_part_of: HashMap<&str, HashMap<&str, HashSet<&str>>> =
            HashMap::from([(
                "+is_a+part_of",
                HashMap::from([
                    (
                        "ABCD:123",
                        HashSet::from(
                            ["BCDE:234", "ABCDE:1234"]
                                .iter()
                                .cloned()
                                .collect::<HashSet<&str>>(),
                        ),
                    ),
                    (
                        "XYZ:123",
                        HashSet::from(
                            ["WXY:234", "WXYZ:1234"]
                                .iter()
                                .cloned()
                                .collect::<HashSet<&str>>(),
                        ),
                    ),
                ]),
            )]);

        let predicates_is_a_plus_part_of: Option<HashSet<&str>> =
            Some(["is_a", "part_of"].iter().map(|&s| s).collect());
        let (closure_map_is_a_plus_part_of, ic_map) =
            convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates_is_a_plus_part_of);
        assert_eq!(
            expected_closure_map_is_a_plus_part_of,
            closure_map_is_a_plus_part_of
        );

        let expected_ic_map_is_a_plus_part_of: HashMap<&str, HashMap<&str, f64>> = {
            let mut expected: HashMap<&str, f64> = HashMap::new();
            let total_count = 8;

            expected.insert("ABCD:123", -(2.0 / total_count as f64).log2());
            expected.insert("BCDE:234", -(1.0 / total_count as f64).log2());
            expected.insert("ABCDE:1234", -(1.0 / total_count as f64).log2());
            expected.insert("XYZ:123", -(2.0 / total_count as f64).log2());
            expected.insert("WXY:234", -(1.0 / total_count as f64).log2());
            expected.insert("WXYZ:1234", -(1.0 / total_count as f64).log2());

            let mut expected_ic_map_is_a_plus_part_of: HashMap<&str, HashMap<&str, f64>> =
                HashMap::new();
            expected_ic_map_is_a_plus_part_of.insert("+is_a+part_of", expected);
            expected_ic_map_is_a_plus_part_of
        };

        assert_eq!(ic_map, expected_ic_map_is_a_plus_part_of);
    }

    #[test]
    fn test_predicate_set_to_string() {
        let predicates_is_a: Option<HashSet<&str>> = Some(["is_a"].iter().map(|&s| s).collect());
        let predicates_is_a_part_of: Option<HashSet<&str>> =
            Some(["is_a", "part_of"].iter().map(|&s| s).collect());
        let predicates_part_of_is_a: Option<HashSet<&str>> =
            Some(["part_of", "is_a"].iter().map(|&s| s).collect());
        let predicates_empty: Option<HashSet<&str>> = None;

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
        let mut closure_table: HashMap<&str, HashMap<&str, HashSet<&str>>> = HashMap::new();
        let mut map: HashMap<&str, HashSet<&str>> = HashMap::new();
        let mut set: HashSet<&str> = HashSet::new();
        set.insert("CARO:0000000");
        set.insert("BFO:0000002");
        set.insert("BFO:0000003");
        map.insert("CARO:0000000", set);
        closure_table.insert("+subClassOf", map.clone());

        let mut set: HashSet<&str> = HashSet::new();
        set.insert("BFO:0000002");
        set.insert("BFO:0000003");
        map.insert("BFO:0000002", set);
        closure_table.insert("+subClassOf", map.clone());

        let mut set: HashSet<&str> = HashSet::new();
        set.insert("BFO:0000003");
        map.insert("BFO:0000003", set);
        closure_table.insert("+subClassOf", map);

        let term = "CARO:0000000";
        let predicates: Option<HashSet<&str>> = Some(HashSet::from(["subClassOf"]));
        let result = expand_term_using_closure(&term, &closure_table, &predicates);

        let expected_result = HashSet::from(["BFO:0000002", "BFO:0000003", "CARO:0000000"]);
        assert_eq!(result, expected_result);
    }
}
