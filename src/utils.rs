use std::collections::{HashMap, HashSet};

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
            num_set1.insert(k.clone());
        }
        if set2.contains(v) {
            num_set2.insert(k.clone());
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
    list_of_tuples: Vec<(String, String, String)>,
) -> (HashMap<String, HashMap<String, HashSet<String>>>, HashMap<String, f64>) {
    let mut subject_map: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();
    let mut freq_map: HashMap<String, usize> = HashMap::new();
    let mut ic_map: HashMap<String, f64> = HashMap::new();
    let mut total_count = 0;

    for (s, p, o) in list_of_tuples {
        // Update frequency count for s and its ancestors
        let mut ancestor = &s;

        *freq_map.entry(s.clone()).or_insert(0) += 1;
        total_count += 1;
        *freq_map.entry(o.clone()).or_insert(0) += 1;
        total_count += 1;

        while let Some(predicate_map) = subject_map.get(ancestor) {
            *freq_map.entry(ancestor.clone()).or_insert(0) += 1;
            total_count += 1;
            ancestor = predicate_map.get("is_a").and_then(|set| set.iter().next()).unwrap_or(&"".to_string());
        }

        match subject_map.get_mut(&s) {
            Some(predicate_map) => match predicate_map.get_mut(&p) {
                Some(object_set) => {
                    object_set.insert(o.clone());
                }
                None => {
                    predicate_map.insert(p.to_string(), HashSet::from([o.clone()]));
                }
            },
            None => {
                let mut p_map = HashMap::new();
                p_map.insert(p.to_string(), HashSet::from([o.clone()]));
                subject_map.insert(s.to_string(), p_map);
            }
        };

        // Update frequency count for o and its ancestors
        let mut ancestor = &o;
        while let Some(predicate_map) = subject_map.get(ancestor) {
            *freq_map.entry(ancestor.clone()).or_insert(0) += 1;
            total_count += 1;
            ancestor = predicate_map.get("is_a").and_then(|set| set.iter().next()).unwrap_or(&"".to_string());
        }
    }

    // calculate IC for all terms using frequency map and total count
    for (k, v) in freq_map.iter_mut() {
        ic_map.insert(k.to_string(), (*v as f64 / total_count as f64).log2());
    }
    (subject_map, ic_map)
}


pub fn expand_term_using_closure(
    term: &String,
    closure_table: &HashMap<String, HashMap<String, HashSet<String>>>,
    predicates: &Option<HashSet<String>>,
) -> HashSet<String> {
    let mut closure: HashSet<String> = HashSet::new();
    if let Some(term_closure) = closure_table.get(term) {
        if predicates.is_none() {
            for (_, closure_terms) in term_closure {
                closure.extend(closure_terms.iter().map(|s| s.to_owned()));
            }
        } else {
            for pred in predicates.as_ref().unwrap() {
                if let Some(closure_terms) = term_closure.get(pred) {
                    closure.extend(closure_terms.iter().map(|s| s.to_owned()));
                }
            }
        }
    }
    closure
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
            (1 as i32, String::from("apple")),
            (2 as i32, String::from("banana")),
            (3 as i32, String::from("orange")),
            (4 as i32, String::from("blueberry")),
            (5 as i32, String::from("blackberry")),
            (6 as i32, String::from("grapes")),
            (7 as i32, String::from("fruits")),
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
        let expected_map: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::from([
            (
                String::from("ABCD:123"),
                HashMap::from([
                    (
                        String::from("is_a"),
                        HashSet::from([String::from("BCDE:234")]),
                    ),
                    (
                        String::from("part_of"),
                        HashSet::from([String::from("ABCDE:1234")]),
                    ),
                ]),
            ),
            (
                String::from("XYZ:123"),
                HashMap::from([
                    (
                        String::from("is_a"),
                        HashSet::from([String::from("WXY:234")]),
                    ),
                    (
                        String::from("part_of"),
                        HashSet::from([String::from("WXYZ:1234")]),
                    ),
                ]),
            ),
        ]);
        //{"ABCD:123": {"is_a": {"BCDE:234"}, "part_of": {"ABCDE:1234"}}, "XYZ:123": {"is_a": {"WXY:234"}, "part_of": {"WXYZ:1234"}}}
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

        let subject_map = convert_list_of_tuples_to_hashmap(list_of_tuples);
        // println!("{:?}",subject_map);
        assert_eq!(expected_map, subject_map);
    }

    #[test]
    fn test_expand_term_using_closure() {
        let mut closure_table: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();
        let mut map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("CARO:0000000"));
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);
        closure_table.insert(String::from("CARO:0000000"), map.clone());

        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);
        closure_table.insert(String::from("BFO:0000002"), map.clone());

        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);
        closure_table.insert(String::from("BFO:0000003"), map);

        let term = String::from("CARO:0000000");
        let predicates = HashSet::from(["subClassOf".to_string()]);
        let result = expand_term_using_closure(&term, &closure_table, &Some(predicates));

        let expected_result = HashSet::from([
            "BFO:0000002".to_string(),
            "BFO:0000003".to_string(),
            "CARO:0000000".to_string(),
        ]);
        assert_eq!(result, expected_result);
    }
}
