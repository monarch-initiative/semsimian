use std::collections::{HashMap, HashSet};

type Predicate = String;
type TermID = String;
type PredicateSetKey = String;

pub fn predicate_set_to_key(predicate_set: &HashSet<String>) -> PredicateSetKey {
    let mut result = String::new();

    let mut vec_of_predicates: Vec<String> = predicate_set.iter().map(|x| x.to_string()).collect();
    vec_of_predicates.sort();

    for predicate in vec_of_predicates {
        result.push_str("+");
        result.push_str(&predicate);
    }
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
    predicates: &Option<HashSet<Predicate>>
) -> (HashMap<String, HashMap<String, HashSet<String>>>, HashMap<String, f64>) {
    let mut subject_map: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();
    let mut freq_map: HashMap<String, usize> = HashMap::new();
    let mut ic_map: HashMap<String, f64> = HashMap::new();
    let mut total_count = 0;
    let empty_string = "".to_string();

    fn get_term_frequencies(
        term: &String,
        predicate: &str,
        subject_map: &mut HashMap<String, HashMap<String, HashSet<String>>>,
        freq_map: &mut HashMap<String, usize>,
        empty_string: &String,
    ) {
        let mut ancestor = term;
        while let Some(predicate_map) = subject_map.get(ancestor) {
            *freq_map.entry(ancestor.clone()).or_insert(0) += 1;
            // 
            ancestor = predicate_map.get(predicate).and_then(|set| set.iter().next()).unwrap_or(empty_string);
        }
    }

    for (s, p, o) in list_of_tuples {
        if !predicates.contains(&p) && !predicates.is_none(){
            continue;
        }
        *freq_map.entry(s.clone()).or_insert(0) += 1;
        total_count += 1;
        *freq_map.entry(o.clone()).or_insert(0) += 1;
        total_count += 1;

        get_term_frequencies(&s, &p, &mut subject_map, &mut freq_map, &empty_string);
        get_term_frequencies(&o, &p, &mut subject_map, &mut freq_map, &empty_string);

        subject_map.entry(s.clone())
            .or_insert_with(HashMap::new)
            .entry(p.clone())
            .or_insert_with(HashSet::new)
            .insert(o.clone());
    }

    for (k, v) in freq_map.iter() {
        ic_map.insert(k.to_string(), (*v as f64 / total_count as f64).log2());
    }

    (subject_map, ic_map)
}


pub fn expand_term_using_closure(
    term: &String,
    closure_table: &HashMap<HashSet<String>, HashMap<String, HashSet<String>>>,
    predicates: &HashSet<String>,
) -> HashSet<String> {
    let mut ancestors: HashSet<String> = HashSet::new();

    for (closure_predicates, closure_map) in closure_table.iter() {
        if closure_predicates == predicates {
            if let Some(ancestors_for_predicates) = closure_map.get(term) {
                ancestors.extend(*ancestors_for_predicates);
            }
        }
    }
    ancestors
}


// pub fn expand_term_using_closure(
//     term: &String,
//     closure_table: &HashMap<HashMap<String, HashSet<String>>>
// ) -> HashSet<String> {
//     let mut closure: HashSet<String> = HashSet::new();
//     let mut ancestor = &[term].iter().cloned().collect();
//     if let Some(term_closure) = closure_table.get(term) {
//         closure = term_closure.unwrap().iter().map(|s| s.to_owned()).collect();
//     }
//     closure
// }

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

        let (subject_map, ic_map) = convert_list_of_tuples_to_hashmap(list_of_tuples);
        // println!("{:?}",subject_map);
        assert_eq!(expected_map, subject_map);

        let expected_ic_values: HashMap<String, f64> = {
            let mut expected = HashMap::new();
            let total_count = 8;
        
            expected.insert(String::from("ABCD:123"), -(2.0 / total_count as f64).log2());
            expected.insert(String::from("BCDE:234"), -(1.0 / total_count as f64).log2());
            expected.insert(String::from("ABCDE:1234"), -(1.0 / total_count as f64).log2());
            expected.insert(String::from("XYZ:123"), -(2.0 / total_count as f64).log2());
            expected.insert(String::from("WXY:234"), -(1.0 / total_count as f64).log2());
            expected.insert(String::from("WXYZ:1234"), -(1.0 / total_count as f64).log2());
        
            expected
        };

        assert_eq!(ic_map, expected_ic_values);

        
    }

    #[test]
    fn test_predicate_set_to_string(){
        let mut predicate_set1: HashSet<String> = HashSet::new();
        predicate_set1.insert(String::from("is_a"));
        assert_eq!(predicate_set_to_key(&predicate_set1), "+is_a");

        let mut predicate_set2: HashSet<String> = HashSet::new();
        predicate_set2.insert(String::from("is_a"));
        predicate_set2.insert(String::from("part_of"));
        assert_eq!(predicate_set_to_key(&predicate_set2), "+is_a+part_of");

        let mut predicate_set3: HashSet<String> = HashSet::new();
        predicate_set3.insert(String::from("part_of"));
        predicate_set3.insert(String::from("is_a"));
        assert_eq!(predicate_set_to_key(&predicate_set3), "+is_a+part_of");
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
        let result = expand_term_using_closure(&term, &closure_table);

        let expected_result = HashSet::from([
            "BFO:0000002".to_string(),
            "BFO:0000003".to_string(),
            "CARO:0000000".to_string(),
        ]);
        assert_eq!(result, expected_result);
    }
}
