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
}
