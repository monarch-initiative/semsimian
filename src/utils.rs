use std::collections::{HashMap, HashSet};

pub fn convert_set_to_hashmap(set1: &HashSet<String>) -> HashMap<i32, String> {
    let mut result = HashMap::new();
    for (idx, item) in set1.iter().enumerate() {
        result.insert(idx as i32 + 1, String::from(item));
    }
    result
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

        // println!("{:?}",convert_set_to_hashmap(&set));
        assert_eq!(set.len(), convert_set_to_hashmap(&set).len());
    }
}
