use std::collections::HashSet;

pub fn calculate_jaccard_similarity(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    /* Returns Jaccard similarity between the two sets. */

    let intersection = set1.intersection(&set2).count();
    let union_measure = set1.union(&set2).count();
    let jaccard = intersection as f64 / union_measure as f64;
    jaccard
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_calculate_jaccard_similarity(){
        let set1:HashSet<String> = HashSet::from([
                                    String::from("apple"),
                                    String::from("banana"),
                                    ]);
        let set2:HashSet<String> = HashSet::from([
                                    String::from("apple"),
                                    String::from("banana"),
                                    String::from("fruit"),
                                    String::from("tropical"),
                                    ]);
        let result = calculate_jaccard_similarity(&set1, &set2);
        println!("{result}");
        assert_eq!(result, 0.5);
    }
}