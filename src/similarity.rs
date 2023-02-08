use std::collections::HashSet;

pub fn calculate_jaccard_similarity(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    /* Returns Jaccard similarity between the two sets. */

    let intersection = set1.intersection(&set2).count();
    let union_measure = set1.union(&set2).count();
    intersection as f64 / union_measure as f64
}