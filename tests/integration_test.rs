use std::collections::HashSet;
extern crate rustsim;
use rustsim::similarity::calculate_semantic_jaccard_similarity;
use rustsim::utils::convert_list_of_tuples_to_hashmap;

#[test]
fn integration_test_semantic_jaccard_similarity() {
    let list_of_tuples = vec![
        ("apple".to_string(), "is_a".to_string(), "fruit".to_string()),
        (
            "apple".to_string(),
            "subclass_of".to_string(),
            "red".to_string(),
        ),
        (
            "cherry".to_string(),
            "subclass_of".to_string(),
            "red".to_string(),
        ),
        (
            "cherry".to_string(),
            "is_a".to_string(),
            "fruit".to_string(),
        ),
        (
            "cherry".to_string(),
            "is_a".to_string(),
            "seeded_fruit".to_string(),
        ),
        (
            "seeded_fruit".to_string(),
            "is_a".to_string(),
            "fruit".to_string(),
        ),
    ];
    let closure_table = convert_list_of_tuples_to_hashmap(list_of_tuples);
    let sem_jaccard = calculate_semantic_jaccard_similarity(
        &closure_table,
        "apple".to_string(),
        "cherry".to_string(),
        &HashSet::from(["is_a".to_string()]),
    );

    assert_eq!(sem_jaccard, 0.5)
}
