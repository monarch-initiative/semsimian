extern crate semsimian;
use semsimian::similarity::calculate_semantic_jaccard_similarity;
use semsimian::utils::convert_list_of_tuples_to_hashmap;
use semsimian::Predicate;
use semsimian::RustSemsimian;

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
    let predicates = Some(vec!["is_a".to_string()]);
    let (closure_table, _) = convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates);
    let sem_jaccard =
        calculate_semantic_jaccard_similarity(&closure_table, "apple", "cherry", &predicates);

    assert_eq!(sem_jaccard, 0.5)
}

#[test]
fn integration_test_jaccard_similarity_from_struct() {
    let triples = vec![
        (
            "apple".to_string(),
            "related_to".to_string(),
            "apple".to_string(),
        ),
        (
            "apple".to_string(),
            "related_to".to_string(),
            "banana".to_string(),
        ),
        (
            "banana".to_string(),
            "related_to".to_string(),
            "banana".to_string(),
        ),
        (
            "banana".to_string(),
            "related_to".to_string(),
            "orange".to_string(),
        ),
        (
            "orange".to_string(),
            "related_to".to_string(),
            "orange".to_string(),
        ),
        (
            "orange".to_string(),
            "related_to".to_string(),
            "pear".to_string(),
        ),
        (
            "pear".to_string(),
            "related_to".to_string(),
            "pear".to_string(),
        ),
        (
            "pear".to_string(),
            "related_to".to_string(),
            "kiwi".to_string(),
        ),
    ];
    let predicates: Option<Vec<Predicate>> = Some(vec!["related_to".to_string()]);

    let mut rs = RustSemsimian::new(Some(triples), predicates, None, None);

    // cant do this as get_closure is private, but is tested in lib

    // // Get the closure and IC map
    // let (closure_table, _) = rs.get_closure_and_ic_map();

    // // Check that the closure table was populated correctly
    // assert_eq!(closure_table["related_to"]["apple"], ["apple", "banana"]);
    // assert_eq!(closure_table["related_to"]["banana"], ["banana", "orange"]);
    // assert_eq!(closure_table["related_to"]["orange"], ["orange", "pear"]);
    // assert_eq!(closure_table["related_to"]["pear"], ["pear", "kiwi"]);

    //should be this:
    //Closure table for triples: {"+related_to": {"apple": {"banana", "apple"}, "banana": {"orange", "banana"}, "pear": {"kiwi", "pear"}, "orange": {"orange", "pear"}}}

    let term1 = "apple".to_string();
    let term2 = "banana".to_string();

    rs.update_closure_and_ic_map();

    let sim = rs.jaccard_similarity(&term1, &term2);

    assert_eq!(sim, 1.0 / 3.0);
}
