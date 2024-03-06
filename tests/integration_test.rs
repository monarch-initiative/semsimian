extern crate semsimian;
use std::collections::HashSet;

use semsimian::enums::MetricEnum;
use semsimian::enums::SearchTypeEnum;
use semsimian::similarity::calculate_average_termset_information_content;
use semsimian::similarity::calculate_semantic_jaccard_similarity;
use semsimian::utils::convert_list_of_tuples_to_hashmap;
use semsimian::Predicate;
use semsimian::RustSemsimian;
use semsimian::TermID;

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

#[test]
fn integration_test_search_and_avg_score() {
    let predicates: Option<Vec<Predicate>> = Some(vec![
        Predicate::from("rdfs:subClassOf"),
        Predicate::from("BFO:0000050"),
    ]);
    let db = Some("tests/data/go-nucleus.db");
    let mut rss = RustSemsimian::new(None, predicates, None, db);

    rss.update_closure_and_ic_map();

    let assoc_predicate: HashSet<TermID> = HashSet::from(["biolink:has_nucleus".to_string()]);
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["GO:".to_string()]);
    let object_terms: HashSet<TermID> =
        HashSet::from(["GO:0031965".to_string(), "GO:0005773".to_string()]);
    let search_type: SearchTypeEnum = SearchTypeEnum::Full;
    let limit: Option<usize> = Some(5);
    let score_metric = MetricEnum::AncestorInformationContent;

    let result = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        true,
        &None,
        &subject_prefixes,
        &search_type,
        &score_metric,
        limit,
    );
    // dbg!(&result);

    // Test case 1: Normal case, entities have terms.
    let entity1: HashSet<TermID> = HashSet::from([
        "GO:0031090".to_string(),
        "GO:0005634".to_string(),
        "GO:0005635".to_string(),
    ]);
    let avg_ic_score = calculate_average_termset_information_content(&rss, &entity1, &object_terms);
    let (score_from_result, _, _) = result[0];
    assert_eq!(score_from_result, avg_ic_score);
    // dbg!(&avg_ic_score);
}
