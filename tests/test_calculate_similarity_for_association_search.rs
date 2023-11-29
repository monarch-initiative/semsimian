use semsimian::{enums::SearchTypeEnum, Predicate, RustSemsimian, TermID};
use std::{collections::HashSet, path::PathBuf};

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]

fn test_calculate_similarity_for_association_search() {
    let mut db_path = PathBuf::new();
    if let Some(home) = std::env::var_os("HOME") {
        db_path.push(home);
        db_path.push(".data/oaklib/phenio.db");
    } else {
        panic!("Failed to get home directory");
    }
    let db = Some(db_path.to_str().expect("Failed to convert path to string"));

    let predicates: Option<Vec<Predicate>> = Some(vec![
        "rdfs:subClassOf".to_string(),
        "BFO:0000050".to_string(),
    ]);
    let mut rss = RustSemsimian::new(None, predicates, None, db);

    rss.update_closure_and_ic_map();
    let object_closure_predicates: HashSet<TermID> =
        HashSet::from(["biolink:has_phenotype".to_string()]);
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["MONDO:".to_string()]);

    let profile_entities: HashSet<TermID> = HashSet::from([
        "HP:0008132".to_string(),
        "HP:0000189".to_string(),
        "HP:0000275".to_string(),
        "HP:0000276".to_string(),
        "HP:0000278".to_string(),
        "HP:0000347".to_string(),
        "HP:0001371".to_string(),
        "HP:0000501".to_string(),
        "HP:0000541".to_string(),
        "HP:0000098".to_string(),
    ]);
    let search_type: SearchTypeEnum = SearchTypeEnum::Full;

    let associations = match rss.get_prefix_expansion_cache(
        &object_closure_predicates,
        &None,
        &subject_prefixes,
        &search_type,
    ) {
        Some(value) => value, // If the value was found, use it
        None => {
            // If the value was not found, set it

            rss.set_prefix_expansion_cache(
                &object_closure_predicates,
                &None,
                &subject_prefixes,
                &search_type,
            )
        }
    };

    let include_similarity_object = false;

    // Call the method with the test data
    let result = rss.calculate_similarity_for_association_search(
        &associations,
        &profile_entities,
        include_similarity_object,
    );
    dbg!(&result.len());
    let expected_len = 10187;

    // Assert that the result is as expected
    assert_eq!(&result.len(), &(expected_len as usize));
}
