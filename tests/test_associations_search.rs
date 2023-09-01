use semsimian::{Predicate, RustSemsimian, TermID};
use std::{collections::HashSet, path::PathBuf};

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]
fn test_associations_search() {
    let mut db_path = PathBuf::new();
    if let Some(home) = std::env::var_os("HOME") {
        db_path.push(home);
        db_path.push(".data/oaklib/phenio.db");
    } else {
        panic!("Failed to get home directory");
    }

    let predicates: Option<Vec<Predicate>> = Some(vec![
        "rdfs:subClassOf".to_string(),
        "BFO:0000050".to_string(),
        "UPHENO:0000001".to_string(),
    ]);

    let mut rss = RustSemsimian::new(None, predicates, None, db_path.to_str());

    rss.update_closure_and_ic_map();

    // Define input parameters for the function
    let assoc_predicate: HashSet<TermID> = HashSet::from(["biolink:has_phenotype".to_string()]);
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["MGI:".to_string()]);
    let object_terms: HashSet<TermID> = HashSet::from(["MP:0003143".to_string()]);
    let limit: Option<usize> = Some(10);

    // Call the function under test
    let result = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        true,
        &None,
        &subject_prefixes,
        limit,
    );

    assert_eq!(result.len() as usize, limit.unwrap());

    dbg!(&result);
}
