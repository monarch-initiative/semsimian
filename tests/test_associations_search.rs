use semsimian::{Predicate, RustSemsimian, TermID};
use std::{collections::HashSet, path::PathBuf};

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]
fn test_large_associations_search() {
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
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["MONDO:".to_string()]);
     // Alzheimer disease 2 profile
    let object_terms: HashSet<TermID> = HashSet::from(
        [
            "HP:0002511".to_string(),
            "HP:0002423".to_string(),
            "HP:0002185".to_string(),
            "HP:0001300".to_string(),
            "HP:0000726".to_string()
        ]
    );
    let limit: Option<usize> = Some(10);

    // Call the function under test
    let result = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        true,
        &None,
        &subject_prefixes,
        false,
        limit,
    );

    assert_eq!({ result.len() }, limit.unwrap());

    // dbg!(&result);
}

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]
fn test_large_associations_quick_search() {
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
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["MONDO:".to_string()]);
    // Alzheimer disease 2 profile
    let object_terms: HashSet<TermID> = HashSet::from(
        [
            "HP:0002511".to_string(),
            "HP:0002423".to_string(),
            "HP:0002185".to_string(),
            "HP:0001300".to_string(),
            "HP:0000726".to_string()
        ]
    );
    
    let limit: Option<usize> = Some(10);

    // Call the function under test
    let result_1 = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        true,
        &None,
        &subject_prefixes,
        false,
        limit,
    );
    // Call the function under test
    let result_2 = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        true,
        &None,
        &subject_prefixes,
        true,
        limit,
    );

    dbg!(&result_1.len());
    dbg!(&result_2.len());


    let result_1_matches: Vec<&String> = result_1.iter().map(|(_, _, c)| c).collect();
    let result_2_matches: Vec<&String> = result_2.iter().map(|(_, _, c)| c).collect();

    let result_1_match_score: Vec<(&f64, &String)> = result_1.iter().map(|(a, _, c)| (a,c)).collect();
    let result_2_match_score: Vec<(&f64, &String)> = result_2.iter().map(|(a, _, c)| (a,c)).collect();

    // ! Writes the matches into a TSV file. SHOULD BE COMMENTED OUT!!!*****
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create("output.tsv").expect("Unable to create file");
    
    for ((result_1_score, result_1), (result_2_score, result_2)) in result_1_match_score.iter().zip(result_2_match_score.iter()) {
        writeln!(file, "{}\t{}\t{}\t{}", result_1,result_1_score, result_2, result_2_score).expect("Unable to write data");
    }
    // ! ********************************************************************
    

    // Assert that there is a full match between result_1_matches and result_2_matches
    let match_count = result_1_matches
        .iter()
        .filter(|&x| result_2_matches.contains(x))
        .count();
    let match_percentage = (match_count as f32 / result_1_matches.len() as f32) * 100.0;

    dbg!(&match_percentage);

    // ! Double check there aren't terms in one and not the other
    let result_1_unique: Vec<_> = result_1_matches
        .iter()
        .filter(|&x| !result_2_matches.contains(x))
        .cloned()
        .collect();

    let result_2_unique: Vec<_> = result_2_matches
        .iter()
        .filter(|&x| !result_1_matches.contains(x))
        .cloned()
        .collect();

    // dbg!(&result_1_unique);
    // dbg!(&result_2_unique);
    // dbg!(&result_1_matches);
    // dbg!(&result_2_matches);
    assert_eq!({ result_1.len() }, limit.unwrap());
    assert_eq!({ result_2.len() }, limit.unwrap());
    assert_eq!(match_percentage, 100.0);
    assert!(result_1_unique.is_empty(), "result_1_unique is not empty");
    assert!(result_2_unique.is_empty(), "result_2_unique is not empty");
}
