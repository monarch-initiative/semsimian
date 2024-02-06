use semsimian::{enums::SearchTypeEnum, Predicate, RustSemsimian, TermID};
use std::time::Instant;
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
    let include_similarity_object = false;

    rss.update_closure_and_ic_map();

    // Define input parameters for the function
    let assoc_predicate: HashSet<TermID> = HashSet::from(["biolink:has_phenotype".to_string()]);
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["MONDO:".to_string()]);

    let object_terms: HashSet<TermID> = HashSet::from([
        //* Alzheimer disease 2 profile
        // "HP:0002511".to_string(),
        // "HP:0002423".to_string(),
        // "HP:0002185".to_string(),
        // "HP:0001300".to_string(),
        // "HP:0000726".to_string(),
        //* Marfan syndrome
        "HP:0100775".to_string(),
        "HP:0003179".to_string(),
        "HP:0001083".to_string(),
        "HP:0000501".to_string(),
        "HP:0002705".to_string(),
        "HP:0004382".to_string(),
        "HP:0004326".to_string(),
        "HP:0002816".to_string(),
        "HP:0004298".to_string(),
        "HP:0002996".to_string(),
        "HP:0002808".to_string(),
        "HP:0002751".to_string(),
        "HP:0002647".to_string(),
        "HP:0002636".to_string(),
        "HP:0002616".to_string(),
        "HP:0002435".to_string(),
        "HP:0002360".to_string(),
        "HP:0007800".to_string(),
        "HP:0032934".to_string(),
        "HP:0012432".to_string(),
        "HP:0007720".to_string(),
        "HP:0002107".to_string(),
        "HP:0002105".to_string(),
        "HP:0007676".to_string(),
        "HP:0000939".to_string(),
        "HP:0000938".to_string(),
        "HP:0002097".to_string(),
        "HP:0012369".to_string(),
        "HP:0000767".to_string(),
        "HP:0000678".to_string(),
        "HP:0012019".to_string(),
        "HP:0010807".to_string(),
        "HP:0000577".to_string(),
        "HP:0000565".to_string(),
        "HP:0000545".to_string(),
        "HP:0000541".to_string(),
        "HP:0000494".to_string(),
        "HP:0000486".to_string(),
        "HP:0006687".to_string(),
        "HP:0007018".to_string(),
        "HP:0000278".to_string(),
        "HP:0000276".to_string(),
        "HP:0000275".to_string(),
        "HP:0000272".to_string(),
        "HP:0000268".to_string(),
        "HP:0000218".to_string(),
        "HP:0000189".to_string(),
        "HP:0000175".to_string(),
        "HP:0000098".to_string(),
        "HP:0000023".to_string(),
        "HP:0001635".to_string(),
        "HP:0001763".to_string(),
        "HP:0005294".to_string(),
        "HP:0003758".to_string(),
        "HP:0003326".to_string(),
        "HP:0003302".to_string(),
        "HP:0003202".to_string(),
        "HP:0003199".to_string(),
        "HP:0005059".to_string(),
        "HP:0003088".to_string(),
        "HP:0025586".to_string(),
        "HP:0005136".to_string(),
        "HP:0001761".to_string(),
        "HP:0001704".to_string(),
        "HP:0001765".to_string(),
        "HP:0001659".to_string(),
        "HP:0001653".to_string(),
        "HP:0001634".to_string(),
        "HP:0001533".to_string(),
        "HP:0001519".to_string(),
        "HP:0008132".to_string(),
        "HP:0001382".to_string(),
        "HP:0001371".to_string(),
        "HP:0001252".to_string(),
        "HP:0001166".to_string(),
        "HP:0001132".to_string(),
        "HP:0000347".to_string(),
        "HP:0001065".to_string(),
        "HP:0000490".to_string(),
        "HP:0000505".to_string(),
        "HP:0000518".to_string(),
        "HP:0000768".to_string(),
        "HP:0004970".to_string(),
        "HP:0004933".to_string(),
        "HP:0004927".to_string(),
        "HP:0002108".to_string(),
        "HP:0004872".to_string(),
        "HP:0012499".to_string(),
        "HP:0002650".to_string(),
    ]);
    let search_type_full: SearchTypeEnum = SearchTypeEnum::Full;
    let limit: usize = 10;

    // Start timing before the function call
    let start = Instant::now();

    //  Get the cache populated.
    rss.associations_search(
        &assoc_predicate,
        &object_terms,
        include_similarity_object,
        &None,
        &subject_prefixes,
        &search_type_full,
        Some(limit),
    );

    // Stop timing after the function call
    let duration = start.elapsed();
    // print the time:
    println!("Round 1 search time: {:?} sec", &duration);
    // Start timing before the function call
    let start = Instant::now();

    // Call the function under test
    let result = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        include_similarity_object,
        &None,
        &subject_prefixes,
        &search_type_full,
        Some(limit),
    );
    // Stop timing after the function call
    let duration = start.elapsed();

    assert_eq!({ result.len() }, limit);

    // dbg!(&result);
    // print the time:
    println!("Round 2 search time: {:?} sec", &duration);
}

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]
fn test_flat_vs_full_search() {
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
    let include_similarity_object = false;

    let mut rss = RustSemsimian::new(None, predicates, None, db_path.to_str());

    rss.update_closure_and_ic_map();

    // Define input parameters for the function
    let assoc_predicate: HashSet<TermID> = HashSet::from(["biolink:has_phenotype".to_string()]);
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["MONDO:".to_string()]);

    let object_terms: HashSet<TermID> = HashSet::from([
        //* Alzheimer disease 2 profile
        // "HP:0002511".to_string(),
        // "HP:0002423".to_string(),
        // "HP:0002185".to_string(),
        // "HP:0001300".to_string(),
        // "HP:0000726".to_string(),
        //* Marfan syndrome
        "HP:0100775".to_string(),
        "HP:0003179".to_string(),
        "HP:0001083".to_string(),
        "HP:0000501".to_string(),
        "HP:0002705".to_string(),
        "HP:0004382".to_string(),
        "HP:0004326".to_string(),
        "HP:0002816".to_string(),
        "HP:0004298".to_string(),
        "HP:0002996".to_string(),
        "HP:0002808".to_string(),
        "HP:0002751".to_string(),
        "HP:0002647".to_string(),
        "HP:0002636".to_string(),
        "HP:0002616".to_string(),
        "HP:0002435".to_string(),
        "HP:0002360".to_string(),
        "HP:0007800".to_string(),
        "HP:0032934".to_string(),
        "HP:0012432".to_string(),
        "HP:0007720".to_string(),
        "HP:0002107".to_string(),
        "HP:0002105".to_string(),
        "HP:0007676".to_string(),
        "HP:0000939".to_string(),
        "HP:0000938".to_string(),
        "HP:0002097".to_string(),
        "HP:0012369".to_string(),
        "HP:0000767".to_string(),
        "HP:0000678".to_string(),
        "HP:0012019".to_string(),
        "HP:0010807".to_string(),
        "HP:0000577".to_string(),
        "HP:0000565".to_string(),
        "HP:0000545".to_string(),
        "HP:0000541".to_string(),
        "HP:0000494".to_string(),
        "HP:0000486".to_string(),
        "HP:0006687".to_string(),
        "HP:0007018".to_string(),
        "HP:0000278".to_string(),
        "HP:0000276".to_string(),
        "HP:0000275".to_string(),
        "HP:0000272".to_string(),
        "HP:0000268".to_string(),
        "HP:0000218".to_string(),
        "HP:0000189".to_string(),
        "HP:0000175".to_string(),
        "HP:0000098".to_string(),
        "HP:0000023".to_string(),
        "HP:0001635".to_string(),
        "HP:0001763".to_string(),
        "HP:0005294".to_string(),
        "HP:0003758".to_string(),
        "HP:0003326".to_string(),
        "HP:0003302".to_string(),
        "HP:0003202".to_string(),
        "HP:0003199".to_string(),
        "HP:0005059".to_string(),
        "HP:0003088".to_string(),
        "HP:0025586".to_string(),
        "HP:0005136".to_string(),
        "HP:0001761".to_string(),
        "HP:0001704".to_string(),
        "HP:0001765".to_string(),
        "HP:0001659".to_string(),
        "HP:0001653".to_string(),
        "HP:0001634".to_string(),
        "HP:0001533".to_string(),
        "HP:0001519".to_string(),
        "HP:0008132".to_string(),
        "HP:0001382".to_string(),
        "HP:0001371".to_string(),
        "HP:0001252".to_string(),
        "HP:0001166".to_string(),
        "HP:0001132".to_string(),
        "HP:0000347".to_string(),
        "HP:0001065".to_string(),
        "HP:0000490".to_string(),
        "HP:0000505".to_string(),
        "HP:0000518".to_string(),
        "HP:0000768".to_string(),
        "HP:0004970".to_string(),
        "HP:0004933".to_string(),
        "HP:0004927".to_string(),
        "HP:0002108".to_string(),
        "HP:0004872".to_string(),
        "HP:0012499".to_string(),
        "HP:0002650".to_string(),
    ]);

    let search_type_flat: SearchTypeEnum = SearchTypeEnum::Flat;
    let search_type_full: SearchTypeEnum = SearchTypeEnum::Full;
    let limit: usize = 10;

    // Call the function under test
    let result_1 = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        include_similarity_object,
        &None,
        &subject_prefixes,
        &search_type_full,
        Some(limit),
    );
    // Call the function under test
    let result_2 = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        include_similarity_object,
        &None,
        &subject_prefixes,
        &search_type_flat,
        Some(limit),
    );

    dbg!(&result_1.len());
    dbg!(&result_2.len());

    let result_1_matches: Vec<&String> = result_1.iter().map(|(_, _, c)| c).collect();
    let result_2_matches: Vec<&String> = result_2.iter().map(|(_, _, c)| c).collect();

    let result_1_match_score: Vec<(&f64, &String)> =
        result_1.iter().map(|(a, _, c)| (a, c)).collect();
    let result_2_match_score: Vec<(&f64, &String)> =
        result_2.iter().map(|(a, _, c)| (a, c)).collect();

    // ! Writes the matches into a TSV file locally. ***********************
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("output.tsv").expect("Unable to create file");

    for ((result_1_score, result_1), (result_2_score, result_2)) in
        result_1_match_score.iter().zip(result_2_match_score.iter())
    {
        writeln!(
            file,
            "{}\t{}\t{}\t{}",
            result_1, result_1_score, result_2, result_2_score
        )
        .expect("Unable to write data");
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
    assert_eq!({ result_1.len() }, limit);
    assert_eq!({ result_2.len() }, limit);
    assert_eq!(match_percentage, 100.0);
    assert!(result_1_unique.is_empty(), "result_1_unique is not empty");
    assert!(result_2_unique.is_empty(), "result_2_unique is not empty");
}

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]
fn test_flat_vs_hybrid_search() {
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
    let include_similarity_object = false;
    let mut rss = RustSemsimian::new(None, predicates, None, db_path.to_str());

    rss.update_closure_and_ic_map();

    // Define input parameters for the function
    let assoc_predicate: HashSet<TermID> = HashSet::from(["biolink:has_phenotype".to_string()]);
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["MONDO:".to_string()]);
    // Alzheimer disease 2 profile
    let object_terms: HashSet<TermID> = HashSet::from([
        //* Alzheimer disease 2 profile
        // "HP:0002511".to_string(),
        // "HP:0002423".to_string(),
        // "HP:0002185".to_string(),
        // "HP:0001300".to_string(),
        // "HP:0000726".to_string(),
        //* Marfan syndrome
        "HP:0100775".to_string(),
        "HP:0003179".to_string(),
        "HP:0001083".to_string(),
        "HP:0000501".to_string(),
        "HP:0002705".to_string(),
        "HP:0004382".to_string(),
        "HP:0004326".to_string(),
        "HP:0002816".to_string(),
        "HP:0004298".to_string(),
        "HP:0002996".to_string(),
        "HP:0002808".to_string(),
        "HP:0002751".to_string(),
        "HP:0002647".to_string(),
        "HP:0002636".to_string(),
        "HP:0002616".to_string(),
        "HP:0002435".to_string(),
        "HP:0002360".to_string(),
        "HP:0007800".to_string(),
        "HP:0032934".to_string(),
        "HP:0012432".to_string(),
        "HP:0007720".to_string(),
        "HP:0002107".to_string(),
        "HP:0002105".to_string(),
        "HP:0007676".to_string(),
        "HP:0000939".to_string(),
        "HP:0000938".to_string(),
        "HP:0002097".to_string(),
        "HP:0012369".to_string(),
        "HP:0000767".to_string(),
        "HP:0000678".to_string(),
        "HP:0012019".to_string(),
        "HP:0010807".to_string(),
        "HP:0000577".to_string(),
        "HP:0000565".to_string(),
        "HP:0000545".to_string(),
        "HP:0000541".to_string(),
        "HP:0000494".to_string(),
        "HP:0000486".to_string(),
        "HP:0006687".to_string(),
        "HP:0007018".to_string(),
        "HP:0000278".to_string(),
        "HP:0000276".to_string(),
        "HP:0000275".to_string(),
        "HP:0000272".to_string(),
        "HP:0000268".to_string(),
        "HP:0000218".to_string(),
        "HP:0000189".to_string(),
        "HP:0000175".to_string(),
        "HP:0000098".to_string(),
        "HP:0000023".to_string(),
        "HP:0001635".to_string(),
        "HP:0001763".to_string(),
        "HP:0005294".to_string(),
        "HP:0003758".to_string(),
        "HP:0003326".to_string(),
        "HP:0003302".to_string(),
        "HP:0003202".to_string(),
        "HP:0003199".to_string(),
        "HP:0005059".to_string(),
        "HP:0003088".to_string(),
        "HP:0025586".to_string(),
        "HP:0005136".to_string(),
        "HP:0001761".to_string(),
        "HP:0001704".to_string(),
        "HP:0001765".to_string(),
        "HP:0001659".to_string(),
        "HP:0001653".to_string(),
        "HP:0001634".to_string(),
        "HP:0001533".to_string(),
        "HP:0001519".to_string(),
        "HP:0008132".to_string(),
        "HP:0001382".to_string(),
        "HP:0001371".to_string(),
        "HP:0001252".to_string(),
        "HP:0001166".to_string(),
        "HP:0001132".to_string(),
        "HP:0000347".to_string(),
        "HP:0001065".to_string(),
        "HP:0000490".to_string(),
        "HP:0000505".to_string(),
        "HP:0000518".to_string(),
        "HP:0000768".to_string(),
        "HP:0004970".to_string(),
        "HP:0004933".to_string(),
        "HP:0004927".to_string(),
        "HP:0002108".to_string(),
        "HP:0004872".to_string(),
        "HP:0012499".to_string(),
        "HP:0002650".to_string(),
    ]);

    let search_type_flat: SearchTypeEnum = SearchTypeEnum::Flat;
    let search_type_hybrid: SearchTypeEnum = SearchTypeEnum::Hybrid;
    let limit: usize = 10;

    // Call the function under test
    let result_1 = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        include_similarity_object,
        &None,
        &subject_prefixes,
        &search_type_flat,
        Some(limit),
    );
    // Call the function under test
    let result_2 = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        include_similarity_object,
        &None,
        &subject_prefixes,
        &search_type_hybrid,
        Some(limit),
    );

    dbg!(&result_1.len());
    dbg!(&result_2.len());

    let result_1_matches: Vec<&String> = result_1.iter().map(|(_, _, c)| c).collect();
    let result_2_matches: Vec<&String> = result_2.iter().map(|(_, _, c)| c).collect();

    let result_1_match_score: Vec<(&f64, &String)> =
        result_1.iter().map(|(a, _, c)| (a, c)).collect();
    let result_2_match_score: Vec<(&f64, &String)> =
        result_2.iter().map(|(a, _, c)| (a, c)).collect();

    // ! Writes the matches into a TSV file locally. ***********************
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("output.tsv").expect("Unable to create file");

    for ((result_1_score, result_1), (result_2_score, result_2)) in
        result_1_match_score.iter().zip(result_2_match_score.iter())
    {
        writeln!(
            file,
            "{}\t{}\t{}\t{}",
            result_1, result_1_score, result_2, result_2_score
        )
        .expect("Unable to write data");
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
    assert_eq!({ result_1.len() }, limit);
    assert_eq!({ result_2.len() }, limit);
    assert_eq!(match_percentage, 100.0);
    assert!(result_1_unique.is_empty(), "result_1_unique is not empty");
    assert!(result_2_unique.is_empty(), "result_2_unique is not empty");
}

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]
fn test_full_vs_hybrid_search() {
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
    let include_similarity_object = false;
    rss.update_closure_and_ic_map();

    // Define input parameters for the function
    let assoc_predicate: HashSet<TermID> = HashSet::from(["biolink:has_phenotype".to_string()]);
    let subject_prefixes: Option<Vec<TermID>> = Some(vec!["MONDO:".to_string()]);
    // Alzheimer disease 2 profile
    let object_terms: HashSet<TermID> = HashSet::from([
        //* Alzheimer disease 2 profile
        // "HP:0002511".to_string(),
        // "HP:0002423".to_string(),
        // "HP:0002185".to_string(),
        // "HP:0001300".to_string(),
        // "HP:0000726".to_string(),
        //* Marfan syndrome
        "HP:0100775".to_string(),
        "HP:0003179".to_string(),
        "HP:0001083".to_string(),
        "HP:0000501".to_string(),
        "HP:0002705".to_string(),
        "HP:0004382".to_string(),
        "HP:0004326".to_string(),
        "HP:0002816".to_string(),
        "HP:0004298".to_string(),
        "HP:0002996".to_string(),
        "HP:0002808".to_string(),
        "HP:0002751".to_string(),
        "HP:0002647".to_string(),
        "HP:0002636".to_string(),
        "HP:0002616".to_string(),
        "HP:0002435".to_string(),
        "HP:0002360".to_string(),
        "HP:0007800".to_string(),
        "HP:0032934".to_string(),
        "HP:0012432".to_string(),
        "HP:0007720".to_string(),
        "HP:0002107".to_string(),
        "HP:0002105".to_string(),
        "HP:0007676".to_string(),
        "HP:0000939".to_string(),
        "HP:0000938".to_string(),
        "HP:0002097".to_string(),
        "HP:0012369".to_string(),
        "HP:0000767".to_string(),
        "HP:0000678".to_string(),
        "HP:0012019".to_string(),
        "HP:0010807".to_string(),
        "HP:0000577".to_string(),
        "HP:0000565".to_string(),
        "HP:0000545".to_string(),
        "HP:0000541".to_string(),
        "HP:0000494".to_string(),
        "HP:0000486".to_string(),
        "HP:0006687".to_string(),
        "HP:0007018".to_string(),
        "HP:0000278".to_string(),
        "HP:0000276".to_string(),
        "HP:0000275".to_string(),
        "HP:0000272".to_string(),
        "HP:0000268".to_string(),
        "HP:0000218".to_string(),
        "HP:0000189".to_string(),
        "HP:0000175".to_string(),
        "HP:0000098".to_string(),
        "HP:0000023".to_string(),
        "HP:0001635".to_string(),
        "HP:0001763".to_string(),
        "HP:0005294".to_string(),
        "HP:0003758".to_string(),
        "HP:0003326".to_string(),
        "HP:0003302".to_string(),
        "HP:0003202".to_string(),
        "HP:0003199".to_string(),
        "HP:0005059".to_string(),
        "HP:0003088".to_string(),
        "HP:0025586".to_string(),
        "HP:0005136".to_string(),
        "HP:0001761".to_string(),
        "HP:0001704".to_string(),
        "HP:0001765".to_string(),
        "HP:0001659".to_string(),
        "HP:0001653".to_string(),
        "HP:0001634".to_string(),
        "HP:0001533".to_string(),
        "HP:0001519".to_string(),
        "HP:0008132".to_string(),
        "HP:0001382".to_string(),
        "HP:0001371".to_string(),
        "HP:0001252".to_string(),
        "HP:0001166".to_string(),
        "HP:0001132".to_string(),
        "HP:0000347".to_string(),
        "HP:0001065".to_string(),
        "HP:0000490".to_string(),
        "HP:0000505".to_string(),
        "HP:0000518".to_string(),
        "HP:0000768".to_string(),
        "HP:0004970".to_string(),
        "HP:0004933".to_string(),
        "HP:0004927".to_string(),
        "HP:0002108".to_string(),
        "HP:0004872".to_string(),
        "HP:0012499".to_string(),
        "HP:0002650".to_string(),
    ]);

    let search_type_full: SearchTypeEnum = SearchTypeEnum::Full;
    let search_type_hybrid: SearchTypeEnum = SearchTypeEnum::Hybrid;
    let limit: usize = 100;

    // Call the function under test
    let result_1 = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        include_similarity_object,
        &None,
        &subject_prefixes,
        &search_type_full,
        Some(limit),
    );
    // Call the function under test
    let result_2 = rss.associations_search(
        &assoc_predicate,
        &object_terms,
        include_similarity_object,
        &None,
        &subject_prefixes,
        &search_type_hybrid,
        Some(limit),
    );

    dbg!(&result_1.len());
    dbg!(&result_2.len());

    let result_1_matches: Vec<&String> = result_1.iter().map(|(_, _, c)| c).collect();
    let result_2_matches: Vec<&String> = result_2.iter().map(|(_, _, c)| c).collect();

    let result_1_match_score: Vec<(&f64, &String)> =
        result_1.iter().map(|(a, _, c)| (a, c)).collect();
    let result_2_match_score: Vec<(&f64, &String)> =
        result_2.iter().map(|(a, _, c)| (a, c)).collect();

    // ! Writes the matches into a TSV file locally. ***********************
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("output.tsv").expect("Unable to create file");

    // Write the column names first
    writeln!(file, "Full\tScore\tHybrid\tScore").expect("Unable to write column names");

    for ((result_1_score, result_1), (result_2_score, result_2)) in
        result_1_match_score.iter().zip(result_2_match_score.iter())
    {
        writeln!(
            file,
            "{}\t{}\t{}\t{}",
            result_1, result_1_score, result_2, result_2_score
        )
        .expect("Unable to write data");
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

    dbg!(&result_1_unique);
    dbg!(&result_2_unique);
    dbg!(&result_1_matches);
    dbg!(&result_2_matches);
    assert_eq!({ result_1.len() }, limit);
    assert_eq!({ result_2.len() }, limit);
    assert_eq!(match_percentage, 100.0);
    assert!(result_1_unique.is_empty(), "result_1_unique is not empty");
    assert!(result_2_unique.is_empty(), "result_2_unique is not empty");
}
