use semsimian::{enums::MetricEnum, Predicate, RustSemsimian, TermID};
use std::{collections::HashSet, path::PathBuf};

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]
fn test_large_termset_pairwise_similarity() {
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
        "UPHENO:0000001".to_string(),
    ]);

    let mut rss = RustSemsimian::new(None, predicates, None, db);
    rss.update_closure_and_ic_map();

    let entity1: HashSet<TermID> = HashSet::from([
        "MP:0010771".to_string(),
        "MP:0002169".to_string(),
        "MP:0005391".to_string(),
        "MP:0005389".to_string(),
        "MP:0005367".to_string(),
    ]);
    let entity2: HashSet<TermID> = HashSet::from([
        "HP:0004325".to_string(),
        "HP:0000093".to_string(),
        "MP:0006144".to_string(),
    ]);
    let score_metric = MetricEnum::AncestorInformationContent;
    // let score_metric = MetricEnum::PhenodigmScore;
    // let score_metric = MetricEnum::JaccardSimilarity;

    // let entity1: HashSet<TermID> = HashSet::from([
    //     "HP:0003394".to_string(),
    //     "HP:0003771".to_string(),
    //     "HP:0012378".to_string(),
    //     "HP:0012450".to_string(),
    //     "HP:0000974".to_string(),
    //     "HP:0001027".to_string(),
    //     "HP:0001030".to_string(),
    //     "HP:0001065".to_string(),
    //     "HP:0001073".to_string(),
    //     "HP:0001075".to_string(),
    //     "HP:0002761".to_string(),
    //     "HP:0001386".to_string(),
    //     "HP:0001537".to_string(),
    //     "HP:0001622".to_string(),
    //     "HP:0001760".to_string(),
    //     "HP:0001762".to_string(),
    //     "HP:0001763".to_string(),
    //     "HP:0001788".to_string(),
    //     "HP:0002035".to_string(),
    //     "HP:0002036".to_string(),
    //     "HP:0002616".to_string(),
    //     "HP:0002650".to_string(),
    //     "HP:0002758".to_string(),
    //     "HP:0002827".to_string(),
    //     "HP:0002829".to_string(),
    //     "HP:0002999".to_string(),
    //     "HP:0003010".to_string(),
    //     "HP:0003083".to_string(),
    //     "HP:0003834".to_string(),
    //     "HP:0000938".to_string(),
    //     "HP:0001058".to_string(),
    //     "HP:0001252".to_string(),
    //     "HP:0001324".to_string(),
    //     "HP:0002013".to_string(),
    //     "HP:0002018".to_string(),
    //     "HP:0002020".to_string(),
    //     "HP:0000015".to_string(),
    //     "HP:0000023".to_string(),
    //     "HP:0000139".to_string(),
    //     "HP:0000286".to_string(),
    //     "HP:0000481".to_string(),
    //     "HP:0000978".to_string(),
    //     "HP:0000993".to_string(),
    //     "HP:0001063".to_string(),
    //     "HP:0004872".to_string(),
    //     "HP:0004944".to_string(),
    //     "HP:0004947".to_string(),
    //     "HP:0001270".to_string(),
    //     "HP:0005294".to_string(),
    //     "HP:0006243".to_string(),
    //     "HP:0007495".to_string(),
    //     "HP:0009763".to_string(),
    //     "HP:0010749".to_string(),
    //     "HP:0010750".to_string(),
    //     "HP:0010754".to_string(),
    //     "HP:0025014".to_string(),
    //     "HP:0025019".to_string(),
    //     "HP:0025509".to_string(),
    //     "HP:0030009".to_string(),
    //     "HP:0031364".to_string(),
    //     "HP:0031653".to_string(),
    //     "HP:0001278".to_string(),
    //     "HP:0001634".to_string(),
    //     "HP:0001653".to_string(),
    //     "HP:0001704".to_string(),
    //     "HP:0002315".to_string(),
    // ]);

    // let entity2: HashSet<TermID> = HashSet::from([
    //     "HP:0003645".to_string(),
    //     "HP:0005261".to_string(),
    //     "HP:0002758".to_string(),
    //     "HP:0003125".to_string(),
    //     "HP:0001892".to_string(),
    //     "HP:0001934".to_string(),
    //     "HP:0000967".to_string(),
    //     "HP:0000978".to_string(),
    //     "HP:0000979".to_string(),
    //     "HP:0040242".to_string(),
    //     "HP:0007420".to_string(),
    //     "HP:0030140".to_string(),
    //     "HP:0001386".to_string(),
    //     "HP:0002829".to_string(),
    //     "HP:0002239".to_string(),
    //     "HP:0011889".to_string(),
    //     "HP:0001907".to_string(),
    //     "HP:0012223".to_string(),
    //     "HP:0009811".to_string(),
    //     "HP:0012233".to_string(),
    //     "HP:0030746".to_string(),
    //     "HP:0002170".to_string(),
    // ]);

    let result = rss.termset_pairwise_similarity(&entity1, &entity2, &score_metric);
    dbg!(&result);
}

#[test]
#[ignore]
#[cfg_attr(feature = "ci", ignore)]
fn test_ancestor_label_presence() {
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
        "UPHENO:0000001".to_string(),
    ]);

    let mut rss = RustSemsimian::new(None, predicates, None, db);
    rss.update_closure_and_ic_map();

    let entity1: HashSet<TermID> = HashSet::from([
        "MP:0010771".to_string(),
        "MP:0002169".to_string(),
        "MP:0005391".to_string(),
        "MP:0005389".to_string(),
        "MP:0005367".to_string(),
    ]);
    let entity2: HashSet<TermID> = HashSet::from([
        "HP:0004325".to_string(),
        "HP:0000093".to_string(),
        "MP:0006144".to_string(),
    ]);
    let score_metric = MetricEnum::AncestorInformationContent;

    let result = rss.termset_pairwise_similarity(&entity1, &entity2, &score_metric);

    dbg!(&result);
    for value in result.subject_best_matches_similarity_map.values() {
        assert!(
            value.get("ancestor_label").is_some(),
            "Ancestor label in subject_best_matches_similarity_map should not be None"
        );
        assert!(
            !value.get("ancestor_label").as_ref().unwrap().is_empty(),
            "Ancestor label in subject_best_matches_similarity_map should not be empty"
        );
    }

    for value in result.object_best_matches_similarity_map.values() {
        assert!(
            value.get("ancestor_label").is_some(),
            "Ancestor label in object_best_matches_similarity_map should not be None"
        );
        assert!(
            !value.get("ancestor_label").as_ref().unwrap().is_empty(),
            "Ancestor label in object_best_matches_similarity_map should not be empty"
        );
    }
}
