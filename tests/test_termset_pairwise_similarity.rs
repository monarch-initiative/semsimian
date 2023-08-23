use std::{path::PathBuf, collections::HashSet};
use semsimian::{Predicate, RustSemsimian, TermID};

#[test]
#[cfg_attr(feature = "ci", ignore)]
fn test_termset_pairwise_similarity() {
    let mut db_path = PathBuf::new();
    if let Some(home) = std::env::var_os("HOME") {
        db_path.push(home);
        db_path.push(".data/oaklib/phenio.db");
    } else {
        panic!("Failed to get home directory");
    }
    // let db = Some("//Users/HHegde/.data/oaklib/phenio.db");
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


    rss.termset_pairwise_similarity(&entity1, &entity2, &None);

}
