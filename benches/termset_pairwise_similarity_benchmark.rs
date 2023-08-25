use std::{path::PathBuf, collections::HashSet};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use semsimian::{Predicate, RustSemsimian, TermID};

fn criterion_benchmark(c: &mut Criterion) {
    let mut db_path = PathBuf::new();
    if let Some(home) = std::env::var_os("HOME") {
        db_path.push(home);
        db_path.push(".data/oaklib/phenio.db");
    } else {
        panic!("Failed to get home directory");
    }
    let db = black_box(Some(db_path.to_str().expect("Failed to convert path to string")));
    
    let predicates: Option<Vec<Predicate>> = black_box(Some(vec![
        "rdfs:subClassOf".to_string(),
        "BFO:0000050".to_string(),
        "UPHENO:0000001".to_string(),
    ]));

    let mut rss = black_box(RustSemsimian::new(None, predicates, None, db));
    rss.update_closure_and_ic_map();
    let entity1: HashSet<TermID> = black_box(HashSet::from([
        "MP:0010771".to_string(),
        "MP:0002169".to_string(),
        "MP:0005391".to_string(),
        "MP:0005389".to_string(),
        "MP:0005367".to_string(),
    ]));
    let entity2: HashSet<TermID> = black_box(HashSet::from([
        "HP:0004325".to_string(),
        "HP:0000093".to_string(),
        "MP:0006144".to_string(),
    ]));
    
    c.bench_function("tsps", |b| b.iter(|| {rss.termset_pairwise_similarity(&entity1, &entity2, &None)}).sample_size(10));

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);