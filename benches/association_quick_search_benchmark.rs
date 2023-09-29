use std::{collections::HashSet, path::PathBuf};

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
    let db = black_box(Some(
        db_path.to_str().expect("Failed to convert path to string"),
    ));

    let predicates: Option<Vec<Predicate>> = black_box(Some(vec![
        "rdfs:subClassOf".to_string(),
        "BFO:0000050".to_string(),
        "UPHENO:0000001".to_string(),
    ]));

    let mut rss = RustSemsimian::new(None, predicates, None, db);
    rss.update_closure_and_ic_map();

    let assoc_predicate: HashSet<TermID> =
        black_box(HashSet::from(["biolink:has_phenotype".to_string()]));
    let subject_prefixes: Option<Vec<TermID>> = black_box(Some(vec!["MGI:".to_string()]));
     // Alzheimer disease 2 profile
    let object_terms: HashSet<TermID> = black_box(HashSet::from(
        [
            "HP:0002511".to_string(),
            "HP:0002423".to_string(),
            "HP:0002185".to_string(),
            "HP:0001300".to_string(),
            "HP:0000726".to_string()
        ]
    ));
    let limit: Option<usize> = black_box(Some(10));

    let mut bench_grp = c.benchmark_group("search_bench_group");
    bench_grp.sample_size(10);
    // .measurement_time(Duration::from_secs(300));
    bench_grp.bench_function("quick_search", move |b| {
        b.iter(|| {
            rss.associations_search(
                &assoc_predicate,
                &object_terms,
                true,
                &None,
                &subject_prefixes,
                true,
                limit,
            )
        })
    });
    bench_grp.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
