use std::{collections::HashSet, path::PathBuf};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use semsimian::{enums::SearchTypeEnum, Predicate, RustSemsimian, TermID};

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
    let subject_prefixes: Option<Vec<TermID>> = black_box(Some(vec!["MONDO:".to_string()]));
    // Alzheimer disease 2 profile
    let object_terms: HashSet<TermID> = black_box(HashSet::from([
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
    ]));
    let search_type: SearchTypeEnum = SearchTypeEnum::Full;
    let include_similarity_object = false;

    let associations = rss.get_or_set_prefix_expansion_cache(
        &assoc_predicate,
        &None,
        &subject_prefixes,
        &search_type,
    );

    let mut bench_grp = c.benchmark_group("search_bench_group");
    bench_grp.sample_size(10);
    // .measurement_time(Duration::from_secs(300));
    bench_grp.bench_function("search_similarity", move |b| {
        b.iter(|| {
            rss.calculate_similarity_for_association_search(
                &associations,
                &object_terms,
                include_similarity_object,
            )
        })
    });
    bench_grp.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
