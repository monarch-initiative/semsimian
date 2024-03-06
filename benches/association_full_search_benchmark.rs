use std::{collections::HashSet, path::PathBuf};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use semsimian::{
    enums::{MetricEnum, SearchTypeEnum},
    Predicate, RustSemsimian, TermID,
};

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

    let mut rss = black_box(RustSemsimian::new(None, predicates, None, db));
    rss.update_closure_and_ic_map();

    let assoc_predicate: HashSet<TermID> =
        black_box(HashSet::from(["biolink:has_phenotype".to_string()]));

    let subject_prefixes: Option<Vec<TermID>> = black_box(Some(vec!["MONDO:".to_string()]));
    let include_similarity_object = false;
    let object_terms: HashSet<TermID> = black_box(HashSet::from([
        //* */ Alzheimer disease 2 profile
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
    ]));
    let search_type: SearchTypeEnum = SearchTypeEnum::Full;
    let limit: Option<usize> = black_box(Some(10));
    let score_metric = MetricEnum::AncestorInformationContent;

    let mut bench_grp = c.benchmark_group("search_bench_group");
    bench_grp.sample_size(10);
    // .measurement_time(Duration::from_secs(300));
    bench_grp.bench_function("search", move |b| {
        b.iter(|| {
            rss.associations_search(
                &assoc_predicate,
                &object_terms,
                include_similarity_object,
                &None,
                &subject_prefixes,
                &search_type,
                &score_metric,
                limit,
            )
        })
    });
    bench_grp.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
