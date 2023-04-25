use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use generator::{done, Generator, Gn}; //https://crates.io/crates/generator
use pyo3::prelude::*;
mod file_io;
pub mod utils;
use file_io::{parse_associations, read_file};
pub mod similarity;
use similarity::{
    calculate_jaccard_similarity, calculate_phenomizer_score,
    calculate_semantic_jaccard_similarity, get_most_recent_common_ancestor_with_score,
};
pub mod closures;
use closures::expand_terms_using_closure;
pub mod structs;
use structs::TermSetPairwiseSimilarity;
pub mod ancestors;
use ancestors::get_intersection_between_sets;
use utils::{convert_list_of_tuples_to_hashmap, numericize_sets};

// Generator<'a, (), & 'a mut TermSetPairwiseSimilarity>
#[pyfunction]
fn run<'a>(input_file: &str, closure_file: &str) -> PyResult<Vec<TermSetPairwiseSimilarity>> {
    /*
    read in TSV file
    csv::ReaderBuilder instead of just csv::Reader because we need to specify
    that the file has no headers.

    test e.g.:
    input_file = "test_set.tsv"
    closure_file = "closures.tsv"
    */

    let data_dict = parse_associations(read_file(Path::new(input_file)));
    let closures_dict = parse_associations(read_file(Path::new(closure_file)));
    let ref_set = data_dict.get("set1").unwrap();
    let mut tsps_information = TermSetPairwiseSimilarity::new();
    let original_subject_termset = ref_set.clone();
    tsps_information.subject_termset =
        expand_terms_using_closure(&original_subject_termset, &closures_dict);
    let mut tsps_vector: Vec<TermSetPairwiseSimilarity> = Vec::new();
    for tsps in iter_tsps(data_dict, closures_dict, tsps_information) {
        // println!("{tsps:#?}");
        tsps_vector.push(tsps);
    }
    Ok(tsps_vector)
}

fn iter_tsps<'a>(
    data_dict: HashMap<String, HashSet<String>>,
    closures_dict: HashMap<String, HashSet<String>>,
    tsps_info: TermSetPairwiseSimilarity,
) -> Generator<'a, (), TermSetPairwiseSimilarity> {
    // iterate over dict
    Gn::new_scoped(move |mut s| {
        for (key, terms) in data_dict {
            let mut tsps: TermSetPairwiseSimilarity = tsps_info.clone();
            tsps.set_id = key.to_string();
            let original_object_termset = terms.clone();
            tsps.object_termset =
                expand_terms_using_closure(&original_object_termset, &closures_dict);
            let (num_tsps_subj_terms, num_tsps_object_terms, _) =
                numericize_sets(&tsps.subject_termset, &tsps.object_termset);
            tsps.best_score =
                calculate_jaccard_similarity(&num_tsps_subj_terms, &num_tsps_object_terms);
            s.yield_(tsps);
        }
        done!();
    })
}

#[pyfunction]
fn jaccard_similarity(set1: HashSet<String>, set2: HashSet<String>) -> PyResult<f64> {
    let (num_set1, num_set2, _) = numericize_sets(&set1, &set2);
    Ok(calculate_jaccard_similarity(&num_set1, &num_set2))
}

#[pyfunction]
fn mrca_and_score(map: HashMap<String, f64>) -> PyResult<(String, f64)> {
    Ok(get_most_recent_common_ancestor_with_score(map))
}

#[pyfunction]
fn get_intersection(set1: HashSet<String>, set2: HashSet<String>) -> PyResult<HashSet<String>> {
    let mut result = HashSet::new();
    for a in get_intersection_between_sets(&set1, &set2).into_iter() {
        result.insert(a.to_string());
    }
    Ok(result)
}

#[pyfunction]
fn semantic_jaccard_similarity(
    closure_table: HashMap<String, HashMap<String, HashSet<String>>>,
    entity1: String,
    entity2: String,
    predicates: HashSet<String>,
) -> PyResult<f64> {
    Ok(calculate_semantic_jaccard_similarity(
        &closure_table,
        entity1,
        entity2,
        &predicates,
    ))
}

#[pyfunction]
fn relationships_to_closure_table(
    list_of_tuples: Vec<(String, String, String)>,
) -> PyResult<HashMap<String, HashMap<String, HashSet<String>>>> {
    Ok(convert_list_of_tuples_to_hashmap(list_of_tuples))
}

#[pyfunction]
fn phenomizer_score(
    map: HashMap<String, HashMap<String, f64>>,
    entity1: HashSet<String>,
    entity2: HashSet<String>,
) -> PyResult<f64> {
    Ok(calculate_phenomizer_score(map, entity1, entity2))
}

#[pymodule]
fn rustsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(mrca_and_score, m)?)?;
    m.add_function(wrap_pyfunction!(get_intersection, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(relationships_to_closure_table, m)?)?;
    m.add_function(wrap_pyfunction!(phenomizer_score, m)?)?;
    Ok(())
}

//TODO: Test the lib module.
