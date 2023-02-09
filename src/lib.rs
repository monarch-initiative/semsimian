#![feature(format_args_capture)]

use std::path::Path;

use pyo3::prelude::*;
mod file_io; use file_io::{read_file, parse_associations};
mod similarity; use similarity::calculate_jaccard_similarity;
mod closures; use closures::expand_terms_using_closure;
mod structs; use structs::TermSetPairwiseSimilarity;


#[pyfunction]
fn run(input_file:&str, closure_file:&str) -> PyResult<()>{
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
    tsps_information.original_subject_termset = ref_set.clone();
    tsps_information.subject_termset = expand_terms_using_closure
                                        (
                                            &tsps_information.original_subject_termset,
                                            &closures_dict
                                        );
    // iterate over dict
    for (key, terms) in &data_dict {
        tsps_information.set_id = key.to_string();
        tsps_information.original_object_termset = terms.clone();
        tsps_information.object_termset = expand_terms_using_closure
                                        (
                                            &tsps_information.original_object_termset,
                                            &closures_dict
                                        );
        tsps_information.jaccard_similarity = calculate_jaccard_similarity
                                        (
                                            &tsps_information.subject_termset,
                                            &tsps_information.object_termset
                                        );
        println!("{tsps_information:#?}")
        
    }
    Ok(())
}

#[pymodule] 
fn rustsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(()) 
}
