use std::{path::Path, collections::{HashMap, HashSet}};

use generator::{done, Generator, Gn}; //https://crates.io/crates/generator
use pyo3::prelude::*;
mod file_io; use file_io::{read_file, parse_associations};
mod similarity; use similarity::calculate_jaccard_similarity;
mod closures; use closures::expand_terms_using_closure;
mod structs; use structs::TermSetPairwiseSimilarity;

// Generator<'a, (), & 'a mut TermSetPairwiseSimilarity>
#[pyfunction]
fn run <'a>(input_file:&str, closure_file:&str) -> PyResult<Vec<TermSetPairwiseSimilarity>>{
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
    let mut tsps_vector:Vec<TermSetPairwiseSimilarity> = Vec::new();
    for tsps in iter_tsps(data_dict, closures_dict, tsps_information){
        // println!("{tsps:#?}");
        //TODO: "yield" tsps instead of just printing.
        tsps_vector.push(tsps);
    }
    Ok(tsps_vector)
}

fn iter_tsps <'a>(
    data_dict:HashMap<String, HashSet<String>>,
    closures_dict:HashMap<String, HashSet<String>>,
    tsps_info:TermSetPairwiseSimilarity,
) -> Generator<'a, (), TermSetPairwiseSimilarity> {
    // iterate over dict
    Gn::new_scoped(move |mut s| {
        for (key, terms) in data_dict {
            let mut tsps:TermSetPairwiseSimilarity = tsps_info.clone();
            tsps.set_id = key.to_string();
            tsps.original_object_termset = terms.clone();
            tsps.object_termset = expand_terms_using_closure
                                            (
                                                &tsps_info.original_object_termset,
                                                &closures_dict
                                            );
            tsps.jaccard_similarity = calculate_jaccard_similarity
                                            (
                                                &tsps_info.subject_termset,
                                                &tsps_info.object_termset
                                            );
            s.yield_(tsps);
        }
        done!();
    })
}


#[pymodule] 
fn rustsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(()) 
}

//TODO: Test the lib module.