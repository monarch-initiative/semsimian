use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod file_io;
pub mod utils;
use file_io::{parse_associations, read_file};
pub mod similarity;
use similarity::{
    calculate_jaccard_similarity, calculate_max_information_content, calculate_phenomizer_score,
    calculate_semantic_jaccard_similarity, get_most_recent_common_ancestor_with_score,
};
pub mod closures;
use closures::expand_terms_using_closure;
pub mod structs;
use structs::TermSetPairwiseSimilarity;
pub mod ancestors;
use ancestors::get_intersection_between_sets;
use utils::{convert_list_of_tuples_to_hashmap, numericize_sets};


#[derive(Default)]
pub struct RustSemsimian {
    ic_map: HashMap<String, f64>,
    closure_map: HashMap<String, HashMap<String, HashSet<String>>>,
}

impl RustSemsimian {
    pub fn new() -> RustSemsimian {
        RustSemsimian {
            ic_map: HashMap::new(),
            closure_map: HashMap::new(),
        }
    }

    pub fn jaccard_similarity(&self, term1: &str, term2: &str) -> f64 {
        let term1_set = self.get_closure(term1);
        let term2_set = self.get_closure(term2);
        let intersection = term1_set.intersection(&term2_set).count() as f64;
        let union = term1_set.union(&term2_set).count() as f64;
        intersection / union
    }

    pub fn information_content(&self, term: &str) -> f64 {
        let ic = self.ic_map.get(term).unwrap_or(&0.0);
        *ic
    }

    fn get_closure(&self, term: &str) -> HashSet<String> {
        let mut closure = HashSet::new();
        let mut stack = vec![term.to_string()];
        while let Some(t) = stack.pop() {
            if !closure.contains(&t) {
                closure.insert(t.clone());
                if let Some(ancestors) = self.closure_map.get(&t) {
                    for parent in ancestors.keys() {
                        stack.push(parent.clone());
                    }
                }
            }
        }
        closure
    }
}

#[pyclass]
pub struct Semsimian {
    ss: RustSemsimian,
}

#[pymethods]
impl Semsimian {
    #[new]
    fn new() -> PyResult<Self> {
        let ss = RustSemsimian::new();
        Ok(Semsimian { ss })
    }

    fn jaccard_similarity(&self, term1: &str, term2: &str) -> PyResult<f64> {
        Ok(self.ss.jaccard_similarity(term1, term2))
    }

    fn information_content(&self, term: &str) -> PyResult<f64> {
        Ok(self.ss.information_content(term))
    }
}

#[pymodule]
fn semsimian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Semsimian>()?;
    Ok(())
}


#[pyfunction]
fn mrca_and_score(map: HashMap<String, f64>) -> PyResult<(String, f64)> {
    Ok(get_most_recent_common_ancestor_with_score(map))
}


#[pyfunction]
fn semantic_jaccard_similarity(
    closure_table: HashMap<String, HashMap<String, HashSet<String>>>,
    entity1: String,
    entity2: String,
    predicates: Option<HashSet<String>>,
) -> PyResult<f64> {
    Ok(calculate_semantic_jaccard_similarity(
        &closure_table,
        entity1,
        entity2,
        &predicates,
    ))
}

#[pyfunction]
fn max_information_content(
    closure_table: HashMap<String, HashMap<String, HashSet<String>>>,
    entity1: String,
    entity2: String,
    predicates: Option<HashSet<String>>,
) -> PyResult<f64> {
    Ok(calculate_max_information_content(
        &closure_table,
        &entity1,
        &entity2,
        &predicates,
    ))
}

#[pyfunction]
fn relationships_to_closure_table(
    list_of_tuples: Vec<(String, String, String)>,
) -> PyResult<(HashMap<String, HashMap<String, HashSet<String>>>, HashMap<String, f64>)> {
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


//TODO: Test the lib module.
