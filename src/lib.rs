use std::{
    collections::{HashMap, HashSet},
};
use pyo3::prelude::*;
pub mod utils;
pub mod similarity;
use similarity::{*};
use utils::{*};

#[derive(Default)]
pub struct RustSemsimian {
    ic_map: HashMap<String, f64>,
    closure_map: HashMap<String, HashMap<String, HashSet<String>>>,
}

impl RustSemsimian {
    pub fn new(spo: Vec<(String, String, String)>) -> RustSemsimian {
        let (closure_map, ic_map) = convert_list_of_tuples_to_hashmap(spo);

        RustSemsimian {
            ic_map,
            closure_map
        }
    }

    pub fn jaccard_similarity(&self, term1: &String, term2: &String, predicates: Option<HashSet<String>>) -> f64 {
        let term1_set = expand_term_using_closure(term1, &self.closure_map, &predicates);
        let term2_set = expand_term_using_closure(term2, &self.closure_map, &predicates);
        let intersection = term1_set.intersection(&term2_set).count() as f64;
        let union = term1_set.union(&term2_set).count() as f64;
        intersection / union
    }

    // TODO: this calculates ic on the fly each time, it should instead use ic_map
    pub fn resnik_similarity(&self, term1: &String, term2: &String, predicates: Option<HashSet<String>>) -> f64 {
        calculate_max_information_content(&self.closure_map, term1, term2, &predicates)
    }

    // TODO: make this predicate aware
    pub fn phenomizer_score(
        map: HashMap<String, HashMap<String, f64>>,
        entity1: HashSet<String>,
        entity2: HashSet<String>,
    ) -> PyResult<f64> {
        Ok(calculate_phenomizer_score(map, entity1, entity2))
    }

}

#[pyclass]
pub struct Semsimian {
    ss: RustSemsimian,
}

#[pymethods]
impl Semsimian {
    #[new]
    fn new(spo: Vec<(String, String, String)>) -> PyResult<Self> {
        let ss = RustSemsimian::new(spo);
        Ok(Semsimian { ss })
    }

    fn jaccard_similarity(&self, term1: String, term2: String, predicates: Option<HashSet<String>>) -> PyResult<f64> {
        Ok(self.ss.jaccard_similarity(&term1, &term2, predicates))
    }

    fn resnik_similarity(&self, term1: String, term2: String, predicates: Option<HashSet<String>>) -> PyResult<f64> {
        Ok(self.ss.resnik_similarity(&term1, &term2, predicates))
    }

}

#[pymodule]
fn semsimian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Semsimian>()?;
    Ok(())
}


//TODO: Test the lib module.
