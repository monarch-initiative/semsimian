use std::{
    collections::{HashMap, HashSet},
};
use pyo3::prelude::*;
pub mod utils;
pub mod similarity;

use similarity::{calculate_max_information_content, calculate_phenomizer_score};
use utils::{convert_list_of_tuples_to_hashmap, expand_term_using_closure};

pub struct RustSemsimian {
    spo: Vec<(String, String, String)>,

    // TODO: Let's change these Strings to something more descriptive, like CURIE or some such
    ic_map: HashMap<HashSet<String>, HashMap<String, f64>>,
    // ic_map is something like {('is_a', 'part_of'), {'GO:1234': 1.234}}

    closure_map: HashMap<HashSet<String>, HashMap<String, HashSet<String>>>,
    // closure_map is something like {('is_a', 'part_of'), {'GO:1234': {'GO:1234', 'GO:5678'}}}
}

impl RustSemsimian {
    // TODO: this is tied directly to Oak, and should be made more generic
    // TODO: also, we should support loading 'custom' ic
    // TODO: also also, we should use str's instead of String
    pub fn new(spo: Vec<(String, String, String)>) -> RustSemsimian {
        // let (closure_map, ic_map) = convert_list_of_tuples_to_hashmap(spo);

        RustSemsimian {
            spo,
            ic_map: HashMap::new(),
            closure_map: HashMap::new(),
        }
    }

    pub fn jaccard_similarity(&self, term1: &String, term2: &String, predicates: Option<HashSet<String>>) -> f64 {
        let (this_closure_map, _) = self.get_closure_and_ic_map(predicates);

        let term1_set = expand_term_using_closure(term1, &this_closure_map, &predicates);
        let term2_set = expand_term_using_closure(term2, &this_closure_map, &predicates);
        let intersection = term1_set.intersection(&term2_set).count() as f64;
        let union = term1_set.union(&term2_set).count() as f64;
        intersection / union
    }

    pub fn resnik_similarity(&self, term1: &String, term2: &String, predicates: Option<HashSet<String>>) -> f64 {
        let (this_closure_map, this_ic_map) = self.get_closure_and_ic_map(predicates);

        calculate_max_information_content(&this_closure_map, &this_ic_map, term1, term2, &predicates)
    }

    // TODO: make this predicate aware, and make it work with the new closure map
    pub fn phenomizer_score(
        map: HashMap<String, HashMap<String, f64>>,
        entity1: HashSet<String>,
        entity2: HashSet<String>,
    ) -> PyResult<f64> {
        Ok(calculate_phenomizer_score(map, entity1, entity2))
    }

    // get closure and ic map for a given set of predicates. if the closure and ic map for the given predicates doesn't exist, create them
    fn get_closure_and_ic_map(&self, predicates: Option<HashSet<String>>) -> (HashMap<HashSet<String>, HashMap<String, HashSet<String>>>, HashMap<HashSet<String>, HashMap<String, f64>>) {
        let closure_and_ic_map = (HashMap::new(), HashMap::new());
        if self.closure_map.contains_key(&predicates) && self.ic_map.contains_key(&predicates) {
            closure_and_ic_map = (self.closure_map.get(&predicates).unwrap(), self.ic_map.get(&predicates).unwrap());
        }
        else {
            closure_and_ic_map = convert_list_of_tuples_to_hashmap(&self.spo, &predicates);
            self.closure_map.insert(predicates.clone(), closure_and_ic_map.0);
            self.ic_map.insert(predicates.clone(), closure_and_ic_map.1);
        }
        closure_and_ic_map
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
