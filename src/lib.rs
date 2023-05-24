use std::{
    collections::{HashMap, HashSet},
};
use pyo3::prelude::*;
pub mod utils;
pub mod similarity;

use similarity::{calculate_max_information_content, calculate_phenomizer_score};
use utils::{convert_list_of_tuples_to_hashmap, expand_term_using_closure, predicate_set_to_key};

// type Predicate = String;
// type TermID = String;
// type PredicateSetKey = String;

pub struct RustSemsimian {
    spo: Vec<(String, String, String)>,

    ic_map: HashMap<String, HashMap<String, f64>>,
    // ic_map is something like {('is_a_+_part_of'), {'GO:1234': 1.234}}

    closure_map: HashMap<String, HashMap<String, HashSet<String>>>,
    // closure_map is something like {('is_a_+_part_of'), {'GO:1234': {'GO:1234', 'GO:5678'}}}
}

impl RustSemsimian {
    // TODO: this is tied directly to Oak, and should be made more generic
    // TODO: also, we should support loading 'custom' ic
    // TODO: also also, we should use str's instead of String
    pub fn new(spo: Vec<(String, String, String)>) -> RustSemsimian {
        // The line below converts Vec<String> to Vec<&str>
        // let new_spo:Vec<&str> = spo.iter().map(|s| s.as_ref()).collect();
        RustSemsimian {
            spo: spo,
            ic_map: HashMap::new(),
            closure_map: HashMap::new(),
        }
    }

    pub fn jaccard_similarity(&mut self, term1: &str, term2: &str, predicates: &Option<HashSet<&str>>) -> f64 {
        let (this_closure_map, _) = self.get_closure_and_ic_map(predicates);

        let term1_set = expand_term_using_closure(term1, &this_closure_map, predicates);
        let term2_set = expand_term_using_closure(term2, &this_closure_map, predicates);

        let intersection = term1_set.intersection(&term2_set).count() as f64;
        let union = term1_set.union(&term2_set).count() as f64;
        intersection / union
    }

    pub fn resnik_similarity(&self, term1: &str, term2: &str, predicates: &Option<HashSet<&str>>) -> f64 {

        calculate_max_information_content(&self.closure_map, &self.ic_map, term1, term2, predicates)
    }

    // TODO: make this predicate aware, and make it work with the new closure map
    pub fn phenomizer_score(
        map: HashMap<&str, HashMap<&str, f64>>,
        entity1: HashSet<&str>,
        entity2: HashSet<&str>,
    ) -> PyResult<f64> {
        Ok(calculate_phenomizer_score(map, entity1, entity2))
    }

    // get closure and ic map for a given set of predicates. if the closure and ic map for the given predicates doesn't exist, create them
    fn get_closure_and_ic_map<'a>(&'a mut self, predicates: &'a Option<HashSet<&'a str>>) ->
            (HashMap<&'a str, HashMap<&'a str, HashSet<&'a str>>>, HashMap<&'a str, HashMap<&'a str, f64>>) {
        let predicate_set_key = &*predicate_set_to_key(predicates);
        if !self.closure_map.contains_key(predicate_set_key) || !self.ic_map.contains_key(predicate_set_key) {
            let (this_closure_map, this_ic_map) = convert_list_of_tuples_to_hashmap(&self.spo, &predicates);
            self.closure_map.insert(predicate_set_key.to_string(), this_closure_map.get(&predicate_set_key).unwrap().clone());
            self.ic_map.insert(predicate_set_key.clone(), this_ic_map.get(&predicate_set_key).unwrap().clone());
        }
        (self.closure_map.clone(), self.ic_map.clone())
    }
}

#[pyclass]
pub struct Semsimian {
    ss: RustSemsimian,
}

#[pymethods]
impl Semsimian {
    #[new]
    fn new(spo: Vec<&str>) -> PyResult<Self> {
        let ss = RustSemsimian::new(spo);
        Ok(Semsimian { ss })
    }

    fn jaccard_similarity(&mut self, term1: &str, term2: &str, predicates: Option<HashSet<&str>>) -> PyResult<f64> {
        Ok(self.ss.jaccard_similarity(&term1, &term2, &predicates))
    }

    fn resnik_similarity(&mut self, term1: &str, term2: &str, predicates: Option<HashSet<&str>>) -> PyResult<f64> {
        Ok(self.ss.resnik_similarity(&term1, &term2, &predicates))
    }

}

#[pymodule]
fn semsimian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Semsimian>()?;
    Ok(())
}


//TODO: Test the lib module.
#[cfg(test)]
mod test {
    #[test]

    fn test_reality() {
        assert_eq!(1, 1);
    }
}