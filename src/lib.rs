use std::{
    collections::{HashMap, HashSet},
};
use pyo3::prelude::*;
pub mod utils;
pub mod similarity;

use similarity::{calculate_max_information_content, calculate_phenomizer_score};
use utils::{convert_list_of_tuples_to_hashmap, expand_term_using_closure, predicate_set_to_key};

type Predicate = String;
type TermID = String;
type PredicateSetKey = String;

pub struct RustSemsimian {
    spo: Vec<(TermID, Predicate, TermID)>,

    ic_map: HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    // ic_map is something like {('is_a_+_part_of'), {'GO:1234': 1.234}}

    closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    // closure_map is something like {('is_a_+_part_of'), {'GO:1234': {'GO:1234', 'GO:5678'}}}
}

impl RustSemsimian {
    // TODO: this is tied directly to Oak, and should be made more generic
    // TODO: also, we should support loading 'custom' ic
    // TODO: also also, we should use str's instead of String
    pub fn new(spo: Vec<(TermID, Predicate, TermID)>) -> RustSemsimian {

        RustSemsimian {
            spo,
            ic_map: HashMap::new(),
            closure_map: HashMap::new(),
        }
    }

    pub fn jaccard_similarity(&mut self, term1: &TermID, term2: &TermID, predicates: &Option<HashSet<Predicate>>) -> f64 {
        let (this_closure_map, _) = self.get_closure_and_ic_map(predicates);

        let term1_set = expand_term_using_closure(term1, &this_closure_map, predicates);
        let term2_set = expand_term_using_closure(term2, &this_closure_map, predicates);

        let intersection = term1_set.intersection(&term2_set).count() as f64;
        let union = term1_set.union(&term2_set).count() as f64;
        intersection / union
    }

    pub fn resnik_similarity(&self, term1: &TermID, term2: &TermID, predicates: &Option<HashSet<Predicate>>) -> f64 {
        calculate_max_information_content(&self.closure_map, &self.ic_map, term1, term2, predicates)
    }

    pub fn all_by_all_pairwise_similarity(
        &mut self,
        subject_terms: &HashSet<TermID>,
        object_terms: &HashSet<TermID>,
        predicates: &Option<HashSet<Predicate>>,
    ) -> HashMap<TermID, HashMap<TermID, (f64, f64)>> {
        let mut similarity_map: HashMap<TermID, HashMap<TermID, (f64, f64)>> = HashMap::new();

        for subject in subject_terms {
            let mut subject_similarities: HashMap<TermID, (f64, f64)> = HashMap::new();
            for object in object_terms {
                let jaccard_sim = self.jaccard_similarity(subject, object, predicates);
                let resnik_sim = self.resnik_similarity(subject, object, predicates);
                subject_similarities.insert(object.clone(), (resnik_sim, jaccard_sim));
            }
            similarity_map.insert(subject.clone(), subject_similarities);
        }

        similarity_map
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
    fn get_closure_and_ic_map(&mut self, predicates: &Option<HashSet<Predicate>>) ->
            (HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>, HashMap<PredicateSetKey, HashMap<TermID, f64>>) {
        let predicate_set_key = predicate_set_to_key(&predicates);
        if !self.closure_map.contains_key(&predicate_set_key) || !self.ic_map.contains_key(&predicate_set_key) {
            let (this_closure_map, this_ic_map) = convert_list_of_tuples_to_hashmap(&self.spo, &predicates);
            self.closure_map.insert(predicate_set_key.clone(), this_closure_map.get(&predicate_set_key).unwrap().clone());
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
    fn new(spo: Vec<(TermID, Predicate, TermID)>) -> PyResult<Self> {
        let ss = RustSemsimian::new(spo);
        Ok(Semsimian { ss })
    }

    fn jaccard_similarity(&mut self, term1: TermID, term2: TermID, predicates: Option<HashSet<Predicate>>) -> PyResult<f64> {
        Ok(self.ss.jaccard_similarity(&term1, &term2, &predicates))
    }

    fn resnik_similarity(&mut self, term1: TermID, term2: TermID, predicates: Option<HashSet<Predicate>>) -> PyResult<f64> {
        Ok(self.ss.resnik_similarity(&term1, &term2, &predicates))
    }

    fn all_by_all_pairwise_similarity(
        &mut self,
        subject_terms: HashSet<TermID>,
        object_terms: HashSet<TermID>,
        predicates: Option<HashSet<Predicate>>,
    ) -> HashMap<TermID, HashMap<TermID, (f64, f64)>> {
        self.ss.all_by_all_pairwise_similarity(&subject_terms, &object_terms, &predicates)
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