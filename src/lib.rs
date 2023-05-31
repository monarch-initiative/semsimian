
use std::{
    collections::{HashMap, HashSet},
};

use pyo3::prelude::*;
pub mod utils;

mod test_utils;

pub mod similarity;
use std::fmt;


use similarity::{calculate_max_information_content, calculate_phenomizer_score};
use utils::{convert_list_of_tuples_to_hashmap, expand_term_using_closure, predicate_set_to_key};

// change to "pub" because it is easier for testing
pub type Predicate = String; 
pub type TermID = String;
pub type PredicateSetKey = String;

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

    pub fn jaccard_similarity(
        &mut self,
        term1: &str,
        term2: &str,
        predicates: &Option<HashSet<Predicate>>,
    ) -> f64 {
        let (this_closure_map, _) = self.get_closure_and_ic_map(predicates);

        let term1_set = expand_term_using_closure(term1, &this_closure_map, predicates);
        let term2_set = expand_term_using_closure(term2, &this_closure_map, predicates);

        let intersection = term1_set.intersection(&term2_set).count() as f64;
        let union = term1_set.union(&term2_set).count() as f64;

        intersection / union
    }

    pub fn resnik_similarity(
        &self,
        term1: &str,
        term2: &str,
        predicates: &Option<HashSet<Predicate>>,
    ) -> f64 {
        calculate_max_information_content(&self.closure_map, &self.ic_map, term1, term2, predicates)
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

    fn jaccard_similarity(
        &mut self,
        term1: TermID,
        term2: TermID,
        predicates: Option<HashSet<Predicate>>,
    ) -> PyResult<f64> {
        Ok(self.ss.jaccard_similarity(&term1, &term2, &predicates))
    }

    fn resnik_similarity(
        &mut self,
        term1: TermID,
        term2: TermID,
        predicates: Option<HashSet<Predicate>>,
    ) -> PyResult<f64> {
        Ok(self.ss.resnik_similarity(&term1, &term2, &predicates))
    }
}

impl fmt::Debug for RustSemsimian {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RustSemsimian {{ spo: {:?}, ic_map: {:?}, closure_map: {:?} }}", self.spo, self.ic_map, self.closure_map)
    }
}


#[pymodule]
fn semsimian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Semsimian>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_jaccard_similarity() {
        let spo_cloned = crate::test_utils::test_constants::SPO_FRUITS.clone();
        let predicates: Option<HashSet<Predicate>> = Some(
            vec!["related_to"].into_iter().map(|s| s.to_string()).collect()
        );
        let no_predicates: Option<HashSet<Predicate>> = None;
        let mut ss = RustSemsimian::new(spo_cloned);
        let (closure_table3, _) = ss.get_closure_and_ic_map(&predicates);
        println!("Closure table for ss  {:?}", closure_table3);
        //Closure table: {"+related_to": {"apple": {"banana", "apple"}, "banana": {"orange", "banana"}, "pear": {"kiwi", "pear"}, "orange": {"orange", "pear"}}}
        let term1 = "apple".to_string();
        let term2 = "banana".to_string();
        let sim = ss.jaccard_similarity(&term1, &term2, &predicates);
        let sim2 = ss.jaccard_similarity(&term1, &term2, &no_predicates);

        assert_eq!(sim, 1.0 / 3.0);
        assert_eq!(sim2, 1.0 / 3.0);

    }


    #[test]
    fn test_get_closure_and_ic_map() {
        let spo_cloned = crate::test_utils::test_constants::SPO_FRUITS.clone();
        let mut semsimian = RustSemsimian::new(spo_cloned);
        println!("semsimian after initialization: {:?}", semsimian);
        let test_predicates: Option<HashSet<Predicate>> = Some(
            vec!["related_to"].into_iter().map(|s| s.to_string()).collect()
        );
        let (closure_map, ic_map) = semsimian.get_closure_and_ic_map(&test_predicates);
        println!("Closure_map from semsimian {:?}", closure_map);
        // Closure_table: {"+related_to": {"orange": {"orange", "pear"}, "pear": {"pear", "kiwi"}, "apple": {"apple", "banana"}, "banana": {"banana", "orange"}}}
        println!("ic_map from semsimian  {:?}", ic_map);
        // ic_map:  {"+related_to": {"apple": 2.415037499278844, "banana": 2.0, "orange": 2.0, "kiwi": 4.0, "pear": 2.0}}
        assert!(!closure_map.is_empty());
        assert!(!ic_map.is_empty());
    }

    #[test]
    fn test_resnik_similarity() {
        let spo_cloned = crate::test_utils::test_constants::SPO_FRUITS.clone();
        let mut rs = RustSemsimian::new(spo_cloned);
        let predicates: Option<HashSet<String>> = Some(
            vec!["related_to".to_string()]
            .into_iter()
            .collect()
        );
        let (closure_map, _ic_map) = rs.get_closure_and_ic_map(&predicates);
        println!("Closure_map from semsimian {:?}", closure_map);
        let sim = rs.resnik_similarity(&"apple".to_string(), &"banana".to_string(), &predicates);
        println!("Do the print{}", sim);
        assert!(sim > 0.0);
        let sim2 = rs.resnik_similarity(&"apple".to_string(), &"apple".to_string(), &predicates);
        println!("DO THE print{}", sim2);
        assert_eq!(sim2, 2.415037499278844);
    }


}

