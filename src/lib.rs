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

    pub fn jaccard_similarity(&self, term1: &str, term2: &str, predicates: HashSet<String>) -> f64 {
        let term1_set = self.get_closure(term1);
        let term2_set = self.get_closure(term2);
        let intersection = term1_set.intersection(&term2_set).count() as f64;
        let union = term1_set.union(&term2_set).count() as f64;
        intersection / union
    }

    // TODO: implement max IC (what do we call this? max information content?, resnik_similarity?)

    pub fn phenomizer_score(
        map: HashMap<String, HashMap<String, f64>>,
        entity1: HashSet<String>,
        entity2: HashSet<String>,
    ) -> PyResult<f64> {
        Ok(calculate_phenomizer_score(map, entity1, entity2))
    }

    // TODO: deal with predicates (see working code elsewhere in this repo)
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
    fn new(spo: Vec<(String, String, String)>) -> PyResult<Self> {
        let ss = RustSemsimian::new(spo);
        Ok(Semsimian { ss })
    }

    fn jaccard_similarity(&self, term1: &str, term2: &str, predicates: HashSet<String>) -> PyResult<f64> {
        Ok(self.ss.jaccard_similarity(term1, term2, predicates))
    }

}

#[pymodule]
fn semsimian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Semsimian>()?;
    Ok(())
}


//TODO: Test the lib module.
