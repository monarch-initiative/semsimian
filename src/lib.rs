use pyo3::prelude::*;

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};
pub mod similarity;

pub mod utils;
use rayon::prelude::*;

mod test_utils;

use std::fmt;

use similarity::{calculate_max_information_content, calculate_phenomizer_score};
use utils::{
    convert_list_of_tuples_to_hashmap, expand_term_using_closure,
    generate_progress_bar_of_length_and_message, predicate_set_to_key,
};

// change to "pub" because it is easier for testing
pub type Predicate = String;
pub type TermID = String;
pub type PredicateSetKey = String;
pub type Jaccard = f64;
pub type Resnik = f64;
pub type Phenodigm = f64;
pub type MostInformativeAncestors = HashSet<TermID>;

#[derive(Clone)]
pub struct RustSemsimian {
    spo: Vec<(TermID, Predicate, TermID)>,

    ic_map: HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    // ic_map is something like {("is_a_+_part_of"), {"GO:1234": 1.234}}
    closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    // closure_map is something like {("is_a_+_part_of"), {"GO:1234": {"GO:1234", "GO:5678"}}}
}

impl RustSemsimian {
    // TODO: this is tied directly to Oak, and should be made more generic
    // TODO: also, we should support loading "custom" ic
    // TODO: generate ic map and closure map using (spo).
    pub fn new(spo: Vec<(TermID, Predicate, TermID)>) -> RustSemsimian {
        RustSemsimian {
            spo,
            ic_map: HashMap::new(),
            closure_map: HashMap::new(),
        }
    }

    pub fn update_closure_and_ic_map(&mut self, predicates: &Option<HashSet<Predicate>>) {
        let predicate_set_key = predicate_set_to_key(predicates);
        let (this_closure_map, this_ic_map) =
            convert_list_of_tuples_to_hashmap(&self.spo, predicates);
        self.closure_map.insert(
            predicate_set_key.clone(),
            this_closure_map.get(&predicate_set_key).unwrap().clone(),
        );
        self.ic_map.insert(
            predicate_set_key.clone(),
            this_ic_map.get(&predicate_set_key).unwrap().clone(),
        );
    }

    pub fn jaccard_similarity(
        &self,
        term1: &str,
        term2: &str,
        predicates: &Option<HashSet<Predicate>>,
    ) -> f64 {
        let apple_set = expand_term_using_closure(term1, &self.closure_map, predicates);
        let fruit_set = expand_term_using_closure(term2, &self.closure_map, predicates);

        let intersection = apple_set.intersection(&fruit_set).count() as f64;
        let union = apple_set.union(&fruit_set).count() as f64;
        intersection / union
    }

    pub fn resnik_similarity(
        &self,
        term1: &str,
        term2: &str,
        predicates: &Option<HashSet<Predicate>>,
    ) -> (HashSet<String>, f64) {
        calculate_max_information_content(&self.closure_map, &self.ic_map, term1, term2, predicates)
    }

    pub fn all_by_all_pairwise_similarity<'a>(
        &'a self,
        subject_terms: &'a HashSet<TermID>,
        object_terms: &'a HashSet<TermID>,
        minimum_jaccard_threshold: &'a Option<f64>,
        minimum_resnik_threshold: &'a Option<f64>,
        predicates: &'a Option<HashSet<Predicate>>,
    ) -> impl Iterator<
        Item = (
            TermID,
            HashMap<TermID, (Jaccard, Resnik, Phenodigm, MostInformativeAncestors)>,
        ),
    > + 'a {
        let self_shared = Arc::new(RwLock::new(self.clone()));
        let pb = generate_progress_bar_of_length_and_message(
            (subject_terms.len() * object_terms.len()) as u64,
            "Building all X all pairwise similarity:",
        );
        let subject_similarities = Arc::new(RwLock::new(HashMap::new()));
        subject_terms.par_iter().for_each(|subject| {
            let mut similarities: HashMap<
                TermID,
                (Jaccard, Resnik, Phenodigm, MostInformativeAncestors),
            > = HashMap::new();
            for object in object_terms.iter() {
                let self_read = self_shared.read().unwrap();
                let jaccard_sim = self_read.jaccard_similarity(subject, object, predicates);
                let (mica, resnik_sim) = self_read.resnik_similarity(subject, object, predicates);

                if minimum_jaccard_threshold.map_or(true, |t| jaccard_sim > t)
                    && minimum_resnik_threshold.map_or(true, |t| resnik_sim > t)
                {
                    similarities.insert(
                        object.clone(),
                        (
                            jaccard_sim,
                            resnik_sim,
                            (resnik_sim * jaccard_sim).sqrt(),
                            mica,
                        ),
                    );
                }
            }
            subject_similarities
                .write()
                .unwrap()
                .insert(subject.clone(), similarities);
            pb.inc(object_terms.len() as u64);
        });
        let cloned_subject_similarities = Arc::try_unwrap(subject_similarities)
            .unwrap()
            .into_inner()
            .unwrap();
        cloned_subject_similarities.into_iter()
    }

    // TODO: make this predicate aware, and make it work with the new closure map
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
        self.ss.update_closure_and_ic_map(&predicates);
        Ok(self.ss.jaccard_similarity(&term1, &term2, &predicates))
    }

    fn resnik_similarity(
        &mut self,
        term1: TermID,
        term2: TermID,
        predicates: Option<HashSet<Predicate>>,
    ) -> PyResult<(HashSet<String>, f64)> {
        self.ss.update_closure_and_ic_map(&predicates);
        Ok(self.ss.resnik_similarity(&term1, &term2, &predicates))
    }

    fn all_by_all_pairwise_similarity(
        &mut self,
        subject_terms: HashSet<TermID>,
        object_terms: HashSet<TermID>,
        minimum_jaccard_threshold: Option<f64>,
        minimum_resnik_threshold: Option<f64>,
        predicates: Option<HashSet<Predicate>>,
    ) -> PyResult<HashMap<String, HashMap<String, (f64, f64, f64, HashSet<String>)>>> {
        // first make sure we have the closure and ic map for the given predicates
        self.ss.update_closure_and_ic_map(&predicates);

        let all_x_all = self.ss.all_by_all_pairwise_similarity(
            &subject_terms,
            &object_terms,
            &minimum_jaccard_threshold,
            &minimum_resnik_threshold,
            &predicates,
        );

        let mut output_map = HashMap::new();
        for (key, value) in all_x_all {
            let inner_map = value.into_iter().collect();
            output_map.insert(key, inner_map);
        }
        Ok(output_map)
    }

    fn get_spo(&self) -> PyResult<Vec<(TermID, Predicate, TermID)>> {
        Ok(self.ss.spo.to_vec())
    }
}

impl fmt::Debug for RustSemsimian {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RustSemsimian {{ spo: {:?}, ic_map: {:?}, closure_map: {:?} }}",
            self.spo, self.ic_map, self.closure_map
        )
    }
}

impl Iterator for RustSemsimian {
    type Item = (
        TermID,
        HashMap<TermID, (Jaccard, Resnik, Phenodigm, MostInformativeAncestors)>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        None
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
    use crate::RustSemsimian;
    use std::collections::HashSet;

    #[test]
    fn test_jaccard_similarity() {
        let spo_cloned = crate::test_utils::test_constants::SPO_FRUITS.clone();
        let predicates: Option<HashSet<Predicate>> = Some(
            vec!["related_to"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        );
        let no_predicates: Option<HashSet<Predicate>> = None;
        let mut ss = RustSemsimian::new(spo_cloned);
        ss.update_closure_and_ic_map(&predicates);
        println!("Closure table for ss  {:?}", ss.closure_map);
        //Closure table: {"+related_to": {"apple": {"banana", "apple"}, "banana": {"orange", "banana"}, "pear": {"kiwi", "pear"}, "orange": {"orange", "pear"}}}
        let apple = "apple".to_string();
        let banana = "banana".to_string();
        let sim = ss.jaccard_similarity(&apple, &banana, &predicates);
        let sim2 = ss.jaccard_similarity(&apple, &banana, &no_predicates);

        assert_eq!(sim, 1.0 / 3.0);
        assert_eq!(sim2, 1.0 / 3.0);
    }

    #[test]
    fn test_get_closure_and_ic_map() {
        let spo_cloned = crate::test_utils::test_constants::SPO_FRUITS.clone();
        let mut semsimian = RustSemsimian::new(spo_cloned);
        println!("semsimian after initialization: {semsimian:?}");
        let test_predicates: Option<HashSet<Predicate>> = Some(
            vec!["related_to"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        );
        semsimian.update_closure_and_ic_map(&test_predicates);
        assert!(!semsimian.closure_map.is_empty());
        assert!(!semsimian.ic_map.is_empty());
    }

    #[test]
    fn test_resnik_similarity() {
        let spo_cloned = crate::test_utils::test_constants::SPO_FRUITS.clone();
        let mut rs = RustSemsimian::new(spo_cloned);
        let predicates: Option<HashSet<String>> =
            Some(vec!["related_to".to_string()].into_iter().collect());
        rs.update_closure_and_ic_map(&predicates);
        println!("Closure_map from semsimian {:?}", rs.closure_map);
        let (_, sim) = rs.resnik_similarity("apple", "banana", &predicates);
        println!("DO THE print{sim}");
        assert_eq!(sim, 1.3219280948873622);
    }

    #[test]
    fn test_all_by_all_pairwise_similarity_with_empty_inputs() {
        let rss = RustSemsimian::new(vec![(
            "apple".to_string(),
            "is_a".to_string(),
            "fruit".to_string(),
        )]);

        let subject_terms: HashSet<TermID> = HashSet::new();
        let object_terms: HashSet<TermID> = HashSet::new();
        let predicates: Option<HashSet<Predicate>> = None;

        let result = rss.all_by_all_pairwise_similarity(
            &subject_terms,
            &object_terms,
            &Some(0.0),
            &Some(0.0),
            &predicates,
        );

        assert_eq!(result.count(), 0);
    }

    #[test]
    fn test_all_by_all_pairwise_similarity() {
        // HP:0000118: Phenotypic Abnormality
        // HP:0001507: Growth Abnormality
        // HP:0010718: Abormality of body weight
        // HP:0001510: Growth delay
        // HP:0004322: Short stature

        // No direct relationship with "HP:0001507" or "HP:0004322"
        // Create a new RustSemsimian object
        let mut rss = RustSemsimian::new(vec![
            (
                "HP:0004322".to_string(),
                "is_a".to_string(),
                "HP:0001510".to_string(),
            ),
            (
                "HP:0004322".to_string(),
                "is_a".to_string(),
                "HP:0010718".to_string(),
            ),
            (
                "HP:0004322".to_string(),
                "is_a".to_string(),
                "HP:0001507".to_string(),
            ),
            (
                "HP:0001510".to_string(),
                "is_a".to_string(),
                "HP:0001507".to_string(),
            ),
            (
                "HP:0001510".to_string(),
                "is_a".to_string(),
                "HP:0001507".to_string(),
            ),
            (
                "HP:0010718".to_string(),
                "is_a".to_string(),
                "HP:0001507".to_string(),
            ),
            (
                "HP:0001507".to_string(),
                "is_a".to_string(),
                "HP:0000118".to_string(),
            ),
        ]);

        // Define some test data
        let subject_terms: HashSet<TermID> = ["HP:0010718", "HP:0001510", "HP:0004322"]
            .iter()
            .map(|&x| x.into())
            .collect();
        let object_terms: HashSet<TermID> = ["HP:0000118", "HP:0001507", "HP:0010718"]
            .iter()
            .map(|&x| x.into())
            .collect();
        // let minimum_jaccard_threshold = Some(0.5);
        // let minimum_resnik_threshold = Some(1.0);
        let minimum_jaccard_threshold = None;
        let minimum_resnik_threshold = None;
        let predicates: Option<HashSet<Predicate>> = Some(HashSet::from(["is_a".to_string()]));

        rss.update_closure_and_ic_map(&predicates);

        // Call the function and collect the results into a vector
        let result: Vec<_> = rss
            .all_by_all_pairwise_similarity(
                &subject_terms,
                &object_terms,
                &minimum_jaccard_threshold,
                &minimum_resnik_threshold,
                &predicates,
            )
            .collect();
        // println!("{result:?}");

        // Define the expected output

        let expected_output: Vec<_> = vec![
            (
                "HP:0010718".to_string(),
                HashMap::from([
                    (
                        "HP:0010718".to_string(),
                        (
                            1.0_f64,
                            0.0_f64,
                            0.0_f64,
                            HashSet::from(["HP:0001507".to_string()]),
                        ),
                    ),
                    (
                        "HP:0000118".to_string(),
                        (0.0_f64, 0.0_f64, 0.0_f64, HashSet::new()),
                    ),
                    (
                        "HP:0001507".to_string(),
                        (0.0_f64, 0.0_f64, 0.0_f64, HashSet::new()),
                    ),
                ]),
            ),
            (
                "HP:0001510".to_string(),
                HashMap::from([
                    (
                        "HP:0000118".to_string(),
                        (0.0_f64, 0.0_f64, 0.0_f64, HashSet::new()),
                    ),
                    (
                        "HP:0001507".to_string(),
                        (0.0_f64, 0.0_f64, 0.0_f64, HashSet::new()),
                    ),
                    (
                        "HP:0010718".to_string(),
                        (
                            1.0_f64,
                            0.0_f64,
                            0.0_f64,
                            HashSet::from(["HP:0001507".to_string()]),
                        ),
                    ),
                ]),
            ),
            (
                "HP:0004322".to_string(),
                HashMap::from([
                    (
                        "HP:0010718".to_string(),
                        (
                            0.3333333333333333_f64,
                            0.0_f64,
                            0.0_f64,
                            HashSet::from(["HP:0001507".to_string()]),
                        ),
                    ),
                    (
                        "HP:0000118".to_string(),
                        (0.0_f64, 0.0_f64, 0.0_f64, HashSet::new()),
                    ),
                    (
                        "HP:0001507".to_string(),
                        (0.0_f64, 0.0_f64, 0.0_f64, HashSet::new()),
                    ),
                ]),
            ),
        ];

        // Compare the actual and expected output
        assert_eq!(result.len(), expected_output.len());
    }

    #[test]
    fn test_resnik_using_bfo() {
        let spo = crate::test_utils::test_constants::BFO_SPO.clone();
        let mut rss = RustSemsimian::new(spo);

        let predicates: Option<HashSet<Predicate>> = Some(HashSet::from([
            "rdfs:subClassOf".to_string(),
            "BFO:0000050".to_string(),
        ]));

        rss.update_closure_and_ic_map(&predicates);
        // println!("IC_map from semsimian {:?}", rss.ic_map);
        let (_, sim) = rss.resnik_similarity("BFO:0000040", "BFO:0000002", &predicates);
        println!("DO THE print {sim}");
        assert_eq!(sim, 0.4854268271702417);
    }
}
