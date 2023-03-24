use std::collections::HashSet;

use pyo3::{pyclass, pymethods};

#[derive(Debug, Clone)]
#[pyclass]
pub struct TermSetPairwiseSimilarity {
    pub set_id: String,
    pub subject_termset: HashSet<String>,
    pub object_termset: HashSet<String>,
    pub subject_best_matches: Vec<String>, // TODO: needs to be BestMatch
    pub object_best_matches: Vec<String>,  // TODO: needs to be BestMatch
    pub average_score: f64,
    pub best_score: f64,
    pub metric: String,
}

#[pymethods]
impl TermSetPairwiseSimilarity {
    #[staticmethod]
    pub fn new() -> TermSetPairwiseSimilarity {
        TermSetPairwiseSimilarity {
            set_id: String::new(),
            subject_termset: HashSet::new(),
            object_termset: HashSet::new(),
            subject_best_matches: Vec::new(),
            object_best_matches: Vec::new(),
            average_score: 0.0, // TODO: this is a placeholder.
            best_score: 0.0,    // TODO: this is a placeholder.
            metric: "SWO:0000243".to_string(),
        }
    }

    pub fn get_set_id(&self) -> String {
        self.set_id.clone()
    }

    pub fn get_subject_termset(&self) -> HashSet<String> {
        self.subject_termset.clone()
    }

    pub fn get_object_termset(&self) -> HashSet<String> {
        self.object_termset.clone()
    }

    pub fn get_subject_best_matches(&self) -> Vec<String> {
        self.subject_best_matches.clone()
    }

    pub fn get_object_best_matches(&self) -> Vec<String> {
        self.object_best_matches.clone()
    }

    pub fn get_average_score(&self) -> f64 {
        self.average_score.clone()
    }

    pub fn get_best_score(&self) -> f64 {
        self.best_score.clone()
    }

    pub fn get_metric(&self) -> String {
        self.metric.clone()
    }
}
