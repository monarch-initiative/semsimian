use std::collections::HashSet;

use pyo3::{pyclass, pymethods};

#[derive(Debug, Clone)]
#[pyclass]
pub struct TermSetPairwiseSimilarity {
    pub set_id: String,
    pub original_subject_termset: HashSet<String>,
    pub subject_termset: HashSet<String>,
    pub original_object_termset: HashSet<String>,
    pub object_termset: HashSet<String>,
    pub jaccard_similarity: f64,
}

#[pymethods]
impl TermSetPairwiseSimilarity {
    #[staticmethod]
    pub fn new() -> TermSetPairwiseSimilarity {
        TermSetPairwiseSimilarity {
            set_id: String::new(),
            original_subject_termset: HashSet::new(),
            subject_termset: HashSet::new(),
            original_object_termset: HashSet::new(),
            object_termset: HashSet::new(),
            jaccard_similarity: 0.0
        }
    }

    pub fn get_set_id(&self) -> String {
        self.set_id.clone()
    }

    pub fn get_original_subject_termset(&self) -> HashSet<String> {
        self.original_subject_termset.clone()
    }

    pub fn get_subject_termset(&self) -> HashSet<String> {
        self.subject_termset.clone()
    }

    pub fn get_original_object_termset(&self) -> HashSet<String> {
        self.original_object_termset.clone()
    }

    pub fn get_object_termset(&self) -> HashSet<String> {
        self.object_termset.clone()
    }

    pub fn get_jaccard_similarity(&self) -> f64 {
        self.jaccard_similarity
    }
}
