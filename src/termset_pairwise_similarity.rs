use pyo3::conversion::IntoPy;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
#[derive(Serialize, Deserialize)]
pub struct TermsetPairwiseSimilarity {
    pub subject_termset: Vec<BTreeMap<String, BTreeMap<String, String>>>,
    pub subject_best_matches: BTreeMap<String, BTreeMap<String, String>>,
    pub subject_best_matches_similarity_map: BTreeMap<String, BTreeMap<String, String>>,
    pub object_termset: Vec<BTreeMap<String, BTreeMap<String, String>>>,
    pub object_best_matches: BTreeMap<String, BTreeMap<String, String>>,
    pub object_best_matches_similarity_map: BTreeMap<String, BTreeMap<String, String>>,
    pub average_score: f64,
    pub best_score: f64,
    pub metric: String,
}
impl TermsetPairwiseSimilarity {
    pub fn new(
        subject_termset: Vec<BTreeMap<String, BTreeMap<String, String>>>,
        subject_best_matches: BTreeMap<String, BTreeMap<String, String>>,
        subject_best_matches_similarity_map: BTreeMap<String, BTreeMap<String, String>>,
        object_termset: Vec<BTreeMap<String, BTreeMap<String, String>>>,
        object_best_matches: BTreeMap<String, BTreeMap<String, String>>,
        object_best_matches_similarity_map: BTreeMap<String, BTreeMap<String, String>>,
        average_score: f64,
        best_score: f64,
        metric: String,
    ) -> TermsetPairwiseSimilarity {
        TermsetPairwiseSimilarity {
            subject_termset,
            subject_best_matches,
            subject_best_matches_similarity_map,
            object_termset,
            object_best_matches,
            object_best_matches_similarity_map,
            average_score,
            best_score,
            metric,
        }
    }
}

impl<'a> IntoPy<PyObject> for &'a TermsetPairwiseSimilarity {
    fn into_py(self, py: Python) -> PyObject {
        // Convert your struct into a PyObject
        let tsps_dict = PyDict::new(py);
        // Create nested dictionaries for subject_best_matches and object_best_matches
        let subject_best_matches_dict = self.subject_best_matches.clone().into_py_dict(py);
        subject_best_matches_dict
            .set_item("similarity", &self.subject_best_matches_similarity_map)
            .unwrap();

        // Set the similarity map as a value for 'similarity' key in subject_best_matches_dict
        let object_best_matches_dict = self.object_best_matches.clone().into_py_dict(py);
        object_best_matches_dict
            .set_item("similarity", &self.object_best_matches_similarity_map)
            .unwrap();

        // Add your struct fields to the dictionary
        tsps_dict
            .set_item("subject_termset", &self.subject_termset)
            .unwrap();
        tsps_dict
            .set_item("object_termset", &self.object_termset)
            .unwrap();
        tsps_dict
            .set_item("subject_best_matches", subject_best_matches_dict)
            .unwrap();

        tsps_dict
            .set_item("object_best_matches", object_best_matches_dict)
            .unwrap();
        tsps_dict
            .set_item("average_score", self.average_score)
            .unwrap();
        tsps_dict.set_item("best_score", self.best_score).unwrap();
        tsps_dict.set_item("metric", &self.metric).unwrap();

        tsps_dict.into()
    }
}
