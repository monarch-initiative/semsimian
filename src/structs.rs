use std::collections::HashSet;

#[derive(Debug)]
pub struct TermSetPairwiseSimilarity {
    pub set_id: String,
    pub original_subject_termset: HashSet<String>,
    pub subject_termset: HashSet<String>,
    pub original_object_termset: HashSet<String>,
    pub object_termset: HashSet<String>,
    pub jaccard_similarity: f64,
}
impl TermSetPairwiseSimilarity {
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
}