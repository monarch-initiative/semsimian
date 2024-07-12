use pyo3::{exceptions::PyValueError, PyResult};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub enum SearchTypeEnum {
    Flat,
    Full,
    Hybrid,
}

impl SearchTypeEnum {
    pub fn as_str(&self) -> &str {
        match *self {
            SearchTypeEnum::Flat => "flat",
            SearchTypeEnum::Full => "full",
            SearchTypeEnum::Hybrid => "hybrid",
        }
    }

    pub fn from_string(s: &str) -> PyResult<Self> {
        match s {
            "flat" => Ok(SearchTypeEnum::Flat),
            "full" => Ok(SearchTypeEnum::Full),
            "hybrid" => Ok(SearchTypeEnum::Hybrid),
            _ => Err(PyValueError::new_err("Invalid search type")),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, Ord, PartialOrd, Eq, PartialEq)]
pub enum MetricEnum {
    #[default]
    AncestorInformationContent,
    JaccardSimilarity,
    PhenodigmScore,
    CosineSimilarity,
}

impl MetricEnum {
    pub fn as_str(&self) -> &str {
        match *self {
            MetricEnum::AncestorInformationContent => "ancestor_information_content",
            MetricEnum::JaccardSimilarity => "jaccard_similarity",
            MetricEnum::PhenodigmScore => "phenodigm_score",
            MetricEnum::CosineSimilarity => "cosine_similarity",
        }
    }

    // Convert an Option<&str> to the corresponding Metric enum variant
    pub fn from_string(metric: &Option<&str>) -> PyResult<Self> {
        match metric.as_deref() {
            Some("jaccard_similarity") => Ok(MetricEnum::JaccardSimilarity),
            Some("phenodigm_score") => Ok(MetricEnum::PhenodigmScore),
            Some("cosine_similarity") => Ok(MetricEnum::CosineSimilarity),
            Some(_) | None => Ok(MetricEnum::AncestorInformationContent), // Default case includes None and any other string
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, Ord, PartialOrd, Eq, PartialEq)]
pub enum DirectionalityEnum {
    #[default]
    Bidirectional,
    SubjectToObject,
    ObjectToSubject,
}

impl DirectionalityEnum {
    pub fn as_str(&self) -> &str {
        match *self {
            DirectionalityEnum::Bidirectional => "bidirectional",
            DirectionalityEnum::SubjectToObject => "subject_to_object",
            DirectionalityEnum::ObjectToSubject => "object_to_subject",
        }
    }

    // Convert an Option<&str> to the corresponding Directionality enum variant
    pub fn from_string(directionality: &Option<&str>) -> PyResult<Self> {
        match directionality.as_deref() {
            Some("subject_to_object") => Ok(DirectionalityEnum::SubjectToObject),
            Some("object_to_subject") => Ok(DirectionalityEnum::ObjectToSubject),
            Some(_) | None => Ok(DirectionalityEnum::Bidirectional), // Default case includes None and any other string
        }
    }
}
