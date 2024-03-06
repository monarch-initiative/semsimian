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

    // Convert a &str to the corresponding Metric enum variant
    pub fn from_string(metric: &str) -> PyResult<Self> {
        match metric {
            "ancestor_information_content" => Ok(MetricEnum::AncestorInformationContent),
            "jaccard_similarity" => Ok(MetricEnum::JaccardSimilarity),
            "phenodigm_score" => Ok(MetricEnum::PhenodigmScore),
            "cosine_similarity" => Ok(MetricEnum::CosineSimilarity),
            _ => Err(PyValueError::new_err("Invalid metric type")),
        }
    }
}
