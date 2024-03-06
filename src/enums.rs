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
    Jaccard,
    Phenodigm,
    Cosine,
}

impl MetricEnum {
    pub fn as_str(&self) -> &str {
        match *self {
            MetricEnum::AncestorInformationContent => "ancestor_information_content",
            MetricEnum::Jaccard => "jaccard",
            MetricEnum::Phenodigm => "phenodigm",
            MetricEnum::Cosine => "cosine",
        }
    }

    // Convert a &str to the corresponding Metric enum variant
    pub fn from_string(metric: &str) -> PyResult<Self> {
        match metric {
            "ancestor_information_content" => Ok(MetricEnum::AncestorInformationContent),
            "jaccard" => Ok(MetricEnum::Jaccard),
            "phenodigm" => Ok(MetricEnum::Phenodigm),
            "cosine" => Ok(MetricEnum::Cosine),
            _ => Err(PyValueError::new_err("Invalid metric type")),
        }
    }
}
