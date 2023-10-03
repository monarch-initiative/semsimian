use pyo3::{exceptions::PyValueError, PyResult};

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
