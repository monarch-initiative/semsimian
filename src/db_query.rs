use crate::{Predicate, TermID};
use pyo3::conversion::IntoPy;
use pyo3::types::PyDict;
use pyo3::{PyObject, Python};
use rusqlite::{Connection, Result};
use std::collections::{HashMap, HashSet};

const TERM_ASSOCIATION_TABLE: &str = "term_association";
const STATEMENTS_TABLE: &str = "statements";
const ENTAILED_EDGE_TABLE: &str = "entailed_edge";

#[derive(Debug, Clone)]
pub struct TermAssociation {
    pub id: TermID,
    pub subject: TermID,
    pub predicate: TermID,
    pub object: TermID,
    pub evidence_type: TermID,
    pub publication: TermID,
    pub source: TermID,
}

impl<'a> IntoPy<PyObject> for &'a TermAssociation {
    fn into_py(self, py: Python) -> PyObject {
        // Convert your struct into a PyObject
        let term_association_dict = PyDict::new(py);

        // Add your struct fields to the dictionary
        term_association_dict.set_item("id", &self.id).unwrap();
        term_association_dict
            .set_item("subject", &self.subject)
            .unwrap();
        term_association_dict
            .set_item("predicate", &self.predicate)
            .unwrap();

        term_association_dict
            .set_item("object", &self.object)
            .unwrap();
        term_association_dict
            .set_item("evidence_type", &self.evidence_type)
            .unwrap();
        term_association_dict
            .set_item("publication", &self.publication)
            .unwrap();
        term_association_dict
            .set_item("source", &self.source)
            .unwrap();

        term_association_dict.into()
    }
}

pub fn get_entailed_edges_for_predicate_list(
    path: &str,
    predicates: &Vec<Predicate>,
) -> Result<Vec<(TermID, Predicate, TermID)>, rusqlite::Error> {
    // Build the SQL query with the provided table name such that 'predicates' are in the Vector predicates.
    let joined_predicates = format!("'{}'", predicates.join("', '"));

    let query = if !predicates.is_empty() {
        format!(
            "SELECT * FROM {} WHERE predicate IN ({})",
            ENTAILED_EDGE_TABLE, joined_predicates
        )
    } else {
        format!("SELECT * FROM {}", ENTAILED_EDGE_TABLE)
    };

    // Open a connection to the SQLite database file
    let conn = Connection::open(path)?;

    // Prepare the SQL query
    let mut stmt = conn.prepare(&query)?;

    // Execute the SQL query and retrieve the results
    let rows = stmt.query_map([], |row| {
        // Access the columns of each row
        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
    })?;

    // Collect the results into a vector
    let entailed_edges: Result<Vec<_>, _> = rows.collect();

    // Return the vector of s-p-o s.
    entailed_edges
}

pub fn get_labels(
    path: &str,
    terms: &[TermID],
) -> Result<HashMap<TermID, String>, rusqlite::Error> {
    let joined_terms = format!("'{}'", terms.join("', '"));
    let query = format!("SELECT subject, value FROM {STATEMENTS_TABLE} WHERE predicate='rdfs:label' AND subject IN ({joined_terms})");

    // Open a connection to the SQLite database file
    let conn = Connection::open(path)?;

    // Prepare the SQL query
    let mut stmt = conn.prepare(&query)?;

    // Execute the SQL query and retrieve the results
    let rows = stmt.query_map([], |row| {
        // Access the columns of each row
        Ok((row.get(0)?, row.get(1)?))
    })?;

    // Create a HashMap to store the results
    let mut result_map = HashMap::new();

    // Iterate over the rows and populate the HashMap
    for row in rows {
        let (key, value) = row?;
        result_map.insert(key, value);
    }

    Ok(result_map)
}

pub fn get_associations(
    path: &str,
    subjects: Option<&[TermID]>,
    predicates: Option<&[TermID]>,
    objects: Option<&[TermID]>,
) -> Result<HashMap<TermID, Vec<TermAssociation>>, rusqlite::Error> {
    let joined_subjects = match subjects {
        Some(subjects) => format!("'{}'", subjects.join("', '")),
        None => String::new(),
    };
    let joined_predicates = match predicates {
        Some(predicates) => format!("'{}'", predicates.join("', '")),
        None => String::new(),
    };
    let joined_objects = match objects {
        Some(objects) => format!("'{}'", objects.join("', '")),
        None => String::new(),
    };

    // Open a connection to the SQLite database file
    let conn = Connection::open(path)?;

    // Depending on which variables are not empty between subjects, predicates and objects, query the table
    // table_name where these the columns subject, predicate and object have at least one value in
    // the corresponding plural variable name.

    let mut query = String::from("SELECT DISTINCT * FROM ");
    query.push_str(TERM_ASSOCIATION_TABLE);

    if subjects.is_some() || predicates.is_some() || objects.is_some() {
        query.push_str(" WHERE ");

        if subjects.is_some() {
            query.push_str("subject IN (");
            query.push_str(&joined_subjects);
            query.push(')');
        }

        if predicates.is_some() {
            if subjects.is_some() {
                query.push_str(" AND ");
            }
            query.push_str("predicate IN (");
            query.push_str(&joined_predicates);
            query.push(')');
        }

        if objects.is_some() {
            if subjects.is_some() || predicates.is_none() {
                query.push_str(" AND ");
            }
            query.push_str("object IN (");
            query.push_str(&joined_objects);
            query.push(')');
        }
    }
    query.push(';');

    // Prepare the SQL query
    let mut stmt = conn.prepare(&query)?;

    // Execute the SQL query and retrieve the results
    let rows = stmt.query_map([], |row| {
        Ok(TermAssociation {
            id: row.get(0)?,
            subject: row.get(1)?,
            predicate: row.get(2)?,
            object: row.get(3)?,
            evidence_type: row.get(4)?,
            publication: row.get(5)?,
            source: row.get(6)?,
        })
    })?;

    // Create a HashMap to store the results
    let mut result_map: HashMap<TermID, Vec<TermAssociation>> = HashMap::new();

    // Iterate over the rows which is of type Rows and populate the HashMap
    for row in rows {
        let term_assoc = row?;
        let subject = &term_assoc.subject;

        // Check if the subject already exists in the HashMap
        if let Some(associations) = result_map.get_mut(subject) {
            associations.push(term_assoc);
        } else {
            // If the subject doesn't exist, create a new entry in the HashMap
            result_map.insert(subject.to_owned(), vec![term_assoc]);
        }
    }

    Ok(result_map)
}

pub fn get_objects_for_subjects(
    path: &str,
    subjects: Option<&[TermID]>,
    predicates: Option<&[TermID]>,
) -> Result<HashSet<TermID>, rusqlite::Error> {
    let joined_subjects = match subjects {
        Some(subjects) => format!("'{}'", subjects.join("', '")),
        None => String::new(),
    };
    let joined_predicates = match predicates {
        Some(predicates) => format!("'{}'", predicates.join("', '")),
        None => String::new(),
    };
    // Open a connection to the SQLite database file
    let conn = Connection::open(path)?;
    let mut query = String::from("SELECT DISTINCT object FROM ");
    query.push_str(TERM_ASSOCIATION_TABLE);
    if subjects.is_some() || predicates.is_some() {
        query.push_str(" WHERE ");

        if subjects.is_some() {
            query.push_str("subject IN (");
            query.push_str(&joined_subjects);
            query.push(')');
        }

        if predicates.is_some() {
            if subjects.is_some() {
                query.push_str(" AND ");
            }
            query.push_str("predicate IN (");
            query.push_str(&joined_predicates);
            query.push(')');
        }
    }
    query.push(';');

    // Prepare the SQL query
    let mut stmt = conn.prepare(&query)?;

    // Execute the SQL query and retrieve the results
    let rows = stmt.query_map([], |row| row.get::<_, TermID>(0))?;

    // Create a HashSet to store the results
    let mut result_set: HashSet<TermID> = HashSet::new();

    // Iterate over the rows which is of type Rows and populate the HashSet
    for row in rows {
        let object = row?;
        result_set.insert(object);
    }

    Ok(result_set)
}

pub fn get_subjects(
    path: &str,
    predicates: Option<&Vec<TermID>>,
    prefixes: Option<&Vec<TermID>>,
) -> Result<HashSet<TermID>, rusqlite::Error> {
    let joined_predicates = match predicates {
        Some(predicates) => format!("'{}'", predicates.join("', '")),
        None => String::new(),
    };
    let conditions = match prefixes {
        Some(prefixes) => {
            let mut conditions = String::with_capacity(prefixes.len() * 20); // Preallocate memory
            for (i, prefix) in prefixes.iter().enumerate() {
                if i > 0 {
                    conditions.push_str(" OR ");
                }
                conditions.push_str(&format!("subject LIKE '{}%'", prefix));
            }
            format!("WHERE {}", conditions)
        }
        None => String::new(),
    };

    // Build the final query
    let mut query = format!(
        "SELECT DISTINCT subject FROM {} {} ",
        TERM_ASSOCIATION_TABLE, conditions
    );
    if predicates.is_some() {
        query.push_str("AND predicate IN (");
        query.push_str(&joined_predicates);
        query.push_str(");");
    }

    // Open a connection to the SQLite database file
    let conn = Connection::open(path)?;

    // Prepare the SQL query
    let mut stmt = conn.prepare(&query)?;

    // Collect the subjects into a HashSet<TermID>
    let subjects: HashSet<TermID> = stmt
        .query_and_then([], |row| row.get::<_, String>(0))
        .unwrap()
        .map(|result| result.unwrap())
        .collect();

    Ok(subjects)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lazy_static::lazy_static;

    // Declare a common variable
    lazy_static! {
        static ref DB_PATH: String = "tests/data/go-nucleus.db".to_string();
        static ref PREDICATE_VEC: Vec<String> = vec!["rdfs:subClassOf".to_string()];
    }

    #[test]
    fn test_get_entailed_edges_for_predicate_list() {
        let db = &DB_PATH;
        let expected_length: usize = 1302;

        // Call the function with the test parameters
        let result = get_entailed_edges_for_predicate_list(db, &PREDICATE_VEC);

        // dbg!(&result);
        // Assert that the function executed successfully
        assert_eq!(result.unwrap().len(), expected_length);
    }

    #[test]
    fn test_get_labels() {
        let db = &DB_PATH;
        let expected_result: HashMap<String, String> =
            HashMap::from([("GO:0099568".to_string(), "cytoplasmic region".to_string())]);
        let result = get_labels(db, &["GO:0099568".to_string()]);
        assert_eq!(result.unwrap(), expected_result);
    }

    #[test]
    fn test_get_associations() {
        let db = &DB_PATH;
        let subject = "GO:0005737";
        let predicate = "biolink:has_nucleus";
        let object = "GO:0005622";
        // Test case 1: Query with non-empty subjects, predicates, and objects
        let result = get_associations(
            db,
            Some(&[subject.to_string()]),
            Some(&[predicate.to_string()]),
            Some(&[object.to_string()]),
        );
        assert!(result.is_ok());
        let associations = result.unwrap();
        // dbg!(associations);
        assert_eq!(associations.len(), 1);
        assert!(associations.contains_key(subject));

        // Test case 2: Query with one subject only
        let result = get_associations(db, Some(&[subject.to_string()]), None, None);

        assert!(result.is_ok());
        let associations = result.unwrap();
        dbg!(&associations);
        assert_eq!(associations.len(), 1);
        assert!(associations.contains_key(subject));

        // Test case 3: Query with no triple. Should return all!
        let result = get_associations(db, None, None, None);
        assert!(result.is_ok());
        let associations = result.unwrap();
        dbg!(&associations.len());
        assert_eq!(associations.len(), 258);

        // Test case 4: Query that yields no results
        let result = get_associations(db, Some(&["GO:51338".to_string()]), None, None);
        // Check the result
        assert!(result.is_ok());
        let associations = result.unwrap();
        assert_eq!(associations.len(), 0);

        // Test case 5: Query with two subjects and a predicate
        let result = get_associations(
            db,
            Some(&[subject.to_string(), "GO:0005622".to_string()]),
            Some(&[predicate.to_string()]),
            None,
        );
        assert!(result.is_ok());
        let associations = result.unwrap();
        dbg!(&associations);
        assert_eq!(associations.len(), 2);
        assert!(associations.contains_key("GO:0005622"));
    }

    #[test]
    fn test_get_subjects_with_prefixes() {
        let db = &DB_PATH;
        let prefixes = vec!["BFO:".to_string()];

        let result = get_subjects(db, None, Some(&prefixes));
        dbg!(&result);
        assert_eq!(result.unwrap().len(), 13)
    }

    #[test]
    fn test_get_subjects_with_prefixes_and_predicates() {
        let db = &DB_PATH;
        let prefixes = vec!["BFO:".to_string()];
        let predicates = vec!["biolink:has_nucleus".to_string()];

        let result = get_subjects(db, Some(&predicates), Some(&prefixes));
        dbg!(&result);
        assert_eq!(result.unwrap().len(), 13)
    }

    #[test]
    fn test_get_subjects_without_prefixes() {
        let db = &DB_PATH;
        let result = get_subjects(db, None, None);
        assert_eq!(result.unwrap().len(), 258)
    }

    #[test]
    fn test_get_objects_for_subjects() {
        let db = &DB_PATH;
        let expected_set = HashSet::from(["GO:0065007".to_string(), "GO:0008150".to_string()]);
        let result = get_objects_for_subjects(
            db,
            Some(&["GO:0050789".to_string()]),
            Some(&["biolink:has_nucleus".to_string()]),
        );
        assert!(result.is_ok());
        let set = result.unwrap();
        assert_eq!(set, expected_set);
    }
}
