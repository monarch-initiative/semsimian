use std::collections::{HashMap, HashSet};

use crate::{Predicate, TermID};
use rusqlite::{Connection, Result};

#[derive(Debug)]
pub struct TermAssociation {
    pub id: TermID,
    pub subject: TermID,
    pub predicate: TermID,
    pub object: TermID,
    pub evidence_type: TermID,
    pub publication: TermID,
    pub source: TermID,
}

pub fn get_entailed_edges_for_predicate_list(
    path: &str,
    predicates: &Vec<Predicate>,
) -> Result<Vec<(TermID, Predicate, TermID)>, rusqlite::Error> {
    let table_name = "entailed_edge";

    // Build the SQL query with the provided table name such that 'predicates' are in the Vector predicates.
    let joined_predicates = format!("'{}'", predicates.join("', '"));

    let query = if !predicates.is_empty() {
        format!(
            "SELECT * FROM {} WHERE predicate IN ({})",
            table_name, joined_predicates
        )
    } else {
        format!("SELECT * FROM {}", table_name)
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
    let table_name = "statements";
    let joined_terms = format!("'{}'", terms.join("', '"));
    let query = format!("SELECT subject, value FROM {table_name} WHERE predicate='rdfs:label' AND subject IN ({joined_terms})");

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
) -> Result<Vec<TermAssociation>, rusqlite::Error> {
    let table_name = "term_association";
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

    let mut query = String::from("SELECT * FROM ");
    query.push_str(table_name);

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
    dbg!(&query);
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
    let mut result_vec: Vec<TermAssociation> = Vec::new();

    // Iterate over the rows which is of type Rows and populate the HashMap
    for row in rows {
        let term_assoc = row?;
        result_vec.push(term_assoc);
    }

    Ok(result_vec)
}

pub fn get_subjects(
    path: &str,
    prefixes: Option<Vec<TermID>>,
) -> Result<HashSet<TermID>, rusqlite::Error> {
    let table_name = "term_association";
    let conditions = match prefixes {
        Some(prefixes) => {
            let conditions: String = prefixes
                .iter()
                .map(|prefix| format!("subject LIKE '{}%'", prefix))
                .collect::<Vec<String>>()
                .join(" OR ");
            format!("WHERE {}", conditions)
        }
        None => String::new(),
    };

    // Build the final query
    let query = format!("SELECT subject FROM {} {};", table_name, conditions);

    // Open a connection to the SQLite database file
    let conn = Connection::open(path)?;

    // Prepare the SQL query
    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

    // Collect the subjects into a HashSet<TermID>
    let subjects: HashSet<TermID> = rows.map(|row| row.unwrap()).collect();

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

        dbg!(&result);
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
        let subject = "GO:0004857";
        let predicate = "biolink:subclass_of";
        let object = "GO:0003674";
        // Test case 1: Query with non-empty subjects, predicates, and objects
        let result = get_associations(
            db,
            Some(&[subject.to_string()]),
            Some(&[predicate.to_string()]),
            Some(&[object.to_string()]),
        );
        assert!(result.is_ok());
        let associations = result.unwrap();
        dbg!(&associations);
        assert_eq!(associations.len(), 1);
        assert_eq!(associations[0].subject, subject);
        assert_eq!(associations[0].predicate, predicate);
        assert_eq!(associations[0].object, object);

        // Test case 2: Query with one subject only
        let result = get_associations(db, Some(&[subject.to_string()]), None, None);

        assert!(result.is_ok());
        let associations = result.unwrap();
        dbg!(&associations);
        assert_eq!(associations.len(), 3);

        // Test case 3: Query with no triple. Should return all!
        let result = get_associations(db, None, None, None);
        assert!(result.is_ok());
        let associations = result.unwrap();
        dbg!(&associations.len());
        assert_eq!(associations.len(), 464);
    }
}
