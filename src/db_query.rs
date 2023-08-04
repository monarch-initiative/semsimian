use std::collections::HashMap;

use crate::{Predicate, TermID};
use rusqlite::{Connection, Result};

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

pub fn get_labels(path: &str, terms: Vec<TermID>) -> Result<HashMap<TermID, String>, rusqlite::Error>{
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
        let result =
            get_entailed_edges_for_predicate_list(db, &PREDICATE_VEC);

        // dbg!(&result);
        // Assert that the function executed successfully
        assert_eq!(result.unwrap().len(), expected_length);
    }

    #[test]
    fn test_get_labels(){
        let db = &DB_PATH;
        let expected_result: HashMap<String, String> = HashMap::from([
            ("GO:0099568".to_string(),"cytoplasmic region".to_string())
        ]);
        let result = get_labels(db, vec!["GO:0099568".to_string()]);
        assert_eq!(result.unwrap(), expected_result);
    }
}
