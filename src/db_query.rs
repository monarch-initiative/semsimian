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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_entailed_edges_for_predicate_list() {
        let db = "tests/data/go-nucleus.db";
        let expected_length:usize = 1302;

        // Call the function with the test parameters
        let result =
            get_entailed_edges_for_predicate_list(db, &vec!["rdfs:subClassOf".to_string()]);

        // Assert that the function executed successfully
        assert_eq!(result.unwrap().len(), expected_length);
    }
}
