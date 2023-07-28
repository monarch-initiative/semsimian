use crate::{Predicate, TermID};
use rusqlite::{Connection, Result};

#[derive(Debug)]
#[allow(dead_code)]
pub struct EntailedEdges {
    subject: TermID,
    predicate: Predicate,
    object: TermID,
}

pub fn get_entailed_edges_for_predicate_list(
    path: &str,
    predicates: Vec<&str>,
) -> Result<Vec<EntailedEdges>, rusqlite::Error> {
    let table_name = "entailed_edge";

    // Build the SQL query with the provided table name such that 'predicates' are in the Vector predicates.
    let joined_predicates = format!("'{}'", predicates.join("', '"));
    let query = format!(
        "SELECT * FROM {} WHERE predicate IN ({})",
        table_name, joined_predicates
    );

    // Open a connection to the SQLite database file
    let conn = Connection::open(path)?;

    // Execute the SQL query and retrieve the results
    let mut stmt = conn.prepare(&query)?;

    let rows = stmt.query_map([], |row| {
        // Access the columns of each row
        Ok(EntailedEdges {
            subject: row.get(0)?,
            predicate: row.get(1)?,
            object: row.get(2)?,
        })
    })?;

    // Collect the results into a vector
    let entailed_edges: Result<Vec<_>, _> = rows.collect();

    // Return the vector of EntailedEdges structs
    entailed_edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_entailed_edges_for_predicate_list() {
        let db = "tests/data/go-nucleus.db";

        // Call the function with the test parameters
        let result = get_entailed_edges_for_predicate_list(db, vec!["rdfs:subClassOf"]);

        // Assert that the function executed successfully
        assert_eq!(result.unwrap().len(), 1302);
    }
}
