use std::{collections::{HashSet, HashMap}, fs::File};
use csv::{ReaderBuilder, Reader};

pub fn read_file(filename: &str) -> Reader<File> {
    /* Build CSV reader from filepath.*/
    ReaderBuilder::new().has_headers(false)
                        .from_path(filename)
                        .unwrap()
}

pub fn parse_associations(mut reader: Reader<File>) -> HashMap<String, HashSet<String>> {
    /* Parse CSV files using ReaderBuilder.*/
    let mut dict_from_csv: HashMap<String, HashSet<String>> = HashMap::new();

    for result in reader.records() {
        let record = result.unwrap();
        let key = &record[0];
        let value = &record[1];
        let n = dict_from_csv.entry(key.to_string());
        n.or_default().insert(value.to_string());
    }
    dict_from_csv
}