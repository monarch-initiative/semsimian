use csv::{Reader, ReaderBuilder};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

pub fn read_file(filename: &Path) -> Reader<File> {
    match File::open(filename) {
        Ok(file) => ReaderBuilder::new().has_headers(false).from_reader(file),
        Err(err) => panic!("Failed to open file: {} {}", filename.display(), err),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_read_file() {
        let input_file = Path::new("./tests/data/test_set.tsv");
        let mut input_reader = read_file(input_file);
        assert_eq!(input_reader.records().count(), 11);
    }

    #[test]
    fn test_parse_associations() {
        let input_file = Path::new("./tests/data/test_set.tsv");
        let input_reader = read_file(input_file);
        let assoc_dict = parse_associations(input_reader);
        println!("Dicts are: {:?}", assoc_dict);
        assert_eq!(assoc_dict.len(), 3)
    }
}
