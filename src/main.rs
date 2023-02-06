use std::{collections::{HashSet, HashMap}, fs::File};
use csv::{ReaderBuilder, Reader};

fn main() {
    // let set1: HashSet<&str> = ["apple", "banana", "cherry"].iter().cloned().collect();
    // let set2: HashSet<&str> = ["banana", "cherry", "date"].iter().cloned().collect();

    /*
    read in TSV file
    csv::ReaderBuilder instead of just csv::Reader because we need to specify
    that the file has no headers.
    */
    // let closure_reader = read_file("closures.tsv");

    let data_dict = parse_associations(read_file("test_set.tsv"));
    let closures_dict = parse_associations(read_file("closures.tsv"));
    let ref_set = expand_terms_using_closure(data_dict.get("set1").unwrap(), &closures_dict);

    // iterate over dict
    for (name, foods) in &data_dict {
        println!("Original HashMap : key => {name} ; value: {foods:?}");
        let expanded_foods = expand_terms_using_closure(foods, &closures_dict);
        println!("Expanded HashMap : key => {name} ; value: {expanded_foods:?}");
        let score:f64 = jaccard_similarity(&ref_set, &expanded_foods);
        println!("Jaccard score : {score:?}")
    }
}

fn jaccard_similarity(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    let intersection = set1.intersection(set2).count();
    let union_measure = set1.union(set2).count();
    intersection as f64 / union_measure as f64
}

fn read_file(filename: &str) -> Reader<File> {
    ReaderBuilder::new().has_headers(false)
                        .from_path(filename)
                        .unwrap()
}

fn parse_associations(mut reader: Reader<File>) -> HashMap<String, HashSet<String>> {
    
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

fn expand_terms_using_closure(terms:&HashSet<String> , term_closure_map: &HashMap<String, HashSet<String>>) -> HashSet<String> {
    let mut expanded_set = HashSet::<String>::new();
    for item in terms.iter() {
        expanded_set.extend(term_closure_map.get(item).unwrap().clone());
    }
    expanded_set
}