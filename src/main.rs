use std::collections::{HashSet, HashMap};
use csv::ReaderBuilder;

fn main() {
    // let set1: HashSet<&str> = ["apple", "banana", "cherry"].iter().cloned().collect();
    // let set2: HashSet<&str> = ["banana", "cherry", "date"].iter().cloned().collect();

    /*
    read in TSV file
    csv::ReaderBuilder instead of just csv::Reader because we need to specify
    that the file has no headers.
    */
    let mut reader = ReaderBuilder::new().has_headers(false)
                                         .from_path("test_set.tsv")
                                         .unwrap();
    let mut dict_from_csv: HashMap<String, HashSet<String>> = HashMap::new();

    for result in reader.records() {
        let record = result.unwrap();
        let name = &record[0];
        let food = &record[1];
        let n = dict_from_csv.entry(name.to_string());
        n.or_default().insert(food.to_string());
    }

    // iterate over dict
    for (name, foods) in &dict_from_csv {
        println!("CSV read as key : value => {name} : {foods:?}");
        let score:f64 = jaccard_similarity(dict_from_csv.get("set1").unwrap(), &foods);
        println!("score : {score:?}")
    }
}

fn jaccard_similarity(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    let intersection = set1.intersection(set2).count();
    let union_measure = set1.union(set2).count();
    intersection as f64 / union_measure as f64
}


