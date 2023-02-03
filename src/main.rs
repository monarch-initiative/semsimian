use std::collections::{HashSet, HashMap};

fn main() {
    let set1: HashSet<&str> = ["apple", "banana", "cherry"].iter().cloned().collect();
    let set2: HashSet<&str> = ["banana", "cherry", "date"].iter().cloned().collect();

    // read in TSV file
    let mut rdr = csv::Reader::from_path("test_set.tsv").unwrap();

    // let mut dict_from_csv = HashMap::new();
    let mut dict_from_csv: HashMap<String, HashSet<String>> = HashMap::new();

    for result in rdr.records() {
        let record = result.unwrap();
        let name = &record[0];
        let fruit = &record[1];

        // println!("name: {}", name);
        // println!("fruit: {}", fruit);

        // dict_from_csv.add(name.to_string(), fruit.to_string());
        let n = dict_from_csv.entry(String::from(name));
        n.or_default().insert(String::from(fruit.to_string()));

        // if set1.contains(fruit) {
        //     println!("{} likes {}", name, fruit);
        // }
        // if set2.contains(fruit) {
        //     println!("{} likes {}", name, fruit);
        // }
    }

    // iterate over dict
    for (name, fruits) in &dict_from_csv {
        println!("CSV read as key : value => {} : {:?}", name, fruits);
    }

    // let similarity = jaccard_similarity(&set1, &set2);
    // println!("Jaccard similarity: {}", similarity);
}

fn jaccard_similarity(set1: &HashSet<&str>, set2: &HashSet<&str>) -> f64 {
    let intersection = set1.intersection(set2).count();
    let union_measure = set1.union(set2).count();
    intersection as f64 / union_measure as f64
}


