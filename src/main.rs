
use std::{collections::{HashSet, HashMap}, fs::File};
use csv::{ReaderBuilder, Reader};

#[derive(Debug)]
struct TermSetPairwiseSimilarity {
    set_id: String,
    original_subject_termset: HashSet<String>,
    subject_termset: HashSet<String>,
    original_object_termset: HashSet<String>,
    object_termset: HashSet<String>,
    jaccard_similarity: f64,
}
impl TermSetPairwiseSimilarity {
    fn new() -> TermSetPairwiseSimilarity {
        TermSetPairwiseSimilarity {
            set_id: String::new(),
            original_subject_termset: HashSet::new(),
            subject_termset: HashSet::new(),
            original_object_termset: HashSet::new(),
            object_termset: HashSet::new(),
            jaccard_similarity: 0.0
        }
    }
}
fn main() {
    // let set1: HashSet<&str> = ["apple", "banana", "cherry"].iter().cloned().collect();
    // let set2: HashSet<&str> = ["banana", "cherry", "date"].iter().cloned().collect();

    /*
    read in TSV file
    csv::ReaderBuilder instead of just csv::Reader because we need to specify
    that the file has no headers.
    */

    let data_dict = parse_associations(read_file("test_set.tsv"));
    let closures_dict = parse_associations(read_file("closures.tsv"));
    let ref_set = data_dict.get("set1").unwrap();

    let mut tsps_information = TermSetPairwiseSimilarity::new();

    tsps_information.original_subject_termset = ref_set.clone();
    tsps_information.subject_termset = expand_terms_using_closure
                                        (
                                            &tsps_information.original_subject_termset,
                                            &closures_dict
                                        );
    // iterate over dict
    for (key, terms) in &data_dict {
        tsps_information.set_id = key.to_string();
        tsps_information.original_object_termset = terms.clone();
        tsps_information.object_termset = expand_terms_using_closure
                                        (
                                            &tsps_information.original_object_termset,
                                            &closures_dict
                                        );
        tsps_information.jaccard_similarity = calculate_jaccard_similarity
                                        (
                                            &tsps_information.subject_termset,
                                            &tsps_information.object_termset
                                        );
        println!("{tsps_information:#?}")
    }
}

fn calculate_jaccard_similarity(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    /* Returns Jaccard similarity between the two sets. */

    let intersection = set1.intersection(set2).count();
    let union_measure = set1.union(set2).count();
    intersection as f64 / union_measure as f64
}

fn read_file(filename: &str) -> Reader<File> {
    /* Build CSV reader from filepath.*/
    ReaderBuilder::new().has_headers(false)
                        .from_path(filename)
                        .unwrap()
}

fn parse_associations(mut reader: Reader<File>) -> HashMap<String, HashSet<String>> {
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

fn expand_terms_using_closure(
        terms:&HashSet<String> ,
        term_closure_map: &HashMap<String, HashSet<String>>
    ) -> HashSet<String> {
    /* Expand terms by inclusing ancestors in the set. */
    let mut expanded_set = HashSet::<String>::new();
    for item in terms.iter() {
        expanded_set.extend(term_closure_map.get(item).unwrap().clone());
    }
    expanded_set
}