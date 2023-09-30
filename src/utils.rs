use indicatif::{ProgressBar, ProgressStyle};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Path;

use csv::{ReaderBuilder, WriterBuilder};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};

use crate::db_query::get_subjects;
use crate::termset_pairwise_similarity::TermsetPairwiseSimilarity;
use crate::SimilarityMap;
type Predicate = String;
type TermID = String;
type PredicateSetKey = String;
type ClosureMap = HashMap<String, HashMap<String, HashSet<String>>>;
type ICMap = HashMap<String, HashMap<String, f64>>;
type BTreeInBTree = BTreeMap<String, BTreeMap<String, String>>;

pub fn predicate_set_to_key(predicates: &Option<Vec<Predicate>>) -> PredicateSetKey {
    let mut result = String::new();

    if predicates.is_none() {
        result.push_str("_all");
    } else {
        let mut vec_of_predicates: Vec<String> = predicates
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.to_string())
            .collect();
        vec_of_predicates.sort();

        for predicate in vec_of_predicates {
            result.push('+');
            result.push_str(&predicate);
        }
    }
    // println!("Returning key: {}", result); // for debugging

    result
}

pub fn convert_set_to_hashmap(set1: &HashSet<String>) -> HashMap<i32, String> {
    set1.iter()
        .enumerate()
        .map(|(idx, item)| (idx as i32 + 1, item.clone()))
        .collect()
}

pub fn numericize_sets(
    set1: &HashSet<String>,
    set2: &HashSet<String>,
) -> (HashSet<i32>, HashSet<i32>, HashMap<i32, String>) {
    let union_set: HashSet<_> = set1.union(set2).cloned().collect();
    let union_set_hashmap = convert_set_to_hashmap(&union_set);

    let num_set1: HashSet<_> = union_set_hashmap
        .iter()
        .filter(|(_, v)| set1.contains(*v))
        .map(|(k, _)| *k)
        .collect();

    let num_set2: HashSet<_> = union_set_hashmap
        .iter()
        .filter(|(_, v)| set2.contains(*v))
        .map(|(k, _)| *k)
        .collect();

    (num_set1, num_set2, union_set_hashmap)
}

pub fn _stringify_sets_using_map(
    set1: &HashSet<i32>,
    set2: &HashSet<i32>,
    map: &HashMap<i32, String>,
) -> (HashSet<String>, HashSet<String>) {
    let str_set1: HashSet<_> = map
        .iter()
        .filter_map(|(k, v)| {
            if set1.contains(k) {
                Some(v.clone())
            } else {
                None
            }
        })
        .collect();

    let str_set2: HashSet<_> = map
        .iter()
        .filter_map(|(k, v)| {
            if set2.contains(k) {
                Some(v.clone())
            } else {
                None
            }
        })
        .collect();

    (str_set1, str_set2)
}

pub fn convert_list_of_tuples_to_hashmap(
    list_of_tuples: &Vec<(TermID, PredicateSetKey, TermID)>,
    predicates: &Option<Vec<String>>,
) -> (ClosureMap, ICMap) {
    let mut closure_map: HashMap<String, HashMap<String, HashSet<String>>> =
        HashMap::with_capacity(list_of_tuples.len());
    let mut freq_map: HashMap<String, usize> = HashMap::with_capacity(list_of_tuples.len());
    let mut ic_map: HashMap<String, HashMap<String, f64>> = HashMap::new();

    let predicate_set_key: PredicateSetKey = predicate_set_to_key(predicates);

    let progress_bar = generate_progress_bar_of_length_and_message(
        list_of_tuples.len() as u64,
        "Building closure and IC map:",
    );

    for (s, p, o) in list_of_tuples.iter() {
        if let Some(predicates) = predicates {
            if !predicates.contains(p) {
                continue;
            }
        }
        // ! As per this below, the frequency map gets populated ONLY if the node is an object (o)
        // ! in the (s, p, o). If the node is a subject (s), it does not count towards the frequency.
        // ! Only with this implemented will the results match with `oaklib`'s `sqlite` implementation
        // ! of semantic similarity.
        *freq_map.entry(o.clone()).or_insert(0) += 1;

        closure_map
            .entry(predicate_set_key.clone())
            .or_insert_with(HashMap::new)
            .entry(String::from(s))
            .or_insert_with(HashSet::new)
            .insert(String::from(o));

        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("done");

    let number_of_nodes = freq_map.len() as f64;

    ic_map
        .entry(predicate_set_key.clone())
        .or_insert_with(HashMap::new)
        .extend(
            freq_map
                .iter()
                .map(|(k, v)| (String::from(k), -(*v as f64 / number_of_nodes).log2())),
        );

    (closure_map, ic_map)
}

pub fn expand_term_using_closure(
    term: &str,
    closure_table: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    predicates: &Option<Vec<Predicate>>,
) -> HashSet<TermID> {
    let mut ancestors: HashSet<String> = HashSet::new();
    let mut this_predicate_set_key = predicate_set_to_key(predicates);
    if this_predicate_set_key == "_all" {
        let closure_table_keys: Vec<String> = closure_table.keys().cloned().collect();
        this_predicate_set_key = closure_table_keys.join("+");
    }

    for (closure_predicate_key, closure_map) in closure_table.iter() {
        if *closure_predicate_key == this_predicate_set_key {
            if let Some(ancestors_for_predicates) = closure_map.get(term) {
                ancestors.extend(ancestors_for_predicates.clone());
            }
        }
    }
    ancestors
}

pub fn generate_progress_bar_of_length_and_message(length: u64, message: &str) -> ProgressBar {
    let progress_bar = ProgressBar::new(length);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "[{{elapsed_precise}}] {message} {{bar:40.cyan/blue}} {{percent}}%"
            ))
            .unwrap(),
    );
    progress_bar
}

pub fn find_embedding_index(embeddings: &[(String, Vec<f64>)], node: &str) -> Option<usize> {
    embeddings.iter().position(|(curie, _)| curie == node)
}

pub fn rearrange_columns_and_rewrite(
    filename: &str,
    sequence: Vec<String>,
) -> Result<(), Box<dyn Error>> {
    // Get the parent directory of the input file
    let parent_dir = Path::new(filename).parent().ok_or("Invalid file path")?;

    // Create a temporary file in the same directory as the input file
    let temp_filename = parent_dir.join("temp_file.tmp");
    let temp_file = File::create(&temp_filename)?;

    // Read the TSV file into a CSV reader
    let file = File::open(filename)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(BufReader::new(file));

    // Get the header row from the reader
    let headers = reader.headers()?.clone();

    // Rearrange the columns based on the provided sequence
    let indices: Vec<usize> = sequence
        .iter()
        .map(|col| headers.iter().position(|h| h == col))
        .collect::<Option<_>>()
        .unwrap_or_else(|| {
            panic!("One or more columns not found in the input file");
        });

    // Create a CSV writer for the temporary file
    let mut writer = WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(BufWriter::new(temp_file));

    // Write the rearranged header row
    writer.write_record(indices.iter().map(|&i| headers.get(i).unwrap()))?;

    // Write the remaining rows with rearranged columns
    for result in reader.records() {
        let record = result?;
        let rearranged_record: Vec<_> = indices.iter().map(|&i| record.get(i).unwrap()).collect();
        writer.write_record(rearranged_record)?;
    }

    // Flush and close the writer
    writer.flush()?;
    drop(writer);

    // Close the input file
    drop(reader);

    // Replace the input file with the temporary file
    if Path::new(filename).exists() {
        fs::remove_file(filename)?;
    }
    fs::rename(&temp_filename, filename)?;

    Ok(())
}

pub fn get_termset_vector(
    terms: &HashSet<String>,
    term_label_hashmap: &HashMap<String, String>,
) -> Vec<BTreeInBTree> {
    let filtered_keys: Vec<&String> = term_label_hashmap
        .keys()
        .filter(|key| terms.contains(*key))
        .collect();

    let mut termset_vector = Vec::with_capacity(filtered_keys.len());

    for key in filtered_keys {
        if let Some(value) = term_label_hashmap.get(key) {
            let inner_btreemap = BTreeMap::from_iter(vec![
                ("id".to_string(), key.clone()),
                ("label".to_string(), value.clone()),
            ]);

            let mut outer_btreemap = BTreeMap::new();
            outer_btreemap.insert(key.clone(), inner_btreemap);

            termset_vector.push(outer_btreemap);
        }
    }

    termset_vector
}

pub fn get_similarity_map(
    term_id: &str,
    best_match: (&String, &(f64, f64, f64, f64, HashSet<String>)),
) -> BTreeMap<String, String> {
    let mut similarity_map = BTreeMap::new();
    if let Some((key, value)) = Some(best_match) {
        similarity_map.insert("jaccard_similarity".to_string(), value.0.to_string());
        similarity_map.insert(
            "ancestor_information_content".to_string(),
            value.1.to_string(),
        );
        similarity_map.insert("phenodigm_score".to_string(), value.2.to_string());
        similarity_map.insert("cosine_similarity".to_string(), value.3.to_string());
        similarity_map.insert("subject_id".to_string(), term_id.to_string());
        similarity_map.insert("object_id".to_string(), key.clone());

        if let Some(ancestor_id) = value.4.iter().next() {
            similarity_map.insert("ancestor_id".to_string(), ancestor_id.clone());
        } else {
            similarity_map.insert("ancestor_id".to_string(), "NO_ANCESTOR_FOUND".to_string());
        }
    } else {
        println!("The HashMap is empty.");
    }

    similarity_map
}

// TODO: Revisit par_iter()
// use rayon::prelude::*;
// use std::sync::{Arc, Mutex};
// pub fn get_best_matches(
//     termset: &Vec<BTreeInBTree>,
//     all_by_all: &SimilarityMap,
//     term_label_map: &HashMap<String, String>,
//     metric: &str,
// ) -> (BTreeInBTree, BTreeInBTree) {
//     let best_matches = Arc::new(Mutex::new(BTreeMap::new()));
//     let best_matches_similarity_map = Arc::new(Mutex::new(BTreeMap::new()));

//     termset.par_iter().for_each(|term| {
//         let term_id = term.keys().next().unwrap();
//         let term_label = &term[term_id]["label"];

//         if let Some(matches) = all_by_all.get(term_id) {
//             let best_match = matches
//                 .iter()
//                 .max_by(|(_, (_, v1, _, _, _)), (_, (_, v2, _, _, _))| v1.partial_cmp(v2).unwrap())
//                 .unwrap();

//             let mut similarity_map = get_similarity_map(term_id, best_match);

//             let ancestor_id = similarity_map.get("ancestor_id").unwrap().clone();
//             let ancestor_label = term_label_map
//                 .get(&ancestor_id)
//                 .cloned()
//                 .unwrap_or_default();
//             let score = similarity_map.get(metric).unwrap().clone();

//             let match_source = term_id;
//             let match_source_label = term_label;
//             let match_target = similarity_map.get("object_id").unwrap().clone();
//             let match_target_label = term_label_map.get(&match_target).unwrap().clone();

//             similarity_map.insert("ancestor_label".to_string(), ancestor_label);
//             let best_matches_key = term_id.to_owned();
//             let mut best_matches_value: BTreeMap<String, String> = BTreeMap::new();
//             // best_matches_value.insert("similarity".to_string(), Box::new(similarity_map.clone()));
//             best_matches_value.insert("match_source".to_string(), match_source.to_owned());
//             best_matches_value.insert(
//                 "match_source_label".to_string(),
//                 match_source_label.to_owned(),
//             );
//             best_matches_value.insert("match_target".to_string(), match_target);
//             best_matches_value.insert("match_target_label".to_string(), match_target_label);
//             best_matches_value.insert("score".to_string(), score);

//             let mut best_matches_guard = best_matches.lock().unwrap();
//             best_matches_guard.insert(best_matches_key.clone(), best_matches_value);

//             let mut best_matches_similarity_map_guard = best_matches_similarity_map.lock().unwrap();
//             best_matches_similarity_map_guard.insert(best_matches_key, similarity_map);
//         }
//     });

//     let best_matches_guard = Arc::try_unwrap(best_matches).unwrap().into_inner().unwrap();
//     let best_matches_similarity_map_guard = Arc::try_unwrap(best_matches_similarity_map)
//         .unwrap()
//         .into_inner()
//         .unwrap();

//     (best_matches_guard, best_matches_similarity_map_guard)
// }

pub fn get_best_matches(
    termset: &[BTreeInBTree],
    all_by_all: &SimilarityMap,
    term_label_map: &HashMap<String, String>,
    metric: &str,
) -> (BTreeInBTree, BTreeInBTree) {
    let mut best_matches = BTreeMap::new();
    let mut best_matches_similarity_map = BTreeMap::new();

    for term in termset.iter() {
        let term_id = term.keys().next().unwrap();
        let term_label = &term[term_id]["label"];

        if let Some(matches) = all_by_all.get(term_id) {
            let best_match = matches
                .iter()
                .max_by(|(_, (_, v1, _, _, _)), (_, (_, v2, _, _, _))| v1.partial_cmp(v2).unwrap())
                .unwrap();

            let mut similarity_map = get_similarity_map(term_id, best_match);

            let ancestor_id = similarity_map.get("ancestor_id").unwrap().clone();
            let ancestor_label = term_label_map
                .get(&ancestor_id)
                .cloned()
                .unwrap_or_default();
            let score = similarity_map.get(metric).unwrap().clone();

            let match_source = term_id;
            let match_source_label = term_label;
            let match_target = similarity_map.get("object_id").unwrap().clone();
            let match_target_label = term_label_map
                .get(&match_target)
                .unwrap_or(&"NO_LABEL".to_string())
                .clone();

            similarity_map.insert("ancestor_label".to_string(), ancestor_label);
            let best_matches_key = term_id.to_owned();
            let mut best_matches_value: BTreeMap<String, String> = BTreeMap::new();
            // best_matches_value.insert("similarity".to_string(), Box::new(similarity_map.clone()));
            best_matches_value.insert("match_source".to_string(), match_source.to_owned());
            best_matches_value.insert(
                "match_source_label".to_string(),
                match_source_label.to_owned(),
            );
            best_matches_value.insert("match_target".to_string(), match_target);
            best_matches_value.insert("match_target_label".to_string(), match_target_label);
            best_matches_value.insert("score".to_string(), score);

            best_matches.insert(best_matches_key.clone(), best_matches_value);
            best_matches_similarity_map.insert(best_matches_key, similarity_map);
        }
    }

    (best_matches, best_matches_similarity_map)
}

pub fn get_best_score(
    subject_best_matches: &BTreeInBTree,
    object_best_matches: &BTreeInBTree,
) -> f64 {
    let max_score = [subject_best_matches, object_best_matches]
        .iter()
        .flat_map(|matches| matches.values())
        .filter_map(|matches| matches.get("score"))
        .filter_map(|score| score.parse::<f64>().ok())
        .fold(f64::NEG_INFINITY, |max_score, score_value| {
            max_score.max(score_value)
        });

    max_score
}

pub fn get_prefix_association_key(
    subject_prefixes: &[TermID],
    object_closure_predicates: &HashSet<TermID>,
    quick_search_flag: &bool,
) -> String {
    // Convert subject_prefixes to a sorted string
    let subject_prefixes_string = subject_prefixes
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<String>>()
        .join("+");

    // Convert object_closure_predicates to a sorted string
    let object_closure_predicates_string = {
        let mut sorted_predicates = object_closure_predicates
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<String>>();
        sorted_predicates.sort();
        sorted_predicates.join("+")
    };

    // Concatenate subject_prefixes_string , object_closure_predicates_string and quick_search_flag
    subject_prefixes_string + &object_closure_predicates_string + &quick_search_flag.to_string()
}

pub fn get_curies_from_prefixes(
    prefixes: Option<&Vec<TermID>>,
    predicates: &Vec<TermID>,
    resource_path: &str,
) -> Vec<TermID> {
    let curies_set = get_subjects(resource_path, Some(predicates), prefixes)
        .unwrap_or_else(|_| panic!("Failed to get curies from prefixes"));

    let curies_vec: Vec<TermID> = curies_set.into_iter().collect();
    curies_vec
}

// Function to create a seeded hash
pub fn seeded_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

pub fn hashed_dual_sort(
    mut result: Vec<(f64, Option<TermsetPairwiseSimilarity>, String)>,
) -> Vec<(f64, Option<TermsetPairwiseSimilarity>, String)> {
    // Sort the result vector by score in descending order and hash of result CURIE in ascending order
    result.sort_by(|a, b| {
        let primary = b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal);
        let secondary = seeded_hash(&a.2).cmp(&seeded_hash(&b.2));
        primary.then(secondary)
    });
    result
}

pub fn sort_with_jaccard_as_tie_breaker(
    mut vec_to_sort: Vec<(f64, Option<TermsetPairwiseSimilarity>, TermID)>,
    flatten_result: &[(f64, Option<TermsetPairwiseSimilarity>, TermID)],
) -> Vec<(f64, Option<TermsetPairwiseSimilarity>, TermID)> {
    let flatten_result_hash: HashMap<_, _> = flatten_result
        .iter()
        .map(|x| (x.2.clone(), x))
        .collect();

    vec_to_sort.sort_unstable_by(|a, b| {
        let score_a = a.0;
        let score_b = b.0;

        if score_a == score_b {
            let tie_breaker_a = flatten_result_hash
                .get(&a.2)
                .unwrap_or(&&(0.0, None, "".to_string()))
                .0;
            let tie_breaker_b = flatten_result_hash
                .get(&b.2)
                .unwrap_or(&&(0.0, None, "".to_string()))
                .0;
            // If the Jaccard score also results in a tie, then consider `seeded_hash` of the term CURIE
            if tie_breaker_a == tie_breaker_b {
                seeded_hash(&b.2).cmp(&seeded_hash(&a.2))
            } else {
                tie_breaker_b.partial_cmp(&tie_breaker_a).unwrap()
            }
        } else {
            score_b.partial_cmp(&score_a).unwrap()
        }
    });

    vec_to_sort
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{Read, Write},
    };

    use super::*;
    #[test]

    fn test_convert_set_to_hashmap() {
        let set: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("mango"),
            String::from("grapes"),
        ]);

        assert_eq!(set.len(), convert_set_to_hashmap(&set).len());
    }

    #[test]
    fn test_numericize_set() {
        let set1: HashSet<String> = HashSet::from([
            String::from("grapes"),
            String::from("blueberry"),
            String::from("fruit"),
            String::from("blackberry"),
        ]);
        let set2: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("fruit"),
            String::from("tropical"),
        ]);

        let (num_set1, num_set2, _) = numericize_sets(&set1, &set2);

        assert_eq!(set1.len(), num_set1.len());
        assert_eq!(set2.len(), num_set2.len());
    }

    #[test]
    fn test_stringify_sets_using_map() {
        let set1: HashSet<i32> = HashSet::from([1, 2, 3, 4, 5]);
        let set2: HashSet<i32> = HashSet::from([3, 4, 5, 6, 7]);
        let map = HashMap::from([
            (1_i32, String::from("apple")),
            (2_i32, String::from("banana")),
            (3_i32, String::from("orange")),
            (4_i32, String::from("blueberry")),
            (5_i32, String::from("blackberry")),
            (6_i32, String::from("grapes")),
            (7_i32, String::from("fruits")),
        ]);
        let (str_set1, str_set2) = _stringify_sets_using_map(&set1, &set2, &map);
        assert_eq!(set1.len(), str_set1.len());
        assert_eq!(set2.len(), str_set2.len());
    }

    #[test]
    fn test_str_to_int_to_back() {
        let set1: HashSet<String> = HashSet::from([
            String::from("grapes"),
            String::from("blueberry"),
            String::from("fruit"),
            String::from("blackberry"),
        ]);
        let set2: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("fruit"),
            String::from("tropical"),
        ]);

        let (num_set1, num_set2, map) = numericize_sets(&set1, &set2);
        let (str_set1, str_set2) = _stringify_sets_using_map(&num_set1, &num_set2, &map);

        assert_eq!(set1, str_set1);
        assert_eq!(set2, str_set2);
    }

    #[test]
    fn test_convert_list_of_tuples_to_hashmap() {
        let list_of_tuples: Vec<(TermID, Predicate, TermID)> = vec![
            (
                String::from("ABCD:123"),
                String::from("is_a"),
                String::from("BCDE:234"),
            ),
            (
                String::from("ABCD:123"),
                String::from("part_of"),
                String::from("ABCDE:1234"),
            ),
            (
                String::from("XYZ:123"),
                String::from("is_a"),
                String::from("WXY:234"),
            ),
            (
                String::from("XYZ:123"),
                String::from("part_of"),
                String::from("WXYZ:1234"),
            ),
        ];

        // test closure map for is_a predicates
        let expected_closure_map_is_a: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::from([(
                String::from("+is_a"),
                HashMap::from([
                    (
                        String::from("ABCD:123"),
                        [String::from("BCDE:234")]
                            .iter()
                            .cloned()
                            .collect::<HashSet<_>>(),
                    ),
                    (
                        String::from("XYZ:123"),
                        [String::from("WXY:234")]
                            .iter()
                            .cloned()
                            .collect::<HashSet<_>>(),
                    ),
                ]),
            )]);

        let predicates_is_a: Option<Vec<Predicate>> =
            Some(["is_a"].iter().map(|&s| s.to_string()).collect());
        let (closure_map_is_a, _) =
            convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates_is_a);
        assert_eq!(expected_closure_map_is_a, closure_map_is_a);

        // test closure_map for is_a + part_of predicates
        let expected_closure_map_is_a_plus_part_of: HashMap<
            PredicateSetKey,
            HashMap<TermID, HashSet<TermID>>,
        > = HashMap::from([(
            String::from("+is_a+part_of"),
            HashMap::from([
                (
                    String::from("ABCD:123"),
                    [String::from("BCDE:234"), String::from("ABCDE:1234")]
                        .iter()
                        .cloned()
                        .collect::<HashSet<TermID>>(),
                ),
                (
                    String::from("XYZ:123"),
                    [String::from("WXY:234"), String::from("WXYZ:1234")]
                        .iter()
                        .cloned()
                        .collect::<HashSet<TermID>>(),
                ),
            ]),
        )]);

        let predicates_is_a_plus_part_of: Option<Vec<Predicate>> =
            Some(["is_a", "part_of"].iter().map(|&s| s.to_string()).collect());
        let (closure_map_is_a_plus_part_of, ic_map) =
            convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates_is_a_plus_part_of);
        assert_eq!(
            expected_closure_map_is_a_plus_part_of,
            closure_map_is_a_plus_part_of
        );

        let expected_ic_map_is_a_plus_part_of: HashMap<PredicateSetKey, HashMap<TermID, f64>> = {
            let mut expected: HashMap<TermID, f64> = HashMap::new();
            // expected.insert(String::from("ABCD:123"), -(0.0 / 6 as f64).log2());
            expected.insert(String::from("BCDE:234"), -(1.0 / 4_f64).log2());
            expected.insert(String::from("ABCDE:1234"), -(1.0 / 4_f64).log2());
            // expected.insert(String::from("XYZ:123"), -(0.0 / 6 as f64).log2());
            expected.insert(String::from("WXY:234"), -(1.0 / 4_f64).log2());
            expected.insert(String::from("WXYZ:1234"), -(1.0 / 4_f64).log2());

            let mut expected_ic_map_is_a_plus_part_of: HashMap<
                PredicateSetKey,
                HashMap<TermID, f64>,
            > = HashMap::new();
            expected_ic_map_is_a_plus_part_of.insert(String::from("+is_a+part_of"), expected);
            expected_ic_map_is_a_plus_part_of
        };

        assert_eq!(ic_map, expected_ic_map_is_a_plus_part_of);

        // Test closure map for None predicates
        let _expected_closure_map_none: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::from([(
                String::from("+is_a+part_of"),
                HashMap::from([
                    (
                        String::from("ABCD:123"),
                        [String::from("BCDE:234"), String::from("ABCDE:1234")]
                            .iter()
                            .cloned()
                            .collect::<HashSet<TermID>>(),
                    ),
                    (
                        String::from("XYZ:123"),
                        [String::from("WXY:234"), String::from("WXYZ:1234")]
                            .iter()
                            .cloned()
                            .collect::<HashSet<TermID>>(),
                    ),
                ]),
            )]);

        let predicates_none: Option<Vec<Predicate>> = None;
        println!("Passing predicates: {predicates_none:?}"); // for debugging

        let (closure_map_none, _) =
            convert_list_of_tuples_to_hashmap(&list_of_tuples, &predicates_none);
        println!("Received closure map: {closure_map_none:?}"); // for debugging

        // when no predicates are specified predicates will be set to _all to cover all relations
        assert!(closure_map_none.contains_key("_all"));
    }

    #[test]
    fn test_predicate_set_to_string() {
        let predicates_is_a: Option<Vec<Predicate>> =
            Some(["is_a"].iter().map(|&s| s.to_string()).collect());
        let predicates_is_a_part_of: Option<Vec<Predicate>> =
            Some(["is_a", "part_of"].iter().map(|&s| s.to_string()).collect());
        let predicates_part_of_is_a: Option<Vec<Predicate>> =
            Some(["part_of", "is_a"].iter().map(|&s| s.to_string()).collect());
        let predicates_empty: Option<Vec<Predicate>> = None;

        assert_eq!(predicate_set_to_key(&predicates_is_a), "+is_a");
        assert_eq!(
            predicate_set_to_key(&predicates_is_a_part_of),
            "+is_a+part_of"
        );
        assert_eq!(
            predicate_set_to_key(&predicates_part_of_is_a),
            "+is_a+part_of"
        );
        assert_eq!(predicate_set_to_key(&predicates_empty), "_all");
    }

    #[test]
    fn test_expand_term_using_closure() {
        let mut closure_table: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::new();
        let mut map: HashMap<PredicateSetKey, HashSet<TermID>> = HashMap::new();
        let mut set: HashSet<TermID> = HashSet::new();
        set.insert(String::from("CARO:0000000"));
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("CARO:0000000"), set);
        closure_table.insert(String::from("+subClassOf"), map.clone());

        let mut set: HashSet<TermID> = HashSet::new();
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("BFO:0000002"), set);
        closure_table.insert(String::from("+subClassOf"), map.clone());

        let mut set: HashSet<TermID> = HashSet::new();
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("BFO:0000003"), set);
        closure_table.insert(String::from("+subClassOf"), map);

        let term = String::from("CARO:0000000");
        let predicates: Option<Vec<Predicate>> = Some(vec!["subClassOf".to_string()]);
        let result_1 = expand_term_using_closure(&term, &closure_table, &predicates);
        let result_2 = expand_term_using_closure(&term, &closure_table, &None);

        let expected_result = HashSet::from([
            "BFO:0000002".to_string(),
            "BFO:0000003".to_string(),
            "CARO:0000000".to_string(),
        ]);
        assert_eq!(result_1, expected_result);
        assert_eq!(result_2, expected_result);
    }

    #[test]
    fn test_rearrange_columns_and_rewrite() {
        // Create a temporary file for testing
        let filename = "tests/data/test_rearrange_data.tsv";
        let mut file = File::create(filename).expect("Failed to create file");
        writeln!(file, "Column A\tColumn B\tColumn C").expect("Failed to write line");
        writeln!(file, "Value 1\tValue 2\tValue 3").expect("Failed to write line");
        writeln!(file, "Value 4\tValue 5\tValue 6").expect("Failed to write line");

        // Define the desired column sequence
        let sequence = vec![
            String::from("Column C"),
            String::from("Column A"),
            String::from("Column B"),
        ];

        // Call the function being tested
        let _ = rearrange_columns_and_rewrite(filename, sequence);

        // Read the modified file and check the contents
        let mut file = File::open(filename).expect("Failed to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Failed to read file");

        println!("{contents:?}");
        assert_eq!(
            contents,
            "Column C\tColumn A\tColumn B\nValue 3\tValue 1\tValue 2\nValue 6\tValue 4\tValue 5\n"
        );

        // Clean up the temporary file
        std::fs::remove_file(filename).expect("Failed to remove file");
    }

    #[test]
    fn test_get_termset_vector() {
        let mut term_label_hashmap = HashMap::new();
        term_label_hashmap.insert("GO:0005575".to_string(), "cellular_component".to_string());
        term_label_hashmap.insert("GO:0099568".to_string(), "cytoplasmic region".to_string());
        term_label_hashmap.insert("GO:0016020".to_string(), "membrane".to_string());

        let terms: HashSet<String> = vec!["GO:0005575".to_string(), "GO:0099568".to_string()]
            .into_iter()
            .collect();

        let result = get_termset_vector(&terms, &term_label_hashmap);

        assert_eq!(result.len(), 2);

        let expected_result: Vec<BTreeInBTree> = vec![
            {
                let mut inner_btreemap = BTreeMap::new();
                inner_btreemap.insert("id".to_string(), "GO:0005575".to_string());
                inner_btreemap.insert("label".to_string(), "cellular_component".to_string());

                let mut outer_btreemap = BTreeMap::new();
                outer_btreemap.insert("GO:0005575".to_string(), inner_btreemap);

                outer_btreemap
            },
            {
                let mut inner_btreemap = BTreeMap::new();
                inner_btreemap.insert("id".to_string(), "GO:0099568".to_string());
                inner_btreemap.insert("label".to_string(), "cytoplasmic region".to_string());

                let mut outer_btreemap = BTreeMap::new();
                outer_btreemap.insert("GO:0099568".to_string(), inner_btreemap);

                outer_btreemap
            },
        ];

        for item in &result {
            assert!(expected_result.contains(item));
        }
    }
}
