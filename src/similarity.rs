use crate::utils::expand_term_using_closure;
use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet};

pub fn calculate_semantic_jaccard_similarity(
    closure_table: &HashMap<String, HashMap<String, HashSet<String>>>,
    entity1: String,
    entity2: String,
    predicates: &Option<HashSet<String>>,
) -> f64 {
    /* Returns semantic Jaccard similarity between the two sets. */
    let entity1_closure = expand_term_using_closure(&entity1, &closure_table, &predicates);
    let entity2_closure = expand_term_using_closure(&entity2, &closure_table, &predicates);
    let jaccard = calculate_jaccard_similarity_str(&entity1_closure, &entity2_closure);
    jaccard
}

pub fn calculate_jaccard_similarity_str(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    /* Returns Jaccard similarity between the two sets. */
    let intersection = set1.intersection(&set2).count();
    let union_measure = set1.union(&set2).count();
    let jaccard = intersection as f64 / union_measure as f64;
    jaccard
}

pub fn calculate_jaccard_similarity(set1: &HashSet<i32>, set2: &HashSet<i32>) -> f64 {
    /* Returns Jaccard similarity between the two sets. */
    let intersection = set1.intersection(&set2).count();
    let union_measure = set1.union(&set2).count();
    let jaccard = intersection as f64 / union_measure as f64;
    jaccard
}

pub fn get_most_recent_common_ancestor_with_score(map: HashMap<String, f64>) -> (String, f64) {
    // Returns Inomration Content (IC) for entities.
    let (curie, max_ic) = map
        .into_iter()
        .max_by_key(|&(_, v)| OrderedFloat(v.abs()))
        .unwrap();
    (curie, max_ic)
}

pub fn calculate_phenomizer_score(
    map: HashMap<String, HashMap<String, f64>>,
    entity1: HashSet<String>,
    entity2: HashSet<String>,
) -> f64 {
    // calculate average resnik sim of all terms in entity1 and their best match in entity2
    let entity1_to_entity2_average_resnik_sim: f64 =
        pairwise_entity_resnik_score(&map, &entity1, &entity2);
    // now do the same for entity2 to entity1
    let entity2_to_entity1_average_resnik_sim: f64 =
        pairwise_entity_resnik_score(&map, &entity2, &entity1);
    // return the average of the two
    return (entity1_to_entity2_average_resnik_sim + entity2_to_entity1_average_resnik_sim) / 2.0;
}

pub fn pairwise_entity_resnik_score(
    map: &HashMap<String, HashMap<String, f64>>,
    entity1: &HashSet<String>,
    entity2: &HashSet<String>,
) -> f64 {
    let mut entity1_to_entity2_sum_resnik_sim = 0.0;

    for e1_term in entity1.clone().into_iter() {
        let mut max_resnik_sim_e1_e2 = 0.0;
        for e2_term in entity2.clone().into_iter() {
            // NB: this will definitely fail if the term is not in the map
            let mica = map.get(&e1_term).unwrap().get(&e2_term).unwrap();
            if mica > &max_resnik_sim_e1_e2 {
                max_resnik_sim_e1_e2 = *mica;
            }
        }
        entity1_to_entity2_sum_resnik_sim += max_resnik_sim_e1_e2;
    }
    let entity1_to_entity2_average_resnik_sim =
        entity1_to_entity2_sum_resnik_sim / entity1.len() as f64;
    return entity1_to_entity2_average_resnik_sim;
}

pub fn calculate_max_information_content(
    closure_table: &HashMap<String, HashMap<String, HashSet<String>>>,
    entity1: &String,
    entity2: &String,
    predicates: &Option<HashSet<String>>,
) -> f64 {
    // CODE TO CALCULATE MAX IC
    let owl_thing = "owl:Thing".to_string();
    let filtered_common_ancestors: Vec<String> = common_ancestors(&closure_table, &entity1, &entity2, &predicates)
        .into_iter()
        .filter(|ancestor| *ancestor != owl_thing) //removes owl:Thing from common ancestor, leaving only common ancestors
        .collect();

    let information_content_scores = calculate_information_content_scores(
        &filtered_common_ancestors,
        &closure_table,
        predicates
    );

    let (_ancestor, max_ic) = mrca_and_score(&information_content_scores);
    max_ic
}

fn common_ancestors(
    closure_table: &HashMap<String, HashMap<String, HashSet<String>>>,
    entity1: &String,
    entity2: &String,
    predicates: &Option<HashSet<String>>,
) -> Vec<String> {

    // TODO: if predicates is empty, need to use ALL predicates

    // if predicates.is_none(){
    //     // if predicates is None, then we need to use ALL predicates
    // }

    if let (Some(e1_ancestors),
        Some(e2_ancestors)) = (entity1_ancestors, entity2_ancestors) {

        // the code below that uses filter_ancestors_by_predicates() should probably be using expand_term_using_closure()
        // let entity1_closure = expand_term_using_closure(&entity1, &closure_table, &predicates);
        // let entity2_closure = expand_term_using_closure(&entity2, &closure_table, &predicates);
        let filtered_e1_ancestors = filter_ancestors_by_predicates(&e1_ancestors, &predicates);
        let filtered_e2_ancestors = filter_ancestors_by_predicates(&e2_ancestors, &predicates);

        filtered_e1_ancestors
            .into_iter()
            .filter(|ancestor| filtered_e2_ancestors.contains(ancestor))
            .collect()
    } else {
        vec![]
    }
}

fn filter_ancestors_by_predicates(
    ancestors: &HashMap<String, HashSet<String>>,
    predicates: &Option<HashSet<String>>,
) -> HashSet<String> {
    match predicates {
        Some(preds) => {
            let mut filtered = HashSet::new();
            for (predicate, entities) in ancestors {
                if preds.contains(predicate) {
                    for entity in entities {
                        filtered.insert(entity.clone());
                    }
                }
            }
            filtered
        }
        None => ancestors
            .values()
            .flat_map(|entities| entities.iter().cloned())
            .collect(),
    }
}

// scores: maps ancestors to corresponding IC scores
fn mrca_and_score(scores: &HashMap<String, f64>) -> (Option<String>, f64) {
    let mut max_ic = 0.0;
    let mut mrca = None;

    for (ancestor, ic) in scores.iter() {
        if *ic > max_ic {
            max_ic = *ic;
            mrca = Some(ancestor.clone());
        }
    }
    (mrca, max_ic)
}

fn calculate_information_content_scores(
    filtered_common_ancestors: &Vec<String>,
    closure_table: &HashMap<String, HashMap<String, HashSet<String>>>,
    predicates: &Option<HashSet<String>>,
) -> HashMap<String, f64> {
    let (term_frequencies, corpus_size) =
        calculate_term_frequencies_and_corpus_size(closure_table, predicates);

    let mut ic_scores = HashMap::new();
    for ancestor in filtered_common_ancestors {
        if let Some(freq) = term_frequencies.get(ancestor) {
            let probability = *freq as f64 / corpus_size as f64;
            let ic = -probability.log2();
            ic_scores.insert(ancestor.clone(), ic);
        }
    }
    ic_scores
}

fn calculate_term_frequencies_and_corpus_size(
    closure_table: &HashMap<String, HashMap<String, HashSet<String>>>,
    predicates: &Option<HashSet<String>>,
) -> (HashMap<String, usize>, usize) {
    let mut term_frequencies = HashMap::new();
    let mut corpus_size = 0;

    for (_entity, predicate_map) in closure_table {
        for (predicate, terms) in predicate_map {
            if predicates.is_none() || predicates.as_ref().unwrap().contains(predicate) {
                for term in terms {
                    *term_frequencies.entry(term.clone()).or_insert(0) += 1;
                }
                corpus_size += terms.len();
            }
        }
    }

    (term_frequencies, corpus_size)
}

#[cfg(test)]
mod tests {
    use crate::utils::numericize_sets;

    use super::*;

    #[test]
    fn test_semantic_jaccard_similarity() {
        let mut closure_table: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();

        // closure table looks like this:
        // CARO:0000000 -> subClassOf -> CARO:0000000, BFO:0000002, BFO:0000003

        // BFO:0000002 -> subClassOf -> BFO:0000002, BFO:0000003

        // BFO:0000003 -> subClassOf -> BFO:0000003
        //             -> partOf -> BFO:0000004

        let mut map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("CARO:0000000"));
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);
        closure_table.insert(String::from("CARO:0000000"), map);
        let mut map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);
        closure_table.insert(String::from("BFO:0000002"), map);
        let mut map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);

        let mut set2: HashSet<String> = HashSet::new();
        set2.insert(String::from("BFO:0000004"));
        map.insert(String::from("partOf"), set2);

        closure_table.insert(String::from("BFO:0000003"), map);
        let mut sco_predicate: HashSet<String> = HashSet::new();
        sco_predicate.insert(String::from("subClassOf"));
        let result = calculate_semantic_jaccard_similarity(
            &closure_table,
            String::from("CARO:0000000"),
            String::from("BFO:0000002"),
            &Some(sco_predicate.clone()),
        );
        println!("{result}");
        assert_eq!(result, 2.0 / 3.0);

        let result2 = calculate_semantic_jaccard_similarity(
            &closure_table,
            String::from("BFO:0000002"),
            String::from("BFO:0000003"),
            &Some(sco_predicate.clone()),
        );
        println!("{result2}");
        assert_eq!(result2, 0.5);

        let mut sco_po_predicate: HashSet<String> = HashSet::new();
        sco_po_predicate.insert(String::from("subClassOf"));
        sco_po_predicate.insert(String::from("partOf"));
        // println!("{closure_table:?}");
        let result3 = calculate_semantic_jaccard_similarity(
            &closure_table,
            String::from("BFO:0000002"),
            String::from("BFO:0000003"),
            &Some(sco_po_predicate.clone()),
        );
        println!("{result3}");
        assert_eq!(result3, 1.0 / 3.0);
    }

    #[test]
    fn test_calculate_jaccard_similarity() {
        let set1: HashSet<String> = HashSet::from([String::from("apple"), String::from("banana")]);
        let set2: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("fruit"),
            String::from("tropical"),
        ]);
        let (num_set1, num_set2, _) = numericize_sets(&set1, &set2);
        let result = calculate_jaccard_similarity(&num_set1, &num_set2);
        println!("{result}");
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_calculate_jaccard_similarity_str() {
        let set1: HashSet<String> = HashSet::from([String::from("apple"), String::from("banana")]);
        let set2: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("fruit"),
            String::from("tropical"),
        ]);
        let result = calculate_jaccard_similarity_str(&set1, &set2);
        println!("{result}");
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_get_most_recent_common_ancestor_with_score() {
        let map: HashMap<String, f64> = HashMap::from([
            (String::from("CARO:0000000"), 21.05),
            (String::from("BFO:0000002"), 0.7069),
            (String::from("BFO:0000003"), 14.89),
        ]);
        let expected_tuple: (String, f64) = (String::from("CARO:0000000"), 21.05);

        let result = get_most_recent_common_ancestor_with_score(map);
        assert_eq!(result, expected_tuple);
    }

    #[test]
    fn test_calculate_phenomizer_score() {
        let map: HashMap<String, HashMap<String, f64>> = HashMap::from([
            (
                String::from("CARO:0000000"),
                HashMap::from([
                    (String::from("CARO:0000000"), 5.0),
                    (String::from("BFO:0000002"), 4.0),
                    (String::from("BFO:0000003"), 3.0),
                ]),
            ),
            (
                String::from("BFO:0000002"),
                HashMap::from([
                    (String::from("CARO:0000000"), 2.0),
                    (String::from("BFO:0000002"), 4.0),
                    (String::from("BFO:0000003"), 3.0),
                ]),
            ),
            (
                String::from("BFO:0000003"),
                HashMap::from([
                    (String::from("CARO:0000000"), 1.0),
                    (String::from("BFO:0000002"), 3.0),
                    (String::from("BFO:0000003"), 4.0),
                ]),
            ),
        ]);

        let mut entity_one = HashSet::new();
        entity_one.insert(String::from("CARO:0000000")); // resnik of best match = 5
        entity_one.insert(String::from("BFO:0000002")); // resnik of best match = 4

        let mut entity_two = HashSet::new();
        entity_two.insert(String::from("BFO:0000003")); // resnik of best match = 3
        entity_two.insert(String::from("BFO:0000002")); // resnik of best match = 4
        entity_two.insert(String::from("CARO:0000000")); // resnik of best match = 5

        let expected = ((5.0 + 4.0) / 2.0 + (3.0 + 4.0 + 5.0) / 3.0) / 2.0;

        let result = calculate_phenomizer_score(map, entity_one, entity_two);
        assert_eq!(result, expected);
    }

    fn test_calculate_max_information_content() {
        let mut closure_table: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();

        // closure table looks like this:
        // CARO:0000000 -> subClassOf -> CARO:0000000, BFO:0000002, BFO:0000003
        // BFO:0000002 -> subClassOf -> BFO:0000002, BFO:0000003
        // BFO:0000003 -> subClassOf -> BFO:0000003

        let mut map: HashMap<String, HashSet<String>> = HashMap::new();
        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("CARO:0000000"));
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);
        closure_table.insert(String::from("CARO:0000000"), map.clone());

        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);
        closure_table.insert(String::from("BFO:0000002"), map.clone());

        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("subClassOf"), set);
        closure_table.insert(String::from("BFO:0000003"), map);

        // Term frequencies:
        // "CARO:0000000": 1
        // "BFO:0000002": 2
        // "BFO:0000003": 3
        //
        // Corpus size: 6 (sum of term frequencies)
        //
        // Information Content (IC) scores:
        // IC("CARO:0000000") = -log2(1/6) ≈ 2.585
        // IC("BFO:0000002") = -log2(2/6) ≈ 1.585
        // IC("BFO:0000003") = -log2(3/6) ≈ 1
        //
        // Max IC for "CARO:0000000" and "BFO:0000002":
        // Common ancestors: "BFO:0000002" and "BFO:0000003"
        // Max IC: 1.585 (IC of "BFO:0000002")


        let predicates = Some(HashSet::from([String::from("subClassOf")]));
        let result = calculate_max_information_content(
            &closure_table,
            &String::from("CARO:0000000"),
            &String::from("BFO:0000002"),
            &predicates,
        );
        println!("Max IC: {}", result);
        let expected_value = 1.585;
        assert!((result - expected_value).abs() < 1e-3, "Expected value: {}, got: {}", expected_value, result);
    }
}



