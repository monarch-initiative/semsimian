use crate::Predicate;
use crate::PredicateSetKey;
use crate::RustSemsimian;
use crate::TermID;
use crate::{
    utils::expand_term_using_closure, utils::find_embedding_index, utils::predicate_set_to_key,
};
use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet};

pub fn calculate_semantic_jaccard_similarity(
    closure_table: &HashMap<String, HashMap<String, HashSet<String>>>,
    entity1: &str,
    entity2: &str,
    predicates: &Option<Vec<String>>,
) -> f64 {
    /* Returns semantic Jaccard similarity between the two sets. */

    let entity1_closure = expand_term_using_closure(entity1, closure_table, predicates);
    let entity2_closure = expand_term_using_closure(entity2, closure_table, predicates);
    let jaccard = calculate_jaccard_similarity_str(&entity1_closure, &entity2_closure);

    println!("SIM: entity1_closure: {entity1_closure:?}");
    println!("SIM: entity2_closure: {entity2_closure:?}");
    println!("SIM: Jaccard: {jaccard}");

    jaccard
}

pub fn calculate_jaccard_similarity_str(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    /* Returns Jaccard similarity between the two sets. */
    let intersection = set1.intersection(set2).count();
    let union_measure = set1.len() + set2.len() - intersection;

    intersection as f64 / union_measure as f64
}

pub fn calculate_jaccard_similarity(set1: &HashSet<i32>, set2: &HashSet<i32>) -> f64 {
    /* Returns Jaccard similarity between the two sets. */
    let intersection = set1.intersection(set2).count();
    let union_measure = set1.len() + set2.len() - intersection;

    intersection as f64 / union_measure as f64
}

pub fn get_most_recent_common_ancestor_with_score(map: HashMap<String, f64>) -> (String, f64) {
    // Returns Inomration Content (IC) for entities.
    let (curie, max_ic) = map
        .into_iter()
        .max_by_key(|&(_, v)| OrderedFloat::from(v.abs()))
        .unwrap();
    (curie, max_ic)
}

pub fn calculate_term_pairwise_information_content(
    closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    ic_map: &HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    entity1: &HashSet<TermID>,
    entity2: &HashSet<TermID>,
    predicates: &Option<Vec<Predicate>>,
) -> f64 {
    // At each iteration, it calculates the IC score using the calculate_max_information_content function,
    // and if the calculated IC score is greater than the current maximum IC (max_resnik_sim_e1_e2),
    // it updates the maximum IC value. Thus, at the end of the iterations,
    // max_resnik_sim_e1_e2 will contain the highest IC score among all the comparisons,
    // representing the best match between entity1 and entity2.
    let entity1_len = entity1.len() as f64;

    let entity1_to_entity2_sum_resnik_sim = entity1.iter().fold(0.0, |sum, e1_term| {
        let max_ic = entity2.iter().fold(0.0, |max_ic, e2_term| {
            let (_max_ic_ancestors1, ic) = calculate_max_information_content(
                closure_map,
                ic_map,
                e1_term,
                e2_term,
                predicates,
            );
            f64::max(max_ic, ic)
        });

        sum + max_ic
    });

    entity1_to_entity2_sum_resnik_sim / entity1_len
}

pub fn calculate_weighted_term_pairwise_information_content(
    rss: &RustSemsimian,
    entity1: &[(TermID, f64, bool)],
    entity2: &[(TermID, f64, bool)],
) -> f64 {
    let sum_of_weights_entity1: f64 = entity1.iter().map(|(_, weight, _)| weight).sum();

    let entity1_to_entity2_sum_resnik_sim =
        entity1
            .iter()
            .fold(0.0, |sum, (e1_term, e1_weight, e1_negated)| {
                // algorithm for negated phenotypes
                // https://docs.google.com/presentation/d/1KjlkejcJf0h6vq1zD7ebNOvkeHWQN4sVUnIA_SumU_E/edit#slide=id.p
                let max_ic = entity2
                    .iter()
                    .fold(0.0, |max_ic, (e2_term, _, e2_negated)| {
                        let ic: f64 = if *e1_negated {
                            if *e2_negated {
                                // case d - both terms are negated
                                // return (min IC of the two) if the terms are the same or one is a subclass of the other
                                if e1_term == e2_term
                                    || get_ancestors_of_term(
                                        e1_term,
                                        &rss.closure_map,
                                        &rss.predicates,
                                    )
                                    .contains(e2_term)
                                    || get_ancestors_of_term(
                                        e2_term,
                                        &rss.closure_map,
                                        &rss.predicates,
                                    )
                                    .contains(e1_term)
                                {
                                    f64::min(
                                        get_ic_of_term(e1_term, &rss.ic_map, &rss.predicates),
                                        get_ic_of_term(e2_term, &rss.ic_map, &rss.predicates),
                                    )
                                } else {
                                    // otherwise, return 0
                                    0.0
                                }
                            } else {
                                // case c - only term1 is negated
                                // return -IC of term2 if term2 is a subclass of term1 or term2 is the same as term1
                                if e2_term == e1_term
                                    || get_ancestors_of_term(
                                        e2_term,
                                        &rss.closure_map,
                                        &rss.predicates,
                                    )
                                    .contains(e1_term)
                                {
                                    -get_ic_of_term(e2_term, &rss.ic_map, &rss.predicates)
                                } else {
                                    0.0
                                }
                            }
                        } else if *e2_negated {
                            // case b - only term2 is negated
                            // return -IC of term1 if term1 is a subclass of term2 or term1 is the same as term2
                            if e1_term == e2_term
                                || get_ancestors_of_term(
                                    e1_term,
                                    &rss.closure_map,
                                    &rss.predicates,
                                )
                                .contains(e2_term)
                            {
                                -get_ic_of_term(e1_term, &rss.ic_map, &rss.predicates)
                            } else {
                                0.0
                            }
                        } else {
                            // case a - neither term is negated, so standard term similarity
                            // return IC of the most informative common ancestor
                            let (_, ic) = calculate_max_information_content(
                                &rss.closure_map,
                                &rss.ic_map,
                                e1_term,
                                e2_term,
                                &rss.predicates,
                            );
                            ic
                        };

                        if f64::abs(ic) > f64::abs(max_ic) {
                            ic
                        } else {
                            max_ic
                        }
                    });
                sum + (max_ic * e1_weight)
            });

    entity1_to_entity2_sum_resnik_sim / sum_of_weights_entity1
}

pub fn get_ic_of_term(
    entity: &str,
    ic_map: &HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    predicates: &Option<Vec<Predicate>>,
) -> f64 {
    // get IC of a single term
    let predicate_set_key = predicate_set_to_key(predicates);
    let ic: f64 = ic_map
        .get(&predicate_set_key)
        .and_then(|ic_map| ic_map.get(entity))
        .copied()
        .unwrap();
    ic
}

pub fn get_ancestors_of_term(
    entity: &str,
    closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    predicates: &Option<Vec<Predicate>>,
) -> HashSet<TermID> {
    // get IC of a single term
    let predicate_set_key = predicate_set_to_key(predicates);
    let ancestors: HashSet<TermID> = closure_map
        .get(&predicate_set_key)
        .and_then(|closure_map| closure_map.get(entity))
        .cloned()
        .unwrap();
    ancestors
}

pub fn calculate_average_termset_information_content(
    semsimian: &RustSemsimian,
    subject_terms: &HashSet<TermID>,
    object_terms: &HashSet<TermID>,
) -> f64 {
    let subject_to_object_average_resnik_sim: f64 = calculate_term_pairwise_information_content(
        &semsimian.closure_map,
        &semsimian.ic_map,
        subject_terms,
        object_terms,
        &semsimian.predicates,
    );

    let object_to_subject_average_resnik_sim: f64 = calculate_term_pairwise_information_content(
        &semsimian.closure_map,
        &semsimian.ic_map,
        object_terms,
        subject_terms,
        &semsimian.predicates,
    );
    (subject_to_object_average_resnik_sim + object_to_subject_average_resnik_sim) / 2.0
}

pub fn calculate_max_information_content(
    closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    ic_map: &HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    entity1: &str,
    entity2: &str,
    predicates: &Option<Vec<Predicate>>,
) -> (HashSet<String>, f64) {
    // Code to calculate max IC and all ancestors that correspond to the IC.
    // The reason a HashSet<TermID> is returned instead of just TermID is
    // explained through the example used for the test in lib.rs named
    // test_all_by_all_pairwise_similarity_with_nonempty_inputs
    // "apple" has 2 ancestors with the same resnik score (food & item)
    // This during the execution of this test. Each time it runs, it randomly
    // picks on or the other. This is expected in a real-world scenario
    // and hence we return a set of all ancestors with the max resnik score rather than one.
    let filtered_common_ancestors = common_ancestors(closure_map, entity1, entity2, predicates);
    let predicate_set_key = predicate_set_to_key(predicates);

    let (max_ic_ancestors, max_ic) = filtered_common_ancestors.into_iter().fold(
        (HashSet::new(), 0.0),
        |(mut max_ic_ancestors, max_ic), ancestor| {
            if let Some(ic_map) = ic_map.get(&predicate_set_key) {
                if let Some(ic) = ic_map.get(&ancestor) {
                    if *ic > max_ic {
                        max_ic_ancestors.clear();
                        max_ic_ancestors.insert(ancestor);
                        (max_ic_ancestors, *ic)
                    } else if *ic == max_ic {
                        max_ic_ancestors.insert(ancestor);
                        (max_ic_ancestors, max_ic)
                    } else {
                        (max_ic_ancestors, max_ic)
                    }
                } else {
                    (max_ic_ancestors, max_ic)
                }
            } else {
                (max_ic_ancestors, max_ic)
            }
        },
    );

    (max_ic_ancestors, max_ic)
}

/// Returns the common ancestors of two entities based on the given closure table and a set of predicates.

fn common_ancestors(
    closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,

    // {"GO:1234": {'is_a': {'GO:5678', 'GO:9012'}, 'part_of': {'GO:3456', 'GO:7890'}}}

    // {"GO:5678": ('is_a', 'part_of'): {['GO:3456', 'GO:7890']}}\

    // {"GO:5678": vec![('is_a', 'part_of')]: {['GO:3456', 'GO:7890']}}

    // {"GO:5678": 'is_a_+_part_of': {['GO:3456', 'GO:7890']}}
    entity1: &str,
    entity2: &str,
    predicates: &Option<Vec<Predicate>>,
) -> Vec<String> {
    // expand_term_using_closure() handles case of the entity being not present -> returning empty set
    let entity1_closure = expand_term_using_closure(entity1, closure_map, predicates);
    let entity2_closure = expand_term_using_closure(entity2, closure_map, predicates);

    entity1_closure
        .iter()
        .filter(|ancestor| entity2_closure.contains(*ancestor))
        .cloned()
        .collect()
}

pub fn calculate_cosine_similarity_for_nodes(
    embeddings: &[(String, Vec<f64>)],
    term1: &str,
    term2: &str,
) -> Option<f64> {
    match (
        find_embedding_index(embeddings, term1),
        find_embedding_index(embeddings, term2),
    ) {
        (Some(index1), Some(index2)) => {
            let embed_1 = &embeddings[index1].1;
            let embed_2 = &embeddings[index2].1;
            Some(calculate_cosine_similarity_for_embeddings(embed_1, embed_2))
        }
        _ => {
            if find_embedding_index(embeddings, term1).is_none() {
                eprintln!("Embedding for term '{}' not found", term1);
            }
            if find_embedding_index(embeddings, term2).is_none() {
                eprintln!("Embedding for term '{}' not found", term2);
            }
            None
        }
    }
}

fn calculate_cosine_similarity_for_embeddings(embed_1: &[f64], embed_2: &[f64]) -> f64 {
    let (dot_product, norm_embed_1, norm_embed_2) = embed_1.iter().zip(embed_2.iter()).fold(
        (0.0, 0.0, 0.0),
        |(dot_product, norm_embed_1, norm_embed_2), (&a, &b)| {
            (
                dot_product + a * b,
                norm_embed_1 + a * a,
                norm_embed_2 + b * b,
            )
        },
    );

    let norm_embed_1 = norm_embed_1.sqrt();
    let norm_embed_2 = norm_embed_2.sqrt();

    if norm_embed_1 == 0.0 || norm_embed_2 == 0.0 {
        return 0.0;
    }

    dot_product / (norm_embed_1 * norm_embed_2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_constants::*;
    use crate::utils::numericize_sets;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_semantic_jaccard_similarity_new() {
        let sco_predicate: Vec<Predicate> = vec![(String::from("subClassOf"))];

        let result = calculate_semantic_jaccard_similarity(
            &CLOSURE_MAP,
            "CARO:0000000",
            "BFO:0000002",
            &Some(sco_predicate.clone()),
        );

        println!("{result:?}");
        assert_eq!(result, 2.0 / 3.0);

        let result2 = calculate_semantic_jaccard_similarity(
            &CLOSURE_MAP,
            "BFO:0000002",
            "BFO:0000003",
            &Some(sco_predicate.clone()),
        );
        println!("{result2:?}");
        assert_eq!(result2, 1.0 / 3.0);

        let sco_po_predicate: Vec<String> =
            vec![String::from("subClassOf"), String::from("partOf")];

        let result3 = calculate_semantic_jaccard_similarity(
            &CLOSURE_MAP2,
            "BFO:0000002",
            "BFO:0000003",
            &Some(sco_po_predicate.clone()),
        );
        println!("{result3:?}");
        assert_eq!(result3, 1.0 / 3.0);
    }

    #[test]
    fn test_semantic_jaccard_similarity_fruits() {
        let _closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::new();
        let related_to_predicate: Vec<Predicate> = vec![String::from("related_to")];
        // the closure set for "apple" includes both "apple" and "banana", and the closure set for "banana" includes "banana" and "orange". The intersection of these two sets is {"banana"}, and the union is {"apple", "banana", "orange"}, so the Jaccard similarity would be 1 / 3 ≈ 0.33
        let result = calculate_semantic_jaccard_similarity(
            &FRUIT_CLOSURE_MAP,
            "apple",
            "banana",
            &Some(related_to_predicate.clone()),
        );
        println!("{result}");
        assert_eq!(result, 1.0 / 3.0);

        let result2 = calculate_semantic_jaccard_similarity(
            &FRUIT_CLOSURE_MAP,
            "banana",
            "orange",
            &Some(related_to_predicate.clone()),
        );
        println!("{result2}");
        assert_eq!(result2, 1.0 / 3.0);

        // NO predicates (should be the same as above)
        let no_predicate: Option<Vec<Predicate>> = None;
        let result2 = calculate_semantic_jaccard_similarity(
            &ALL_NO_PRED_MAP,
            "banana",
            "orange",
            &no_predicate,
        );
        println!("{result2}");
        assert_eq!(result2, 1.0 / 3.0);
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

    // TODO: test that closure map in Semsimian object is correct
    // TODO: test that ic map in Semsimian object is correct

    #[test]
    fn test_calculate_max_information_content() {
        let _ic_map: HashMap<PredicateSetKey, HashMap<TermID, f64>> = [(
            String::from("+subClassOf"),
            [
                (String::from("CARO:0000000"), 2.585),
                (String::from("BFO:0000002"), 1.585),
                (String::from("BFO:0000003"), 1.0),
            ]
            .iter()
            .cloned()
            .collect(),
        )]
        .iter()
        .cloned()
        .collect();

        // closure map looks like this:
        // {'subClassOf': {'CARO:0000000': {'CARO:0000000', 'BFO:0000002', 'BFO:0000003'},
        //                 'BFO:0000002':  {'BFO:0000002', 'BFO:0000003'},
        //                 'BFO:0000003':  {'BFO:0000003'}}}

        let mut closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::new();

        let mut map: HashMap<PredicateSetKey, HashSet<TermID>> = HashMap::new();
        let mut set: HashSet<TermID> = HashSet::new();
        set.insert(String::from("CARO:0000000"));
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("CARO:0000000"), set.clone());
        closure_map.insert(String::from("+subClassOf"), map.clone());

        set.clear();
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("BFO:0000002"), set.clone());
        closure_map.insert(String::from("+subClassOf"), map.clone());

        set.clear();
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("BFO:0000003"), set.clone());
        closure_map.insert(String::from("+subClassOf"), map);

        // Term frequencies:

        // "CARO:0000000": 1
        // "BFO:0000002": 2
        // "BFO:0000003": 2
        // "BFO:0000004": 1
        // The corpus size (sum of term frequencies) would be 6.

        // Using these term frequencies, the IC scores can be calculated as follows:

        // IC("CARO:0000000") = -log2(1/6) ≈ 2.585
        // IC("BFO:0000002") = -log2(2/6) ≈ 1.585
        // IC("BFO:0000003") = -log2(2/6) ≈ 1.585
        // IC("BFO:0000004") = -log2(1/6) ≈ 2.585
        //
        // Max IC for "CARO:0000000" and "BFO:0000002":
        // Common ancestors: "BFO:0000002" and "BFO:0000003"
        // Max IC: 1.585 (IC of "BFO:0000002")

        let predicates = Some(vec![String::from("subClassOf")]);
        let (_, result) = calculate_max_information_content(
            &CLOSURE_MAP,
            &IC_MAP,
            &String::from("CARO:0000000"),
            &String::from("BFO:0000002"),
            &predicates,
        );

        println!("Max IC: {result:?}");
        let expected_value = 1.585;
        assert!(
            (result - expected_value).abs() < 1e-3,
            "Expected value: {expected_value}, got: {result:?}"
        );
    }

    #[test]
    fn test_calculate_cosine_similarity_for_nodes() {
        // Test case 1: Non-empty embeddings, terms exist
        let embeddings = vec![
            ("term1".to_string(), vec![1.0, -2.0, 3.0]),
            ("term2".to_string(), vec![-4.0, 5.0, -6.0]),
        ];
        let term1 = "term1";
        let term2 = "term2";
        let expected_result = -0.9746318461970762;
        assert_eq!(
            calculate_cosine_similarity_for_nodes(&embeddings, term1, term2).unwrap(),
            expected_result
        );

        // Test case 2: Non-empty embeddings, one term doesn't exist
        let embeddings = vec![
            ("term1".to_string(), vec![1.0, -2.0, 3.0]),
            ("term2".to_string(), vec![-4.0, 5.0, -6.0]),
        ];
        let term1 = "term1";
        let term2 = "term3"; // Term3 doesn't exist in embeddings
        assert!(calculate_cosine_similarity_for_nodes(&embeddings, term1, term2).is_none());

        // Test case 3: Empty embeddings
        let embeddings: Vec<(String, Vec<f64>)> = vec![];
        let term1 = "term1";
        let term2 = "term2";
        assert!(calculate_cosine_similarity_for_nodes(&embeddings, term1, term2).is_none());
    }

    #[test]
    fn test_calculate_cosine_similarity_for_embeddings() {
        // Test case 1: Non-zero similarity
        let embed_1 = vec![1.0, -2.0, 3.0];
        let embed_2 = vec![-4.0, 5.0, -6.0];
        let similarity = calculate_cosine_similarity_for_embeddings(&embed_1, &embed_2);
        assert_eq!(similarity, -0.9746318461970762);

        // Test case 2: Zero similarity
        let embed_1 = vec![1.0, 0.0, 0.0];
        let embed_2 = vec![0.0, 1.0, 0.0];
        let similarity = calculate_cosine_similarity_for_embeddings(&embed_1, &embed_2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_calculate_term_pairwise_information_content() {
        let predicates: Option<Vec<Predicate>> = Some(vec![Predicate::from("subClassOf")]);

        // Test case 1: Normal case, entities have terms.
        let entity1: HashSet<TermID> = vec!["CARO:0000000", "BFO:0000002"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let entity2: HashSet<TermID> = vec!["BFO:0000003", "BFO:0000004"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let resnik_score = calculate_term_pairwise_information_content(
            &CLOSURE_MAP,
            &IC_MAP,
            &entity1,
            &entity2,
            &predicates,
        );
        let expected_value = 1.0;

        println!("Case 1 resnik_score: {resnik_score}");
        assert!((resnik_score - expected_value).abs() < f64::EPSILON);

        // Test case 2: Normal case, entities have terms.
        let entity1: HashSet<TermID> = vec!["CARO:0000000"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let entity2: HashSet<TermID> = vec!["BFO:0000002"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let resnik_score = calculate_term_pairwise_information_content(
            &CLOSURE_MAP,
            &IC_MAP,
            &entity1,
            &entity2,
            &predicates,
        );
        let expected_value = 1.585;

        println!("Case 2 resnik_score: {resnik_score}");
        assert!((resnik_score - expected_value).abs() < f64::EPSILON);

        // Test case 3: Normal case, entities have terms.
        let entity1: HashSet<TermID> = vec!["BFO:0000002", "BFO:0000004", "BFO:0000003"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let entity2: HashSet<TermID> = vec!["BFO:0000003", "BFO:0000004"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let resnik_score = calculate_term_pairwise_information_content(
            &CLOSURE_MAP,
            &IC_MAP,
            &entity1,
            &entity2,
            &predicates,
        );
        let expected_value = 0.6666666666666666;

        println!("Case 3 resnik_score: {resnik_score}");
        assert!((resnik_score - expected_value).abs() < f64::EPSILON);

        // Test case 4: Normal case, entities have terms.
        let entity1: HashSet<TermID> = vec!["CARO:0000000", "BFO:0000002", "BFO:0000004"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let entity2: HashSet<TermID> = vec!["BFO:0000002", "BFO:0000004"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let resnik_score = calculate_term_pairwise_information_content(
            &CLOSURE_MAP,
            &IC_MAP,
            &entity1,
            &entity2,
            &predicates,
        );
        let expected_value = 1.0566666666666666;

        println!("Case 4 resnik_score: {resnik_score}");
        assert!((resnik_score - expected_value).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_average_termset_information_content() {
        let predicates: Option<Vec<Predicate>> = Some(vec![
            Predicate::from("rdfs:subClassOf"),
            Predicate::from("BFO:0000050"),
        ]);
        let db = Some("tests/data/go-nucleus.db");
        let mut rss = RustSemsimian::new(None, predicates, None, db);

        rss.update_closure_and_ic_map();

        // Test case 1: Normal case, entities have terms.
        let entity1: HashSet<TermID> = HashSet::from(["GO:0005634".to_string()]);

        let entity2: HashSet<TermID> = HashSet::from(["GO:0031965".to_string()]);

        let phenio_score = calculate_average_termset_information_content(&rss, &entity1, &entity2);
        let expected_value = 5.8496657269155685;

        println!("Case X pheno_score: {phenio_score}");
        assert_eq!(phenio_score, expected_value);

        // Test case 2: Normal case, entities have terms.
        let entity1: HashSet<TermID> =
            HashSet::from(["GO:0005634".to_string(), "GO:0016020".to_string()]);

        let entity2: HashSet<TermID> =
            HashSet::from(["GO:0031965".to_string(), "GO:0005773".to_string()]);

        let phenio_score = calculate_average_termset_information_content(&rss, &entity1, &entity2);
        let expected_value = 5.4154243283740175;

        println!("Case 3 pheno_score: {phenio_score}");
        assert_eq!(phenio_score, expected_value);
    }
}
