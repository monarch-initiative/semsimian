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

    // println!("SIM: entity1_closure: {entity1_closure:?}");
    // println!("SIM: entity2_closure: {entity2_closure:?}");
    // println!("SIM: Jaccard: {jaccard}");

    calculate_jaccard_similarity_str(&entity1_closure, &entity2_closure)
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

pub fn calculate_average_of_max_information_content(
    rss: &RustSemsimian,
    entity1: &HashSet<TermID>,
    entity2: &HashSet<TermID>,
) -> f64 {
    // At each iteration, it calculates the IC score using the calculate_max_information_content function,
    // and if the calculated IC score is greater than the current maximum IC (max_resnik_sim_e1_e2),
    // it updates the maximum IC value. Thus, at the end of the iterations,
    // max_resnik_sim_e1_e2 will contain the highest IC score among all the comparisons,
    // representing the best match between entity1 and entity2.
    let entity1_len = entity1.len() as f64;

    let entity1_to_entity2_sum_resnik_sim = entity1.iter().fold(0.0, |sum, e1_term| {
        let max_ic = entity2.iter().fold(0.0, |max_ic, e2_term| {
            let (_max_ic_ancestors1, ic) =
                calculate_max_information_content(rss, e1_term, e2_term, &rss.predicates);
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
                                || get_ancestors_of_term(e1_term, &rss.closure_map, &rss.predicates)
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
                                rss,
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

pub fn calculate_average_of_max_jaccard_similarity(
    rss: &RustSemsimian,
    entity1: &HashSet<TermID>,
    entity2: &HashSet<TermID>,
) -> f64 {
    // At each iteration, it calculates the Jaccard similarity using the calculate_jaccard_similarity function,
    // and if the calculated Jaccard similarity is greater than the current maximum Jaccard similarity (max_jaccard_sim_e1_e2),
    // it updates the maximum Jaccard value. Thus, at the end of the iterations,
    // max_jaccard_sim_e1_e2 will contain the highest Jaccard similarity among all the comparisons,
    // representing the best match between entity1 and entity2.
    let entity1_len = entity1.len() as f64;

    let entity1_to_entity2_sum_jaccard_similarity = entity1.iter().fold(0.0, |sum, e1_term| {
        let max_jaccard_similarity = entity2.iter().fold(0.0, |max_jaccard, e2_term| {
            let score = calculate_semantic_jaccard_similarity(
                &rss.closure_map,
                e1_term,
                e2_term,
                &rss.predicates,
            );
            f64::max(max_jaccard, score)
        });

        sum + max_jaccard_similarity
    });

    entity1_to_entity2_sum_jaccard_similarity / entity1_len
}

pub fn calculate_average_of_max_phenodigm_score(
    rss: &RustSemsimian,
    entity1: &HashSet<TermID>,
    entity2: &HashSet<TermID>,
) -> f64 {
    // At each iteration, it calculates the phenodigm score using the phenodigm_score function,
    // and if the calculated phenodigm score is greater than the current maximum phenodigm score (max_pheno_e1_e2),
    // it updates the maximum phenodigm value. Thus, at the end of the iterations,
    // max_pheno_e1_e2 will contain the highest phenodigm score among all the comparisons,
    // representing the best match between entity1 and entity2.
    let entity1_len = entity1.len() as f64;

    let entity1_to_entity2_sum_phenodigm_score = entity1.iter().fold(0.0, |sum, e1_term| {
        let max_phenodigm_score = entity2.iter().fold(0.0, |max_pheno, e2_term| {
            let score = rss.phenodigm_score(e1_term, e2_term);
            f64::max(max_pheno, score)
        });

        sum + max_phenodigm_score
    });

    entity1_to_entity2_sum_phenodigm_score / entity1_len
}

pub fn calculate_average_termset_jaccard_similarity(
    rss: &RustSemsimian,
    subject_terms: &HashSet<TermID>,
    object_terms: &HashSet<TermID>,
) -> f64 {
    let subject_to_object_jaccard_similarity: f64 =
        calculate_average_of_max_jaccard_similarity(rss, subject_terms, object_terms);
    let object_to_subject_jaccard_similarity: f64 =
        calculate_average_of_max_jaccard_similarity(rss, object_terms, subject_terms);
    (subject_to_object_jaccard_similarity + object_to_subject_jaccard_similarity) / 2.0
}

pub fn calculate_average_termset_phenodigm_score(
    semsimian: &RustSemsimian,
    subject_terms: &HashSet<TermID>,
    object_terms: &HashSet<TermID>,
) -> f64 {
    let subject_to_object_average_of_max_phenodigm_score: f64 =
        calculate_average_of_max_phenodigm_score(semsimian, subject_terms, object_terms);

    let object_to_subject_average_of_max_phenodigm_score: f64 =
        calculate_average_of_max_phenodigm_score(semsimian, object_terms, subject_terms);

    (subject_to_object_average_of_max_phenodigm_score
        + object_to_subject_average_of_max_phenodigm_score)
        / 2.0
}

pub fn calculate_average_termset_information_content(
    semsimian: &RustSemsimian,
    subject_terms: &HashSet<TermID>,
    object_terms: &HashSet<TermID>,
) -> f64 {
    let subject_to_object_average_of_max_resnik_sim: f64 =
        calculate_average_of_max_information_content(semsimian, subject_terms, object_terms);

    let object_to_subject_average_of_max_resnik_sim: f64 =
        calculate_average_of_max_information_content(semsimian, object_terms, subject_terms);
    (subject_to_object_average_of_max_resnik_sim + object_to_subject_average_of_max_resnik_sim)
        / 2.0
}

pub fn calculate_max_information_content(
    // closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    // ic_map: &HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    rss: &RustSemsimian,
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
    // picks one or the other. This is expected in a real-world scenario
    // and hence we return a set of all ancestors with the max resnik score rather than one.
    let filtered_common_ancestors =
        common_ancestors(&rss.closure_map, entity1, entity2, predicates);
    let predicate_set_key = predicate_set_to_key(predicates);
    let max_ic_key = format!("{entity1}_{entity2}");

    // Check if max_ic_key exists in rss.max_ic_cache
    if let Some((max_ic_ancestors, max_ic)) = rss.max_ic_cache.get(&max_ic_key) {
        return (max_ic_ancestors.clone(), *max_ic);
    }

    // Get ic_map once before the loop
    let ic_map = match rss.ic_map.get(&predicate_set_key) {
        Some(ic_map) => ic_map,
        None => return (HashSet::new(), 0.0), // If there's no ic_map, return early
    };

    let (max_ic_ancestors, max_ic) = filtered_common_ancestors.into_iter().fold(
        (HashSet::new(), 0.0),
        |(mut max_ic_ancestors, max_ic), ancestor| {
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
    use crate::utils::numericize_sets;
    use crate::{enums::MetricEnum, test_constants::constants_for_tests::*};
    use std::collections::{HashMap, HashSet};
    use rstest::rstest;

    #[rstest]
    #[case(
        &CLOSURE_MAP,
        "CARO:0000000",
        "BFO:0000002",
        vec!["subClassOf"],
        2.0 / 3.0
    )]
    #[case(
        &CLOSURE_MAP,
        "BFO:0000002",
        "BFO:0000003",
        vec!["subClassOf"],
        1.0 / 3.0
    )]
    #[case(
        &CLOSURE_MAP2,
        "BFO:0000002",
        "BFO:0000003",
        vec!["subClassOf", "partOf"],
        1.0 / 3.0
    )]
    fn test_semantic_jaccard_similarity_new(
        #[case] closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
        #[case] term1: &str,
        #[case] term2: &str,
        #[case] predicates: Vec<&str>,
        #[case] expected_result: f64,
    ) {
        let predicate_vec: Vec<Predicate> = predicates.into_iter().map(Predicate::from).collect();

        let result = calculate_semantic_jaccard_similarity(
            closure_map,
            term1,
            term2,
            &Some(predicate_vec),
        );

        println!("{result:?}");
        assert_eq!(result, expected_result);
    }

    #[rstest]
    #[case(
        &FRUIT_CLOSURE_MAP,
        "apple",
        "banana",
        Some(vec!["related_to"]),
        1.0 / 3.0
    )]
    #[case(
        &FRUIT_CLOSURE_MAP,
        "banana",
        "orange",
        Some(vec!["related_to"]),
        1.0 / 3.0
    )]
    #[case(
        &ALL_NO_PRED_MAP,
        "banana",
        "orange",
        None,
        1.0 / 3.0
    )]
    fn test_semantic_jaccard_similarity_fruits(
        #[case] closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
        #[case] term1: &str,
        #[case] term2: &str,
        #[case] predicates_option: Option<Vec<&str>>,
        #[case] expected_result: f64,
    ) {
        let predicate_vec_option = predicates_option.map(|preds| preds.into_iter().map(Predicate::from).collect());

        let result = calculate_semantic_jaccard_similarity(
            closure_map,
            term1,
            term2,
            &predicate_vec_option,
        );

        println!("{result}");
        assert_eq!(result, expected_result);
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
        let predicates: Option<Vec<Predicate>> = Some(vec![Predicate::from("rdfs:subClassOf")]);
        let mut rss = RustSemsimian::new(Some(BFO_SPO.clone()), predicates.clone(), None, None);
        rss.update_closure_and_ic_map();
        let (_, result) = calculate_max_information_content(
            &rss,
            &String::from("BFO:0000003"),
            &String::from("BFO:0000035"),
            &predicates,
        );
        dbg!(&rss.ic_map);
        println!("Max IC: {result:?}");
        let expected_value = 1.9593580155026542;
        assert!(
            (result - expected_value).abs() < 1e-3,
            "Expected value: {expected_value}, got: {result:?}"
        );
    }

    #[rstest]
    #[case(
        vec![
            ("term1".to_string(), vec![1.0, -2.0, 3.0]),
            ("term2".to_string(), vec![-4.0, 5.0, -6.0]),
        ],
        "term1",
        "term2",
        Some(-0.9746318461970762)
    )]
    #[case(
        vec![
            ("term1".to_string(), vec![1.0, -2.0, 3.0]),
            ("term2".to_string(), vec![-4.0, 5.0, -6.0]),
        ],
        "term1",
        "term3", // Term3 doesn't exist in embeddings
        None
    )]
    #[case(
        vec![],
        "term1",
        "term2",
        None
    )]
    fn test_calculate_cosine_similarity_for_nodes(
        #[case] embeddings: Vec<(String, Vec<f64>)>,
        #[case] term1: &str,
        #[case] term2: &str,
        #[case] expected_result: Option<f64>
    ) {
        match expected_result {
            Some(expected_value) => {
                assert_eq!(
                    calculate_cosine_similarity_for_nodes(&embeddings, term1, term2).unwrap(),
                    expected_value
                );
            },
            None => {
                assert!(calculate_cosine_similarity_for_nodes(&embeddings, term1, term2).is_none());
            }
        }
    }

    #[rstest]
    #[case(vec![1.0, -2.0, 3.0], vec![-4.0, 5.0, -6.0], -0.9746318461970762)]
    #[case(vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], 0.0)]
    fn test_calculate_cosine_similarity_for_embeddings(
        #[case] embed_1: Vec<f64>,
        #[case] embed_2: Vec<f64>,
        #[case] expected_similarity: f64,
    ) {
        let similarity = calculate_cosine_similarity_for_embeddings(&embed_1, &embed_2);
        assert_eq!(similarity, expected_similarity);
    }

    type TermID = String; // Define TermID type if not already defined

    #[rstest]
    #[case(
        vec!["CARO:0000000", "BFO:0000002"],
        vec!["BFO:0000003", "BFO:0000004"],
        0.24271341358512086
    )]
    #[case(
        vec!["BFO:0000003"],
        vec!["BFO:0000035"],
        1.9593580155026542
    )]
    #[case(
        vec!["BFO:0000002", "BFO:0000004", "BFO:0000003"],
        vec!["BFO:0000003", "BFO:0000004"],
        1.191355953205954
    )]
    #[case(
        vec!["CARO:0000000", "BFO:0000002", "BFO:0000004"],
        vec!["BFO:0000003", "BFO:0000004"],
        0.5382366147050694
    )]
    fn test_calculate_avg_of_max_information_content(
        #[case] entity1_terms: Vec<&str>,
        #[case] entity2_terms: Vec<&str>,
        #[case] expected_value: f64,
    ) {
        let predicates: Option<Vec<Predicate>> = Some(vec![Predicate::from("rdfs:subClassOf")]);
        let mut rss = RustSemsimian::new(Some(BFO_SPO.clone()), predicates.clone(), None, None);
        rss.update_closure_and_ic_map();

        let entity1: HashSet<TermID> = entity1_terms.into_iter().map(|s| s.to_string()).collect();
        let entity2: HashSet<TermID> = entity2_terms.into_iter().map(|s| s.to_string()).collect();

        let resnik_score = calculate_average_of_max_information_content(&rss, &entity1, &entity2);

        println!("Resnik score: {resnik_score}");
        assert!((resnik_score - expected_value).abs() < f64::EPSILON);
    }

    // These comments are the manual calculations for the test cases below, for future reference

    // These are the values that are being used in the manual calculations below:
    // Ontology Term Pair	    Max IC	            Jaccard Similarity
    // BFO:0000002_BFO:0000003	0	                0.3333333333333333
    // CARO:0000000_BFO:0000004	0	                0
    // BFO:0000002_BFO:0000004	0.48542682717024171	0.6666666666666666
    // CARO:0000000_BFO:0000003	0	                0
    // BFO:0000003_BFO:0000035	1.9593580155026542	0.6666666666666666
    // BFO:0000004_BFO:0000004	1.1292830169449666	1
    // CARO:0000000_BFO:0000002	0	                0
    // BFO:0000002_BFO:0000002	0.48542682717024171	1
    // BFO:0000004_BFO:0000002	0.48542682717024171	0.6666666666666666
    // CARO:0000000_BFO:0000001	0	                0
    // BFO:0000002_BFO:0000001	0	                0.5
    // BFO:0000004_BFO:0000001	0	                0.3333333333333333
    // BFO:0000004_BFO:0000003	0	                0.25
    // BFO:0000003_BFO:0000003	1.9593580155026542	1

    // Case 1:
    // Entity1: ["CARO:0000000", "BFO:0000002"]
    // Entity2: ["BFO:0000003", "BFO:0000004"]
    // For CARO:0000000 to Entity2:
    //   To BFO:0000003: phenodigm = sqrt(0 * 0) = 0
    //   To BFO:0000004: phenodigm = sqrt(0 * 0) = 0
    //   Max phenodigm for CARO:0000000: 0
    // For BFO:0000002 to Entity2:
    //   To BFO:0000003: phenodigm = sqrt(0 * 0.3333333333333333) = 0
    //   To BFO:0000004: phenodigm = sqrt(0.48542682717024171 * 0.6666666666666666) ≈ 0.5688742258
    //   Max phenodigm for BFO:0000002: 0.5688742258
    // Average max phenodigm from Entity1 to Entity2: (0 + 0.5688742258) / 2 ≈ 0.2844371129

    // Case 2:
    // Entity1: ["BFO:0000003"]
    // Entity2: ["BFO:0000035"]
    // For BFO:0000003 to Entity2:
    //   To BFO:0000035: sqrt(1.9593580155026542 * 0.6666666666666666) ≈ 1.1429079915
    //   Max phenodigm for BFO:0000003: 1.1429079915
    // Average max phenodigm from Entity1 to Entity2: 1.1429079915

    // Case 3:
    // Entity1: ["BFO:0000002", "BFO:0000004", "BFO:0000003"]
    // Entity2: ["BFO:0000003", "BFO:0000004"]
    // For BFO:0000002 to Entity2:
    //   To BFO:0000003: sqrt(0 * 0.3333333333333333) = 0
    //   To BFO:0000004: sqrt(0.48542682717024171 * 0.6666666666666666) ≈ 0.5688742258
    //   Max phenodigm for BFO:0000002: 0.568872988
    // For BFO:0000004 to Entity2:
    //   To BFO:0000003: sqrt(0 * 0.25) ≈ 0
    //   To BFO:0000004: sqrt(1.1292830169449666 * 1) = 1.0626772873
    //   Max phenodigm for BFO:0000004: 1.0626772873
    // For BFO:0000003 to Entity2:
    //   To BFO:0000003: sqrt(1.9593580155026542 * 1) ≈ 1.399770701
    //   To BFO:0000004: sqrt(0 * 0.25) ≈ 0
    //   Max phenodigm for BFO:0000003: 1.399770701
    // Average max phenodigm from Entity1 to Entity2: (0.568872988 + 1.0626772873 + 1.399770701) / 3 ≈ 1.0104403254

    // Case 4:
    // Entity1: ["CARO:0000000", "BFO:0000002", "BFO:0000004"]
    // Entity2: ["BFO:0000001", "BFO:0000004"]
    // For CARO:0000000 to Entity2:
    //   To BFO:0000001: sqrt(0 * 0) = 0
    //   To BFO:0000004: sqrt(0 * 0) = 0
    //   Max phenodigm for CARO:0000000: 0
    // For BFO:0000002 to Entity2:
    //   To BFO:0000001: sqrt(0 * 0.5) = 0
    //   To BFO:0000004: sqrt(0.48542682717024171 * 0.6666666666666666) ≈ 0.5688742258
    //   Max phenodigm for BFO:0000002: 0.5688742258
    // For BFO:0000004 to Entity2:
    //   To BFO:0000001: sqrt(0 * 0.3333333333333333) = 0
    //   To BFO:0000004: sqrt(1.1292830169449666 * 1) = 1.0626772873
    //   Max phenodigm for BFO:0000004: 1.0626772873
    // Average max phenodigm from Entity1 to Entity2: (0 + 0.5688742258 + 1.0626772873) / 3 ≈ 0.5438505044

    #[rstest]
    #[case(
        vec!["CARO:0000000", "BFO:0000002"],
        vec!["BFO:0000003", "BFO:0000004"],
        0.28443711290026885
    )]
    #[case(
        vec!["BFO:0000003"],
        vec!["BFO:0000035"],
        1.142907991485653
    )]
    #[case(
        vec!["BFO:0000002", "BFO:0000004", "BFO:0000003"],
        vec!["BFO:0000003", "BFO:0000004"],
        1.0104407380486145
    )]
    #[case(
        vec!["CARO:0000000", "BFO:0000002", "BFO:0000004"],
        vec!["BFO:0000001", "BFO:0000004"],
        0.5438505043671094
    )]
    fn test_calculate_avg_of_max_phenodigm_score(
        #[case] entity1_terms: Vec<&str>,
        #[case] entity2_terms: Vec<&str>,
        #[case] expected_value: f64,
    ) {
        let predicates: Option<Vec<Predicate>> = Some(vec![Predicate::from("rdfs:subClassOf")]);
        let mut rss = RustSemsimian::new(Some(BFO_SPO.clone()), predicates.clone(), None, None);
        rss.update_closure_and_ic_map();

        let entity1: HashSet<TermID> = entity1_terms.into_iter().map(|s| s.to_string()).collect();
        let entity2: HashSet<TermID> = entity2_terms.into_iter().map(|s| s.to_string()).collect();

        let phenodigm_score = calculate_average_of_max_phenodigm_score(&rss, &entity1, &entity2);

        println!("Phenodigm score: {phenodigm_score}");
        assert!((phenodigm_score - expected_value).abs() < f64::EPSILON);
    }


    #[rstest]
    #[case(vec!["CARO:0000000", "BFO:0000002"], vec!["BFO:0000003", "BFO:0000004"], 0.3333333333333333)]
    #[case(vec!["BFO:0000003"], vec!["BFO:0000035"], 0.6666666666666666)]
    #[case(vec!["BFO:0000002", "BFO:0000004", "BFO:0000003"], vec!["BFO:0000001", "BFO:0000004"], 0.7222222222222222)]
    #[case(vec!["CARO:0000000", "BFO:0000002", "BFO:0000004"], vec!["BFO:0000001", "BFO:0000004"], 0.5555555555555555)]
    fn test_calculate_avg_of_max_jaccard_similarity(
        #[case] entity1_terms: Vec<&str>,
        #[case] entity2_terms: Vec<&str>,
        #[case] expected_value: f64,
    ) {
        let predicates: Option<Vec<Predicate>> = Some(vec![Predicate::from("rdfs:subClassOf")]);
        let entity1: HashSet<TermID> = entity1_terms.into_iter().map(|s| s.to_string()).collect();
        let entity2: HashSet<TermID> = entity2_terms.into_iter().map(|s| s.to_string()).collect();

        let mut rss = RustSemsimian::new(Some(BFO_SPO.clone()), predicates.clone(), None, None);
        rss.update_closure_and_ic_map();

        let jaccard_similarity =
            calculate_average_of_max_jaccard_similarity(&rss, &entity1, &entity2);

        println!("Jaccard similarity: {jaccard_similarity}");
        assert!((jaccard_similarity - expected_value).abs() < f64::EPSILON);
    }

    // These comments are the manual calculations for the test cases below, for future reference

    // These are the values that are being used in the manual calculations below:

    // GO Term Pair	            Max IC
    // GO:0005773_GO:0031965	5.112700132749362
    // GO:0016020_GO:0031965	4.8496657269155685
    // GO:0016020_GO:0005773	2.264703226194412
    // GO:0005773_GO:0005773	7.4346282276367246
    // GO:0005634_GO:0005773	5.112700132749362
    // GO:0005634_GO:0031965	5.8496657269155685

    // Case 1:
    // Entity1: ["GO:0005634", "GO:0016020"]
    // Entity2: ["GO:0031965", "GO:0005773"]

    // For GO:0005634 to Entity2:
    //   To GO:0031965: phenodigm = 5.8496657269155685
    //   To GO:0005773: phenodigm = 5.112700132749362
    //   Max phenodigm for GO:0005634: 5.8496657269155685
    // For GO:0016020 to Entity2:
    //   To GO:0031965: phenodigm = 4.8496657269155685
    //   To GO:0005773: phenodigm = 2.264703226194412
    //   Max phenodigm for GO:0016020: 4.8496657269155685
    // Average max phenodigm from Entity1 to Entity2: (5.8496657269155685 + 4.8496657269155685) / 2 ≈ 5.3496657269155685

    // For GO:0031965 to Entity1:
    //   To GO:0005634: phenodigm = 5.8496657269155685
    //   To GO:0016020: phenodigm = 4.8496657269155685
    //   Max phenodigm for GO:0031965: 5.8496657269155685
    // For GO:0005773 to Entity1:
    //   To GO:0005634: phenodigm = 5.112700132749362
    //   To GO:0016020: phenodigm = 2.264703226194412
    //   Max phenodigm for GO:0005773: 5.112700132749362
    // Average max phenodigm from Entity2 to Entity1: (5.8496657269155685 + 5.112700132749362) / 2 = 5.481182929832465

    // Average of the two averages: (5.3496657269155685 + 5.481182929832465) / 2 ≈ 5.4154243284

    // Case 2:
    // Entity1: ["GO:0005634", "GO:0016020", "GO:0005773"]
    // Entity2: ["GO:0031965", "GO:0005773"]

    // For GO:0005634 to Entity2:
    //   To GO:0031965: phenodigm = 5.8496657269155685
    //   To GO:0005773: phenodigm = 5.112700132749362
    //   Max phenodigm for GO:0005634: 5.8496657269155685
    // For GO:0016020 to Entity2:
    //   To GO:0031965: phenodigm = 4.8496657269155685
    //   To GO:0005773: phenodigm = 2.264703226194412
    //   Max phenodigm for GO:0016020: 4.8496657269155685
    // For GO:0005773 to Entity2:
    //   To GO:0031965: phenodigm = 5.112700132749362
    //   To GO:0005773: phenodigm = 7.4346282276367246
    //   Max phenodigm for GO:0005773: 7.4346282276367246
    // Average max phenodigm from Entity1 to Entity2: (5.8496657269155685 + 4.8496657269155685 + 7.4346282276367246) / 3 = 6.0446532272

    // For GO:0031965 to Entity1:
    //   To GO:0005634: phenodigm = 5.8496657269155685
    //   To GO:0016020: phenodigm = 4.8496657269155685
    //   To GO:0005773: phenodigm = 5.112700132749362
    //   Max phenodigm for GO:0031965: 5.8496657269155685
    // For GO:0005773 to Entity1:
    //   To GO:0005634: phenodigm = 5.112700132749362
    //   To GO:0016020: phenodigm = 2.264703226194412
    //   To GO:0005773: phenodigm = 7.4346282276367246
    //   Max phenodigm for GO:0005773: 7.4346282276367246
    // Average max phenodigm from Entity2 to Entity1: (5.8496657269155685 + 7.4346282276367246) / 2 = 6.6421469773

    // Average of the two averages: (6.0446532272 + 6.6421469773) / 2 ≈ 6.3434001023

    #[rstest]
    #[case(
        vec!["GO:0005634", "GO:0016020"],
        vec!["GO:0031965", "GO:0005773"],
        5.4154243283740175,
        MetricEnum::AncestorInformationContent
    )]
    #[case(
        vec!["GO:0005634", "GO:0016020", "GO:0005773"],
        vec!["GO:0031965", "GO:0005773"],
        6.34340010221605,
        MetricEnum::AncestorInformationContent
    )]
    fn test_calculate_average_termset_information_content(
        #[case] entity1_terms: Vec<&str>,
        #[case] entity2_terms: Vec<&str>,
        #[case] expected_value: f64,
        #[case] score_metric: MetricEnum,
    ) {
        let predicates: Option<Vec<Predicate>> = Some(vec![
            Predicate::from("rdfs:subClassOf"),
            Predicate::from("BFO:0000050"),
        ]);
        let db = Some("tests/data/go-nucleus.db");
        let mut rss = RustSemsimian::new(None, predicates, None, db);

        rss.update_closure_and_ic_map();

        let entity1: HashSet<TermID> = entity1_terms.into_iter().map(|s| s.to_string()).collect();
        let entity2: HashSet<TermID> = entity2_terms.into_iter().map(|s| s.to_string()).collect();

        let avg_ic_score = calculate_average_termset_information_content(&rss, &entity1, &entity2);
        assert_eq!(avg_ic_score, expected_value);

        let tsps = rss.termset_pairwise_similarity(&entity1, &entity2, &score_metric);
        dbg!(&tsps);
    }

    // These comments are the manual calculations for the test cases below, for future reference

    // These are the values that are being used in the manual calculations below:

    // GO Term Pair	            Max IC	            Jaccard Similarity
    // GO:0005773_GO:0031965	5.112700132749362	0.6
    // GO:0016020_GO:0031965	4.8496657269155685	0.34782608695652173
    // GO:0016020_GO:0005773	2.264703226194412	0.3888888888888889
    // GO:0005773_GO:0005773	7.4346282276367246	1
    // GO:0005634_GO:0005773	5.112700132749362	0.8333333333333334
    // GO:0005634_GO:0031965	5.8496657269155685	0.6956521739130435

    // Case 1:
    // Entity1: ["GO:0005634", "GO:0016020"]
    // Entity2: ["GO:0031965", "GO:0005773"]

    // For GO:0005634 to Entity2:
    //   To GO:0031965: phenodigm = sqrt(5.8496657269155685 * 0.6956521739130435) = 2.0172587042
    //   To GO:0005773: phenodigm = sqrt(5.112700132749362 * 0.8333333333333334) = 2.064118079
    //   Max phenodigm for GO:0005634: 2.064118079
    // For GO:0016020 to Entity2:
    //   To GO:0031965: phenodigm = sqrt(4.8496657269155685 * 0.34782608695652173) = 1.2987841441
    //   To GO:0005773: phenodigm = sqrt(2.264703226194412 * 0.3888888888888889) = 0.9384657273
    //   Max phenodigm for GO:0016020: 1.2987841441
    // Average max phenodigm from Entity1 to Entity2: (2.064118079 + 1.2987841441) / 2 ≈ 1.6814511116

    // For GO:0031965 to Entity1:
    //   To GO:0005634: phenodigm = sqrt(5.8496657269155685 * 0.6956521739130435) = 2.0172587042
    //   To GO:0016020: phenodigm = sqrt(4.8496657269155685 * 0.34782608695652173) = 1.2987841441
    //   Max phenodigm for GO:0031965: 2.0172587042
    // For GO:0005773 to Entity1:
    //   To GO:0005634: phenodigm = sqrt(5.112700132749362 * 0.8333333333333334) = 2.064118079
    //   To GO:0016020: phenodigm = sqrt(2.264703226194412 * 0.3888888888888889) = 0.9384657273
    //   Max phenodigm for GO:0005773: 2.064118079
    // Average max phenodigm from Entity2 to Entity1: (2.0172587042 + 2.064118079) / 2 ≈ 2.0406883916

    // Average of the two averages: (1.6814511116 + 2.0406883916) / 2 ≈ 1.8610697516

    // Case 2:
    // Entity1: ["GO:0005634", "GO:0016020", "GO:0005773"]
    // Entity2: ["GO:0031965", "GO:0005773"]

    // For GO:0005634 to Entity2:
    //   To GO:0031965: phenodigm = sqrt(5.8496657269155685 * 0.6956521739130435) = 2.0172587042
    //   To GO:0005773: phenodigm = sqrt(5.112700132749362 * 0.8333333333333334) = 2.064118079
    //   Max phenodigm for GO:0005634: 2.064118079
    // For GO:0016020 to Entity2:
    //   To GO:0031965: phenodigm = sqrt(4.8496657269155685 * 0.34782608695652173) = 1.2987841441
    //   To GO:0005773: phenodigm = sqrt(2.264703226194412 * 0.3888888888888889) = 0.9384657273
    //   Max phenodigm for GO:0016020: 1.2987841441
    // For GO:0005773 to Entity2:
    //   To GO:0031965: phenodigm = sqrt(5.112700132749362 * 0.6) = 1.751462269
    //   To GO:0005773: phenodigm = sqrt(7.4346282276367246 * 1) = 2.7266514679
    //   Max phenodigm for GO:0005773: 2.7266514679
    // Average max phenodigm from Entity1 to Entity2: (2.064118079 + 1.2987841441 + 2.7266514679) / 3 ≈ 2.0298512303

    // For GO:0031965 to Entity1:
    //   To GO:0005634: phenodigm = sqrt(5.8496657269155685	* 0.6956521739130435) = 2.0172587042
    //   To GO:0016020: phenodigm = sqrt(4.8496657269155685	* 0.34782608695652173) = 1.2987841441
    //   To GO:0005773: phenodigm = sqrt(5.112700132749362	* 0.6) = 1.751462269
    //   Max phenodigm for GO:0031965: 2.0172587042
    // For GO:0005773 to Entity1:
    //   To GO:0005634: phenodigm = sqrt(5.112700132749362 * 0.8333333333333334) = 2.064118079
    //   To GO:0016020: phenodigm = sqrt(2.264703226194412 * 0.3888888888888889) = 0.9384657273
    //   To GO:0005773: phenodigm = sqrt(7.4346282276367246	* 1) = 2.7266514679
    //   Max phenodigm for GO:0005773: 2.7266514679
    // Average max phenodigm from Entity2 to Entity1: (2.0172587042 + 2.7266514679) / 2 = 2.3719550861

    // Average of the two averages: (2.0298512303 + 2.3719550861) / 2 ≈ 2.2009031582

    #[rstest]
    #[case(
        vec!["GO:0005634", "GO:0016020"],
        vec!["GO:0031965", "GO:0005773"],
        1.8610697515464185,
        MetricEnum::PhenodigmScore
    )]
    #[case(
        vec!["GO:0005634", "GO:0016020", "GO:0005773"],
        vec!["GO:0031965", "GO:0005773"],
        2.2009031581929213,
        MetricEnum::PhenodigmScore
    )]
    fn test_calculate_average_termset_phenodigm_score(
        #[case] entity1_terms: Vec<&str>,
        #[case] entity2_terms: Vec<&str>,
        #[case] expected_value: f64,
        #[case] score_metric: MetricEnum,
    ) {
        let predicates: Option<Vec<Predicate>> = Some(vec![
            Predicate::from("rdfs:subClassOf"),
            Predicate::from("BFO:0000050"),
        ]);
        let db = Some("tests/data/go-nucleus.db");
        let mut rss = RustSemsimian::new(None, predicates, None, db);

        rss.update_closure_and_ic_map();

        let entity1: HashSet<TermID> = entity1_terms.into_iter().map(|s| s.to_string()).collect();
        let entity2: HashSet<TermID> = entity2_terms.into_iter().map(|s| s.to_string()).collect();

        let avg_phenodigm_score = calculate_average_termset_phenodigm_score(&rss, &entity1, &entity2);
        assert_eq!(avg_phenodigm_score, expected_value);

        let tsps = rss.termset_pairwise_similarity(&entity1, &entity2, &score_metric);
        dbg!(&tsps);
    }


    // These comments are the manual calculations for the test cases below, for future reference

    // These are the values that are being used in the manual calculations below:

    // GO Term Pair	            Jaccard Similarity
    // GO:0005773_GO:0031965    0.6
    // GO:0016020_GO:0031965	0.34782608695652173
    // GO:0016020_GO:0005773    0.3888888888888889
    // GO:0005773_GO:0005773	1
    // GO:0005634_GO:0005773    0.8333333333333334
    // GO:0005634_GO:0031965	0.6956521739130435

    // Case 1:
    // Entity1: ["GO:0005634", "GO:0016020"]
    // Entity2: ["GO:0031965", "GO:0005773"]

    // For GO:0005634 to Entity2:
    //   To GO:0031965: phenodigm = 0.6956521739130435
    //   To GO:0005773: phenodigm = 0.8333333333333334
    //   Max phenodigm for GO:0005634: 0.8333333333333334
    // For GO:0016020 to Entity2:
    //   To GO:0031965: phenodigm = 0.34782608695652173
    //   To GO:0005773: phenodigm = 0.3888888888888889
    //   Max phenodigm for GO:0016020: 0.3888888888888889
    // Average max phenodigm from Entity1 to Entity2: (0.8333333333333334 + 0.3888888888888889) / 2 ≈ 0.6111111111

    // For GO:0031965 to Entity1:
    //   To GO:0005634: phenodigm = 0.6956521739130435
    //   To GO:0016020: phenodigm = 0.34782608695652173
    //   Max phenodigm for GO:0031965: 0.6956521739130435
    // For GO:0005773 to Entity1:
    //   To GO:0005634: phenodigm = 0.8333333333333334
    //   To GO:0016020: phenodigm = 0.3888888888888889
    //   Max phenodigm for GO:0005773: 0.8333333333333334
    // Average max phenodigm from Entity2 to Entity1: (0.6956521739130435 + 0.8333333333333334) / 2 = 0.7644927536

    // Average of the two averages: (0.6111111111 + 0.7644927536) / 2 ≈ 0.6878019324

    // Case 2:
    // Entity1: ["GO:0005634", "GO:0016020", "GO:0005773"]
    // Entity2: ["GO:0031965", "GO:0005773"]

    // For GO:0005634 to Entity2:
    //   To GO:0031965: phenodigm = 0.6956521739130435
    //   To GO:0005773: phenodigm = 0.8333333333333334
    //   Max phenodigm for GO:0005634: 0.8333333333333334
    // For GO:0016020 to Entity2:
    //   To GO:0031965: phenodigm = 0.34782608695652173
    //   To GO:0005773: phenodigm = 0.3888888888888889
    //   Max phenodigm for GO:0016020: 0.3888888888888889
    // For GO:0005773 to Entity2:
    //   To GO:0031965: phenodigm = 0.6
    //   To GO:0005773: phenodigm = 1
    //   Max phenodigm for GO:0005773: 1
    // Average max phenodigm from Entity1 to Entity2: (0.8333333333333334 + 0.3888888888888889 + 1) / 3 ≈ 0.7407407407

    // For GO:0031965 to Entity1:
    //   To GO:0005634: phenodigm = 0.6956521739130435
    //   To GO:0016020: phenodigm = 0.34782608695652173
    //   To GO:0005773: phenodigm = 0.6
    //   Max phenodigm for GO:0031965: 0.6956521739130435
    // For GO:0005773 to Entity1:
    //   To GO:0005634: phenodigm = 0.8333333333333334
    //   To GO:0016020: phenodigm = 0.3888888888888889
    //   To GO:0005773: phenodigm = 1
    //   Max phenodigm for GO:0005773: 1
    // Average max phenodigm from Entity2 to Entity1: (0.6956521739130435 + 1) / 2 = 0.847826087

    // Average of the two averages: (0.7407407407 + 0.847826087) / 2 ≈ 0.7942834139
    #[rstest]
    #[case(
        vec!["GO:0005634", "GO:0016020"],
        vec!["GO:0031965", "GO:0005773"],
        0.6878019323671498,
        MetricEnum::JaccardSimilarity
    )]
    #[case(
        vec!["GO:0005634", "GO:0016020", "GO:0005773"],
        vec!["GO:0031965", "GO:0005773"],
        0.7942834138486312,
        MetricEnum::JaccardSimilarity
    )]
    fn test_calculate_average_termset_jaccard_similarity(
        #[case] entity1_terms: Vec<&str>,
        #[case] entity2_terms: Vec<&str>,
        #[case] expected_value: f64,
        #[case] score_metric: MetricEnum,
    ) {
        let predicates: Option<Vec<Predicate>> = Some(vec![
            Predicate::from("rdfs:subClassOf"),
            Predicate::from("BFO:0000050"),
        ]);
        let db = Some("tests/data/go-nucleus.db");
        let mut rss = RustSemsimian::new(None, predicates, None, db);

        rss.update_closure_and_ic_map();

        let entity1: HashSet<TermID> = entity1_terms.into_iter().map(|s| s.to_string()).collect();
        let entity2: HashSet<TermID> = entity2_terms.into_iter().map(|s| s.to_string()).collect();

        let avg_jaccard_similarity = calculate_average_termset_jaccard_similarity(&rss, &entity1, &entity2);
        assert_eq!(avg_jaccard_similarity, expected_value);

        let tsps = rss.termset_pairwise_similarity(&entity1, &entity2, &score_metric);
        dbg!(&tsps);
    }
}
