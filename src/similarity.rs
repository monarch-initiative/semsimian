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

// test_calculate_avg_of_max_phenodigm_score
    // Case 1
        // These are the values that are being used in the manual calculations below
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

        // Entity1 to Entity2
        // Entity1: ["CARO:0000000", "BFO:0000002"]
        // Entity2: ["BFO:0000003", "BFO:0000004"]
        // Calculations for Entity1 to Entity2
        // For CARO:0000000:
        //
        // To BFO:0000003: phenodigm = sqrt(0 * 0) = 0
        // To BFO:0000004: phenodigm = sqrt(0 * 0) = 0
        // Max phenodigm for CARO:0000000: 0
        // For BFO:0000002:
        //
        // To BFO:0000003: phenodigm = sqrt(0 * 0.3333333333333333) = 0
        // To BFO:0000004: phenodigm = sqrt(0.48542682717024171 * 0.6666666666666666) ≈ sqrt(0.32361788478016114) ≈ 0.568872988
        // Max phenodigm for BFO:0000002: 0.568872988
        // Average max phenodigm from Entity1 to Entity2: (0 + 0.568872988) / 2 ≈ 0.284436494

    // Case 2
        // Entity1 to Entity2
        // Entity1: ["BFO:0000003"]
        // Entity2: ["BFO:0000035"]

        // Max IC for "BFO:0000003_BFO:0000035": 1.9593580155026542
        // Jaccard similarity for "BFO:0000003_BFO:0000035": 0.6666666666666666
        // Phenodigm Score Calculation
        // Phenodigm = sqrt(1.9593580155026542 * 0.6666666666666666) ≈ 1.142907991485653


        // For CARO:0000000:
        // Comparing to BFO:0000001: Max IC = 0, Jaccard = 0
        // Comparing to BFO:0000004: Max IC = 0, Jaccard = 0
        // Max phenodigm for CARO:0000000: 0 (since both comparisons yield 0)

        // For BFO:0000002:
        // Comparing to BFO:0000001: Max IC = 0, Jaccard = 0.5, phenodigm is (sqrt(0 * 0.5) = 0
        // Comparing to BFO:0000004: Max IC = 0.48542682717024171, Jaccard = 0.6666666666666666
        // phenodigm score is (sqrt(0.48542682717024171 * 0.6666666666666666) ≈ 0.5688742258
        // Max phenodigm for BFO:0000002: 0.5688742258

        // For BFO:0000004:
        // Comparing to BFO:0000002: Max IC = 0.48542682717024171, Jaccard = 0.6666666666666666
        // Comparing to itself: Max IC = 1.1292830169449666, Jaccard = 1
        // Max phenodigm for BFO:0000004: sqrt(1.1292830169449666 * 1) ≈ 1.062689

        // Average max phenodigm from Entity1 to Entity2: (0 + 0.5688742258 + 1.062689) / 3 ≈ 0.5438544086
        /////////

        // Entity2 to Entity1
        // For BFO:0000001:
        // Comparing to CARO:0000000 Max IC = 0, Jaccard = 0
        // Comparing to BFO:0000002 Max IC = 0, Jaccard = 0.5, phenodigm is (sqrt(0 * 0.5) = 0
        // Comparing to BFO:0000004 Max IC = 0, Jaccard = 0.3333333333333333, phenodigm is (sqrt(0 * 0.3333333333333333) = 0
        // Max phenodigm for BFO:0000001: 0 (since all comparisons yield 0)

        // For BFO:0000004
        // Comparing to CARO:0000000 Max IC = 0, Jaccard = 0
        // Comparing to BFO:0000002 Max IC = 0.48542682717024171, Jaccard = 0.6666666666666666, phenodigm is
        //       (sqrt(0.48542682717024171 * 0.6666666666666666) ≈ 0.5688742258
        // Comparing to itself Max IC = 1.1292830169449666, Jaccard = 1, phenodigm is (sqrt(1.1292830169449666 * 1) ≈ 1.062689
        // Max phenodigm for BFO:0000004: 1.062689
        // Average max phenodigm from Entity2 to Entity1: (0 + 1.062689) / 2 ≈ 0.5313445

        // Final Average Maximum Phenodigm Score
        // average = (0.5438544086 + 0.5313445)/2 = 0.5375994543