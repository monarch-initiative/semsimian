use crate::{utils::expand_term_using_closure, utils::predicate_set_to_key};
use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet};

type Predicate = String;
type TermID = String;
type PredicateSetKey = String;

pub fn calculate_semantic_jaccard_similarity(
    closure_table: &HashMap<String, HashMap<String, HashSet<String>>>,
    entity1: &str,
    entity2: &str,
    predicates: &Option<HashSet<String>>,
) -> f64 {
    /* Returns semantic Jaccard similarity between the two sets. */
    let entity1_closure = expand_term_using_closure(entity1, closure_table, &predicates);
    let entity2_closure = expand_term_using_closure(entity2, closure_table, &predicates);
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
    closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    ic_map: &HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    entity1: HashSet<String>,
    entity2: HashSet<String>,
    predicates: &Option<HashSet<String>>
) -> f64 {
    // calculate average resnik sim of all terms in entity1 and their best match in entity2
    let entity1_to_entity2_average_resnik_sim: f64 =
        pairwise_entity_resnik_score(closure_map, ic_map,  &entity1, &entity2, &predicates);
    // now do the same for entity2 to entity1
    let entity2_to_entity1_average_resnik_sim: f64 =
        pairwise_entity_resnik_score(closure_map, ic_map, &entity2, &entity1, &predicates);
    // return the average of the two
    return (entity1_to_entity2_average_resnik_sim + entity2_to_entity1_average_resnik_sim) / 2.0;
}

pub fn pairwise_entity_resnik_score(
    closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    ic_map: &HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    entity1: &HashSet<String>,
    entity2: &HashSet<String>,
    predicates: &Option<HashSet<String>>,
) -> f64 {
    let mut entity1_to_entity2_sum_resnik_sim = 0.0;

    for e1_term in entity1.clone().into_iter() {
        let mut max_resnik_sim_e1_e2 = 0.0;
        for e2_term in entity2.clone().into_iter() {
            // NB: this will definitely fail if the term is not in the map
            let this_resnik = calculate_max_information_content(closure_map, ic_map, &e1_term, &e2_term, &predicates);
            if this_resnik > max_resnik_sim_e1_e2 {
                max_resnik_sim_e1_e2 = this_resnik;
            }
        }
        entity1_to_entity2_sum_resnik_sim += max_resnik_sim_e1_e2;
    }
    let entity1_to_entity2_average_resnik_sim =
        entity1_to_entity2_sum_resnik_sim / entity1.len() as f64;
    return entity1_to_entity2_average_resnik_sim;
}

pub fn calculate_max_information_content(
    closure_map: &HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    ic_map: &HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    entity1: &str,
    entity2: &str,
    predicates: &Option<HashSet<Predicate>>,
) -> f64 {
    // CODE TO CALCULATE MAX IC
    let filtered_common_ancestors: Vec<String> =
        common_ancestors(closure_map, &entity1, &entity2, &predicates);

    let predicate_set_key = predicate_set_to_key(predicates);

    // for each member of filtered_common_ancestors, find the entry for it in ic_map
    let mut max_ic: f64 = 0.0;
    for ancestor in filtered_common_ancestors.iter() {
        if let Some(ic) = ic_map
            .get(&predicate_set_key)
            .expect("Finding ancestor in ic map")
            .get(ancestor)
        {
            if *ic > max_ic {
                max_ic = *ic;
            }
        }
    }
    // then return the String and f64 for the filtered_common_ancestors with the highest f64
    max_ic
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
    predicates: &Option<HashSet<Predicate>>,
) -> Vec<String> {
    // expand_term_using_closure() handles case of the entity being not present -> returning empty set
    let entity1_closure = expand_term_using_closure(entity1, closure_map, predicates);
    let entity2_closure = expand_term_using_closure(entity2, closure_map, predicates);

    entity1_closure
        .into_iter()
        .filter(|ancestor| entity2_closure.contains(ancestor))
        .collect()
}

// scores: maps ancestors to corresponding IC scores
fn _mrca_and_score(scores: &HashMap<TermID, f64>) -> (Option<TermID>, f64) {
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

#[cfg(test)]
mod tests {
    use crate::utils::numericize_sets;

    use super::*;

    #[test]
    fn test_semantic_jaccard_similarity() {
        let mut closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::new();

        // closure map looks like this:
        // +subClassOf -> CARO:0000000 -> CARO:0000000, BFO:0000002, BFO:0000003
        //             -> BFO:0000002 -> BFO:0000002, BFO:0000003
        //             -> BFO:0000003 -> BFO:0000003
        //             -> BFO:0000004 -> BFO:0000004

        let mut map: HashMap<TermID, HashSet<TermID>> = HashMap::new();
        let mut set: HashSet<TermID> = HashSet::new();
        set.insert(String::from("CARO:0000000"));
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("CARO:0000000"), set);

        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("BFO:0000002"));
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("BFO:0000002"), set);

        let mut set: HashSet<String> = HashSet::new();
        set.insert(String::from("BFO:0000003"));
        map.insert(String::from("BFO:0000003"), set);
        closure_map.insert(String::from("+subClassOf"), map);

        // make another closure map for subclassof + partof
        // +partOf+subClassOf -> CARO:0000000 -> CARO:0000000, BFO:0000002, BFO:0000003
        //             -> BFO:0000002 -> BFO:0000002, BFO:0000003
        //             -> BFO:0000003 -> BFO:0000003, BFO:0000004 <- +partOf
        //             -> BFO:0000004 -> BFO:0000004
        let mut closure_map2: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> =
            HashMap::new();
        closure_map2.insert(
            String::from("+partOf+subClassOf"),
            closure_map.get("+subClassOf").unwrap().clone(),
        );
        closure_map2
            .get_mut("+partOf+subClassOf")
            .unwrap()
            .get_mut(&String::from("BFO:0000003"))
            .unwrap()
            .insert(String::from("BFO:0000004"));

        let mut sco_predicate: HashSet<Predicate> = HashSet::new();
        sco_predicate.insert(String::from("subClassOf"));

        let result = calculate_semantic_jaccard_similarity(
            &closure_map,
            "CARO:0000000",
            "BFO:0000002",
            &Some(sco_predicate.clone()),
        );
        println!("{result}");
        assert_eq!(result, 2.0 / 3.0);

        let result2 = calculate_semantic_jaccard_similarity(
            &closure_map,
            "BFO:0000002",
            "BFO:0000003",
            &Some(sco_predicate.clone()),
        );
        println!("{result2}");
        assert_eq!(result2, 0.5);

        let mut sco_po_predicate: HashSet<String> = HashSet::new();
        sco_po_predicate.insert(String::from("subClassOf"));
        sco_po_predicate.insert(String::from("partOf"));

        // with the refactor of closure map, this test doesn't really test anything more than
        // the previous tests
        let result3 = calculate_semantic_jaccard_similarity(
            &closure_map2,
            "BFO:0000002",
            "BFO:0000003",
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
        let closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = HashMap::from(
            [(
                    String::from("+subClassOf"),
                    HashMap::from([
                        (
                            String::from("CARO:0000000"),
                            HashSet::from([
                                String::from("CARO:0000000"),
                                String::from("BFO:0000002"),
                                String::from("BFO:0000003"),
                            ]),
                        ),
                        (
                            String::from("BFO:0000002"),
                            HashSet::from([
                                String::from("BFO:0000002"),
                                String::from("BFO:0000003"),
                            ]),
                        ),
                        (
                            String::from("BFO:0000003"),
                            HashSet::from([
                                String::from("BFO:0000003"),
                                String::from("BFO:0000004"),
                            ]),
                        ),
                        (
                            String::from("BFO:0000004"),
                            HashSet::from([String::from("BFO:0000004")]),
                        ),
                    ]),
                ),
                (
                    String::from("+partOf"),
                    HashMap::from([
                        (
                            String::from("BFO:0000003"),
                            HashSet::from([String::from("BFO:0000004")]),
                        ),
                    ]),
                )]
        );
        let ic_map: HashMap<PredicateSetKey, HashMap<TermID, f64>> = [(
            String::from("+subClassOf"),
            [
                (String::from("CARO:0000000"), 5.0),
                (String::from("BFO:0000002"), 4.0),
                (String::from("BFO:0000003"), 3.0),
            ]
            .iter()
            .cloned()
            .collect(),
        )]
        .iter()
        .cloned()
        .collect();

        let mut entity_one = HashSet::new();
        entity_one.insert(String::from("CARO:0000000")); // resnik of best match = 5
        entity_one.insert(String::from("BFO:0000002")); // resnik of best match = 4

        let mut entity_two = HashSet::new();
        entity_two.insert(String::from("BFO:0000003")); // resnik of best match = 3 (for subClassOf only, best match = 0)
        entity_two.insert(String::from("BFO:0000002")); // resnik of best match = 4
        entity_two.insert(String::from("CARO:0000000")); // resnik of best match = 5

        // test with just subClassOf predicate
        let expected = ((5.0 + 4.0) / 2.0 + (0.0 + 4.0 + 5.0) / 3.0) / 2.0;
        let predicate_set = Some(HashSet::from([String::from("subClassOf")]));
        let result = calculate_phenomizer_score(&closure_map, &ic_map, entity_one.clone(), entity_two.clone(), &predicate_set);
        assert_eq!(result, expected);

        // test with just partOf predicate
        // TODO

        // test with both subClassOf and partOf predicates
        // TODO
        let expected = ((5.0 + 4.0) / 2.0 + (3.0 + 4.0 + 5.0) / 3.0) / 2.0;
        let predicate_set = Some(HashSet::from([String::from("subClassOf")]));
        let result = calculate_phenomizer_score(&closure_map, &ic_map, entity_one.clone(), entity_two.clone(), &predicate_set);
        assert_eq!(result, expected);

        // test with no predicates set (that is, all predicates should be used)
        // TODO
    }

    // TODO: test that closure map in Semsimian object is correct
    // TODO: test that ic map in Semsimian object is correct

    #[test]
    fn test_calculate_max_information_content() {
        let ic_map: HashMap<PredicateSetKey, HashMap<TermID, f64>> = [(
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
            &closure_map,
            &ic_map,
            &String::from("CARO:0000000"),
            &String::from("BFO:0000002"),
            &predicates,
        );
        println!("Max IC: {}", result);
        let expected_value = 1.585;
        assert!(
            (result - expected_value).abs() < 1e-3,
            "Expected value: {}, got: {}",
            expected_value,
            result
        );
    }
}
