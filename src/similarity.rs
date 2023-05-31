use crate::{utils::expand_term_using_closure, utils::predicate_set_to_key};
use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet};
use crate::PredicateSetKey;
use crate::TermID;
use crate::Predicate;


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

    println!("SIM: entity1_closure: {:?}", entity1_closure);
    println!("SIM: entity2_closure: {:?}", entity2_closure);
    println!("SIM: Jaccard: {}", jaccard);

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
    use std::collections::{HashMap, HashSet};
    use crate::test_utils::test_constants::*;



    #[test]
    fn test_semantic_jaccard_similarity_new() {
        let mut sco_predicate: HashSet<Predicate> = HashSet::new();
        sco_predicate.insert(String::from("subClassOf"));

        let result = calculate_semantic_jaccard_similarity(
            &*CLOSURE_MAP,
            "CARO:0000000",
            "BFO:0000002",
            &Some(sco_predicate.clone()),
        );

        println!("{:?}", result);
        assert_eq!(result, 2.0 / 3.0);
        
        // NO Predicate
        // let mut sco_no_predicate: HashSet<Predicate> = HashSet::new();
        let result2 = calculate_semantic_jaccard_similarity(
            &*CLOSURE_MAP,
            "BFO:0000002",
            "BFO:0000003",
            &Some(sco_predicate.clone()),
        );
        println!("{:?}", result2);
        assert_eq!(result2, 1.0 / 3.0);

        let mut sco_po_predicate: HashSet<String> = HashSet::new();
        sco_po_predicate.insert(String::from("subClassOf"));
        sco_po_predicate.insert(String::from("partOf"));

        let result3 = calculate_semantic_jaccard_similarity(
            &*CLOSURE_MAP2,
            "BFO:0000002",
            "BFO:0000003",
            &Some(sco_po_predicate.clone()),
        );
        println!("{:?}", result3);
        assert_eq!(result3, 1.0 / 3.0);
    }


    #[test]
    fn test_semantic_jaccard_similarity_fruits() {
        let _closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = HashMap::new();
        let mut related_to_predicate: HashSet<Predicate> = HashSet::new();
        related_to_predicate.insert(String::from("related_to"));
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
        let no_predicate: Option<HashSet<Predicate>> = None;
        let result2 = calculate_semantic_jaccard_similarity(
            &ALL_NO_PRED_MAP,
            "banana",
            "orange",
            &no_predicate);
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

    #[test]
    fn test_calculate_phenomizer_score() {
        let mut entity_one = HashSet::new();
        entity_one.insert(String::from("CARO:0000000")); // resnik of best match = 5
        entity_one.insert(String::from("BFO:0000002")); // resnik of best match = 4
    
        let mut entity_two = HashSet::new();
        entity_two.insert(String::from("BFO:0000003")); // resnik of best match = 3
        entity_two.insert(String::from("BFO:0000002")); // resnik of best match = 4
        entity_two.insert(String::from("CARO:0000000")); // resnik of best match = 5
    
        let expected = ((5.0 + 4.0) / 2.0 + (3.0 + 4.0 + 5.0) / 3.0) / 2.0;
    
        let result = calculate_phenomizer_score(MAP.clone(), entity_one, entity_two);
        assert_eq!(result, expected);
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

        let predicates = Some(HashSet::from([String::from("subClassOf")]));
        let result = calculate_max_information_content(
            &CLOSURE_MAP,
            &IC_MAP,
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
