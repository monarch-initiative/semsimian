#[cfg(test)]
pub mod test_constants {
    use crate::{Predicate, PredicateSetKey, TermID};
    use lazy_static::lazy_static;
    use std::collections::{HashMap, HashSet};

    lazy_static! {
       pub static ref CLOSURE_MAP: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = {
         let mut closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = HashMap::new();
         let mut map: HashMap<TermID, HashSet<TermID>> = HashMap::new();
         let mut set: HashSet<TermID> = HashSet::new();

         // closure map looks like this:
         // +subClassOf -> CARO:0000000 -> CARO:0000000, BFO:0000002, BFO:0000003
         //             -> BFO:0000002 -> BFO:0000002, BFO:0000003
         //             -> BFO:0000003 -> BFO:0000003
         //             -> BFO:0000004 -> BFO:0000004

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
         set.insert(String::from("BFO:0000004"));
         map.insert(String::from("BFO:0000003"), set);

         closure_map.insert(String::from("+subClassOf"), map);
         closure_map
       };

       pub static ref CLOSURE_MAP2: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = {
         let mut closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = HashMap::new();

          // make another closure map for subclassof + partof
         // +partOf+subClassOf -> CARO:0000000 -> CARO:0000000, BFO:0000002, BFO:0000003
         //             -> BFO:0000002 -> BFO:0000002, BFO:0000003
         //             -> BFO:0000003 -> BFO:0000003, BFO:0000004 <- +partOf
         //             -> BFO:0000004 -> BFO:0000004

         closure_map.insert(String::from("+partOf+subClassOf"), CLOSURE_MAP.get("+subClassOf").unwrap().clone());
         closure_map.get_mut("+partOf+subClassOf").unwrap().get_mut(&String::from("BFO:0000003")).unwrap().insert(String::from      ("BFO:0000004"));
         closure_map
       };

       pub static ref FRUIT_CLOSURE_MAP: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = {
         let mut closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = HashMap::new();
         let _map: HashMap<TermID, HashSet<TermID>> = HashMap::new();

               // closure map looks like this:
         // +related_to -> apple -> apple, banana
         //             -> banana -> banana, orange
         //             -> orange -> orange, pear
         //             -> pear -> pear, kiwi


         let mut map: HashMap<TermID, HashSet<TermID>> = HashMap::new();
         let mut set: HashSet<TermID> = HashSet::new();
         set.insert(String::from("apple"));
         set.insert(String::from("banana"));
         map.insert(String::from("apple"), set);

         let mut set: HashSet<TermID> = HashSet::new();
         set.insert(String::from("banana"));
         set.insert(String::from("orange"));
         map.insert(String::from("banana"), set);

         let mut set: HashSet<TermID> = HashSet::new();
         set.insert(String::from("orange"));
         set.insert(String::from("pear"));
         map.insert(String::from("orange"), set);

         let mut set: HashSet<TermID> = HashSet::new();
         set.insert(String::from("pear"));
         set.insert(String::from("kiwi"));
         map.insert(String::from("pear"), set);

         closure_map.insert(String::from("+related_to"), map);
         closure_map
       };

       pub static ref ALL_NO_PRED_MAP: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = {
         let mut closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>> = HashMap::new();
         let _map: HashMap<TermID, HashSet<TermID>> = HashMap::new();

               // closure map looks like this:
         // _all -> apple -> apple, banana
         //             -> banana -> banana, orange
         //             -> orange -> orange, pear
         //             -> pear -> pear, kiwi


         let mut map: HashMap<TermID, HashSet<TermID>> = HashMap::new();
         let mut set: HashSet<TermID> = HashSet::new();
         set.insert(String::from("apple"));
         set.insert(String::from("banana"));
         map.insert(String::from("apple"), set);

         let mut set: HashSet<TermID> = HashSet::new();
         set.insert(String::from("banana"));
         set.insert(String::from("orange"));
         map.insert(String::from("banana"), set);

         let mut set: HashSet<TermID> = HashSet::new();
         set.insert(String::from("orange"));
         set.insert(String::from("pear"));
         map.insert(String::from("orange"), set);

         let mut set: HashSet<TermID> = HashSet::new();
         set.insert(String::from("pear"));
         set.insert(String::from("kiwi"));
         map.insert(String::from("pear"), set);

         closure_map.insert(String::from("_all"), map);
         closure_map
       };

       pub static ref SPO_FRUITS: Vec<(TermID, Predicate, TermID)> = {
                    // turns into closure map like this:
         // +related_to -> apple -> apple, banana
         //             -> banana -> banana, orange
         //             -> orange -> orange, pear
         //             -> pear -> pear, kiwi
         let spo: Vec<(TermID, Predicate, TermID)> = vec![
             (String::from("apple"), String::from("related_to"), String::from("apple")),
             (String::from("apple"), String::from("related_to"), String::from("banana")),
             (String::from("banana"), String::from("related_to"), String::from("banana")),
             (String::from("banana"), String::from("related_to"), String::from("orange")),
             (String::from("orange"), String::from("related_to"), String::from("orange")),
             (String::from("orange"), String::from("related_to"), String::from("pear")),
             (String::from("pear"), String::from("related_to"), String::from("pear")),
             (String::from("pear"), String::from("related_to"), String::from("kiwi")),
         ];

         spo
       };

       pub static ref BFO_SPO: Vec<(TermID, Predicate, TermID)> = {
        let spo:Vec<(TermID, Predicate, TermID)> = vec![
            (
                "BFO:0000035".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000035".to_string(),
            ),
            (
                "BFO:0000016".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000016".to_string(),
            ),
            (
                "BFO:0000034".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000016".to_string(),
            ),
            (
                "BFO:0000144".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000144".to_string(),
            ),
            (
                "BFO:0000031".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000031".to_string(),
            ),
            (
                "BFO:0000011".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000011".to_string(),
            ),
            (
                "BFO:0000026".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000026".to_string(),
            ),
            (
                "BFO:0000144".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000015".to_string(),
            ),
            (
                "BFO:0000182".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000015".to_string(),
            ),
            (
                "BFO:0000015".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000015".to_string(),
            ),
            (
                "BFO:0000142".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000140".to_string(),
            ),
            (
                "BFO:0000146".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000140".to_string(),
            ),
            (
                "BFO:0000140".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000140".to_string(),
            ),
            (
                "BFO:0000147".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000140".to_string(),
            ),
            (
                "BFO:0000020".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000023".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000008".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000019".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000006".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000142".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000029".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000009".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000002".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000141".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000026".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000018".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000017".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000038".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000004".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000182".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000011".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000040".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000144".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000030".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000028".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000031".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000016".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000015".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000034".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000027".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000147".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000148".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000001".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000035".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000140".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000024".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000146".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000145".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000003".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000001".to_string(),
            ),
            (
                "BFO:0000023".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000023".to_string(),
            ),
            (
                "BFO:0000029".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000029".to_string(),
            ),
            (
                "BFO:0000142".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000142".to_string(),
            ),
            (
                "BFO:0000030".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000030".to_string(),
            ),
            (
                "BFO:0000028".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000028".to_string(),
            ),
            (
                "BFO:0000148".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000148".to_string(),
            ),
            (
                "BFO:0000038".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000038".to_string(),
            ),
            (
                "BFO:0000003".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000148".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000008".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000011".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000038".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000144".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000015".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000182".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000035".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000003".to_string(),
            ),
            (
                "BFO:0000182".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000182".to_string(),
            ),
            (
                "BFO:0000018".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000018".to_string(),
            ),
            (
                "BFO:0000024".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000024".to_string(),
            ),
            (
                "BFO:0000147".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000147".to_string(),
            ),
            (
                "BFO:0000017".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000017".to_string(),
            ),
            (
                "BFO:0000023".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000017".to_string(),
            ),
            (
                "BFO:0000016".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000017".to_string(),
            ),
            (
                "BFO:0000034".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000017".to_string(),
            ),
            (
                "BFO:0000024".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000040".to_string(),
            ),
            (
                "BFO:0000040".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000040".to_string(),
            ),
            (
                "BFO:0000027".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000040".to_string(),
            ),
            (
                "BFO:0000030".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000040".to_string(),
            ),
            (
                "BFO:0000027".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000027".to_string(),
            ),
            (
                "BFO:0000034".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000147".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000002".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000009".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000023".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000030".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000142".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000140".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000026".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000141".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000006".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000029".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000018".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000016".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000146".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000027".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000017".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000040".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000004".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000019".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000028".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000145".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000031".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000020".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000024".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000002".to_string(),
            ),
            (
                "BFO:0000018".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000006".to_string(),
            ),
            (
                "BFO:0000026".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000006".to_string(),
            ),
            (
                "BFO:0000009".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000006".to_string(),
            ),
            (
                "BFO:0000006".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000006".to_string(),
            ),
            (
                "BFO:0000028".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000006".to_string(),
            ),
            (
                "BFO:0000019".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000020".to_string(),
            ),
            (
                "BFO:0000145".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000020".to_string(),
            ),
            (
                "BFO:0000020".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000020".to_string(),
            ),
            (
                "BFO:0000017".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000020".to_string(),
            ),
            (
                "BFO:0000016".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000020".to_string(),
            ),
            (
                "BFO:0000023".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000020".to_string(),
            ),
            (
                "BFO:0000034".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000020".to_string(),
            ),
            (
                "BFO:0000034".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000034".to_string(),
            ),
            (
                "BFO:0000009".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000009".to_string(),
            ),
            (
                "BFO:0000146".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000146".to_string(),
            ),
            (
                "BFO:0000019".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000019".to_string(),
            ),
            (
                "BFO:0000145".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000019".to_string(),
            ),
            (
                "BFO:0000040".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000030".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000009".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000147".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000141".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000028".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000004".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000006".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000029".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000018".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000026".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000140".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000146".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000027".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000142".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000024".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000004".to_string(),
            ),
            (
                "BFO:0000140".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000147".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000142".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000141".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000026".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000146".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000018".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000009".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000006".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000028".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000029".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000141".to_string(),
            ),
            (
                "BFO:0000145".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000145".to_string(),
            ),
            (
                "BFO:0000038".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000008".to_string(),
            ),
            (
                "BFO:0000148".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000008".to_string(),
            ),
            (
                "BFO:0000008".to_string(),
                "rdfs:subClassOf".to_string(),
                "BFO:0000008".to_string(),
            ),
        ];
        spo
       };

       pub static ref IC_MAP: HashMap<PredicateSetKey, HashMap<TermID, f64>> = {

         // {
         //   "+subClassOf": {
         //       "CARO:0000000": 2.585,
         //       "BFO:0000002": 1.585,
         //       "BFO:0000003": 1.0
         //   }
         // }

         let mut ic_map: HashMap<PredicateSetKey, HashMap<TermID, f64>> = HashMap::new();
         let mut inner_map: HashMap<TermID, f64> = HashMap::new();

         inner_map.insert(String::from("CARO:0000000"), 2.585);
         inner_map.insert(String::from("BFO:0000002"), 1.585);
         inner_map.insert(String::from("BFO:0000003"), 1.0);

         ic_map.insert(String::from("+subClassOf"), inner_map);
         ic_map
     };

       pub static ref MAP: HashMap<String, HashMap<String, f64>> = {

       //   {
       //     "CARO:0000000": {
       //         "CARO:0000000": 5.0,
       //         "BFO:0000002": 4.0,
       //         "BFO:0000003": 3.0
       //     },
       //     "BFO:0000002": {
       //         "CARO:0000000": 2.0,
       //         "BFO:0000002": 4.0,
       //         "BFO:0000003": 3.0
       //     },
       //     "BFO:0000003": {
       //         "CARO:0000000": 1.0,
       //         "BFO:0000002": 3.0,
       //         "BFO:0000003": 4.0
       //     }
       // }

         let mut map = HashMap::new();

         let inner_map1: HashMap<String, f64> = [
             (String::from("CARO:0000000"), 5.0),
             (String::from("BFO:0000002"), 4.0),
             (String::from("BFO:0000003"), 3.0),
         ].iter().cloned().collect();
         map.insert(String::from("CARO:0000000"), inner_map1);

         let inner_map2: HashMap<String, f64> = [
             (String::from("CARO:0000000"), 2.0),
             (String::from("BFO:0000002"), 4.0),
             (String::from("BFO:0000003"), 3.0),
         ].iter().cloned().collect();
         map.insert(String::from("BFO:0000002"), inner_map2);

         let inner_map3: HashMap<String, f64> = [
             (String::from("CARO:0000000"), 1.0),
             (String::from("BFO:0000002"), 3.0),
             (String::from("BFO:0000003"), 4.0),
         ].iter().cloned().collect();
         map.insert(String::from("BFO:0000003"), inner_map3);

         map
       };
    }
}
