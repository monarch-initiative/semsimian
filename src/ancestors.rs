use generator::{done, Generator, Gn};
use std::collections::HashSet;

pub fn get_intersection_between_sets<'a>(
    set1: &'a HashSet<String>,
    set2: &'a HashSet<String>,
) -> Generator<'a, (), &'a String> {
    /* Yields intersection between the two sets. e.g. subject_ancestors and object_ancestors. */
    Gn::new_scoped(move |mut s| {
        for element in set1.intersection(&set2) {
            s.yield_(element);
        }
        done!();
    })
}

#[cfg(test)]

mod tests {
    use super::*;
    #[test]
    fn test_get_common_ancestors() {
        let set1: HashSet<String> = HashSet::from([String::from("apple"), String::from("banana")]);
        let set2: HashSet<String> = HashSet::from([
            String::from("apple"),
            String::from("banana"),
            String::from("mango"),
            String::from("grapes"),
        ]);

        for a in get_intersection_between_sets(&set1, &set2).into_iter() {
            // println!("{:?}",a);
            assert!(set1.intersection(&set2).into_iter().any(|x| x == a));
        }
    }
}
