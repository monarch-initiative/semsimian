use db_query::get_entailed_edges_for_predicate_list;
use pyo3::prelude::*;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    sync::{Arc, Mutex, RwLock},
};

pub mod db_query;
pub mod similarity;
pub mod termset_pairwise_similarity;
pub mod utils;

use rayon::prelude::*;

mod test_utils;

use std::fmt;

use similarity::{
    calculate_average_termset_information_content, calculate_cosine_similarity_for_nodes,
    calculate_max_information_content,
};
use utils::{
    convert_list_of_tuples_to_hashmap, expand_term_using_closure,
    generate_progress_bar_of_length_and_message, get_best_matches, get_termset_vector,
    predicate_set_to_key, rearrange_columns_and_rewrite,
};

use db_query::get_labels;
use lazy_static::lazy_static;
use termset_pairwise_similarity::TermsetPairwiseSimilarity;

use crate::utils::get_best_score;

// change to "pub" because it is easier for testing
pub type Predicate = String;
pub type TermID = String;
pub type PredicateSetKey = String;
pub type Jaccard = f64;
pub type Resnik = f64;
pub type Phenodigm = f64;
pub type Cosine = f64;
pub type MostInformativeAncestors = HashSet<TermID>;
type SimilarityMap = HashMap<
    TermID,
    HashMap<TermID, (Jaccard, Resnik, Phenodigm, Cosine, MostInformativeAncestors)>,
>;
type Embeddings = Vec<(String, Vec<f64>)>;

lazy_static! {
    static ref RESOURCE_PATH: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
}

#[derive(Clone)]
pub struct RustSemsimian {
    spo: Vec<(TermID, Predicate, TermID)>,
    predicates: Option<Vec<Predicate>>,
    ic_map: HashMap<PredicateSetKey, HashMap<TermID, f64>>,
    // ic_map is something like {("is_a_+_part_of"), {"GO:1234": 1.234}}
    closure_map: HashMap<PredicateSetKey, HashMap<TermID, HashSet<TermID>>>,
    // closure_map is something like {("is_a_+_part_of"), {"GO:1234": {"GO:1234", "GO:5678"}}}
    embeddings: Embeddings,
    pairwise_similarity_attributes: Option<Vec<String>>,
}

impl RustSemsimian {
    // TODO: this is tied directly to Oak, and should be made more generic
    // TODO: also, we should support loading "custom" ic

    pub fn new(
        spo: Option<Vec<(TermID, Predicate, TermID)>>,
        predicates: Option<Vec<Predicate>>,
        pairwise_similarity_attributes: Option<Vec<String>>,
        resource_path: Option<&str>,
    ) -> RustSemsimian {
        if spo.is_none() && resource_path.is_none() {
            panic!("If no `spo` is provided, `resource_path` is required.");
        }
        if let Some(resource_path) = resource_path {
            *RESOURCE_PATH.lock().unwrap() = Some(resource_path.to_string().clone());
        }
        let spo = match spo {
            Some(spo) => spo,
            None => {
                match get_entailed_edges_for_predicate_list(
                    resource_path.unwrap(),
                    predicates.as_ref().unwrap_or(&Vec::new()),
                ) {
                    Ok(edges) => edges,
                    Err(err) => panic!("Resource returned nothing with predicates: {}", err),
                }
            }
        };

        let predicates = match predicates {
            Some(predicates) => Some(predicates),
            None => {
                let mut unique_predicates = HashSet::new();
                for (_, predicate, _) in &spo {
                    unique_predicates.insert(predicate.to_owned());
                }
                Some(unique_predicates.into_iter().collect())
            }
        };

        RustSemsimian {
            spo,
            predicates,
            pairwise_similarity_attributes,
            ic_map: HashMap::new(),
            closure_map: HashMap::new(),
            embeddings: Vec::new(),
        }
    }

    pub fn update_closure_and_ic_map(&mut self) {
        let predicate_set_key = predicate_set_to_key(&self.predicates);
        let (this_closure_map, this_ic_map) =
            convert_list_of_tuples_to_hashmap(&self.spo, &self.predicates);
        self.closure_map.insert(
            predicate_set_key.clone(),
            this_closure_map.get(&predicate_set_key).unwrap().clone(),
        );
        self.ic_map.insert(
            predicate_set_key.clone(),
            this_ic_map.get(&predicate_set_key).unwrap().clone(),
        );
    }

    pub fn load_embeddings(&mut self, embeddings_file: &str) {
        if let Ok(file) = File::open(embeddings_file) {
            let reader = BufReader::new(file);

            let mut embeddings: Vec<(String, Vec<f64>)> = Vec::new();
            let mut lines = reader.lines();

            // Skip the header row
            lines.next();

            for line in lines.flatten() {
                let values: Vec<&str> = line.split('\t').collect();
                let curie = values[0].to_string();
                let embedding: Vec<f64> = values[1..]
                    .iter()
                    .filter_map(|value| value.parse().ok())
                    .collect();
                embeddings.push((curie, embedding));
            }

            self.embeddings = embeddings;
        }
    }

    pub fn jaccard_similarity(&self, term1: &str, term2: &str) -> f64 {
        let apple_set = expand_term_using_closure(term1, &self.closure_map, &self.predicates);
        let fruit_set = expand_term_using_closure(term2, &self.closure_map, &self.predicates);

        let intersection = apple_set.intersection(&fruit_set).count() as f64;
        let union = apple_set.union(&fruit_set).count() as f64;
        intersection / union
    }

    pub fn resnik_similarity(&self, term1: &str, term2: &str) -> (HashSet<String>, f64) {
        calculate_max_information_content(
            &self.closure_map,
            &self.ic_map,
            term1,
            term2,
            &self.predicates,
        )
    }

    pub fn cosine_similarity(&self, term1: &str, term2: &str, embeddings: &Embeddings) -> f64 {
        calculate_cosine_similarity_for_nodes(embeddings, term1, term2).unwrap()
    }

    pub fn all_by_all_pairwise_similarity(
        &self,
        subject_terms: &HashSet<TermID>,
        object_terms: &HashSet<TermID>,
        minimum_jaccard_threshold: &Option<f64>,
        minimum_resnik_threshold: &Option<f64>,
    ) -> SimilarityMap {
        let self_shared = Arc::new(RwLock::new(self.clone()));
        let pb = generate_progress_bar_of_length_and_message(
            (subject_terms.len() * object_terms.len()) as u64,
            "Building all X all pairwise similarity:",
        );

        let similarity_map: SimilarityMap = subject_terms
            .par_iter() // parallelize computations
            .map(|subject| {
                let mut subject_similarities: HashMap<
                    TermID,
                    (Jaccard, Resnik, Phenodigm, Cosine, MostInformativeAncestors),
                > = HashMap::new();
                for object in object_terms.iter() {
                    let self_read = self_shared.read().unwrap();
                    let jaccard_similarity = self_read.jaccard_similarity(subject, object);
                    let (ancestor_id, ancestor_information_content) =
                        self_read.resnik_similarity(subject, object);
                    let cosine_similarity = match !self_read.embeddings.is_empty() {
                        true => self_read.cosine_similarity(subject, object, &self_read.embeddings),
                        false => std::f64::NAN,
                    };

                    if minimum_jaccard_threshold.map_or(true, |t| jaccard_similarity > t)
                        && minimum_resnik_threshold
                            .map_or(true, |t| ancestor_information_content > t)
                    {
                        subject_similarities.insert(
                            object.clone(),
                            (
                                jaccard_similarity,
                                ancestor_information_content,
                                (ancestor_information_content * jaccard_similarity).sqrt(),
                                cosine_similarity,
                                ancestor_id,
                            ),
                        );
                    }

                    pb.inc(1);
                }
                (subject.clone(), subject_similarities)
            })
            .collect();
        pb.finish_with_message("done");
        similarity_map
    }

    pub fn all_by_all_pairwise_similarity_with_output(
        &self,
        subject_terms: &HashSet<TermID>,
        object_terms: &HashSet<TermID>,
        minimum_jaccard_threshold: &Option<f64>,
        minimum_resnik_threshold: &Option<f64>,
        outfile: &Option<&str>,
    ) {
        let self_shared = Arc::new(RwLock::new(self.clone()));
        let pb = generate_progress_bar_of_length_and_message(
            (subject_terms.len() * object_terms.len()) as u64,
            "Building all X all pairwise similarity:",
        );
        let outfile = outfile.unwrap_or("similarity_map.tsv");
        let file = File::create(outfile).unwrap();
        let writer = Arc::new(Mutex::new(BufWriter::new(file)));
        let mut column_names: Vec<String> = vec![
            "subject_id".to_string(),
            "object_id".to_string(),
            "jaccard_similarity".to_string(),
            "ancestor_information_content".to_string(),
            "phenodigm_score".to_string(),
            "cosine_similarity".to_string(),
            "ancestor_id".to_string(),
        ];
        column_names.sort();
        let sorted_attributes = match &self.pairwise_similarity_attributes {
            Some(attributes) => {
                let mut cloned_attributes = attributes.clone();
                cloned_attributes.sort();
                Some(cloned_attributes)
            }
            None => None,
        };

        let output_columns_vector = sorted_attributes.as_ref().unwrap_or(&column_names).clone();

        let column_names_as_str = output_columns_vector.join("\t") + "\n";

        // Write the column names to the TSV file
        let mut writer_1 = writer.lock().unwrap();
        writer_1.write_all(column_names_as_str.as_bytes()).unwrap();
        drop(writer_1);

        subject_terms
            .par_iter() // parallelize computations
            .for_each(|subject_id| {
                for object_id in object_terms.iter() {
                    let self_read = self_shared.read().unwrap();
                    let jaccard_similarity = self_read.jaccard_similarity(subject_id, object_id);
                    let (ancestor_id, ancestor_information_content) =
                        self_read.resnik_similarity(subject_id, object_id);
                    let cosine_similarity = match !self_read.embeddings.is_empty() {
                        true => self_read.cosine_similarity(
                            subject_id,
                            object_id,
                            &self_read.embeddings,
                        ),
                        false => std::f64::NAN,
                    };

                    let mut output_map: BTreeMap<&str, String> = BTreeMap::new();

                    for name in &output_columns_vector {
                        output_map.insert(name, "".to_string());
                    }

                    // Overwrite output_map values with variable values that correspond to the keys if they exist
                    if let Some(value) = output_map.get_mut("subject_id") {
                        *value = subject_id.to_string();
                    }
                    if let Some(value) = output_map.get_mut("object_id") {
                        *value = object_id.to_string();
                    }
                    if let Some(value) = output_map.get_mut("jaccard_similarity") {
                        *value = jaccard_similarity.to_string();
                    }
                    if let Some(value) = output_map.get_mut("ancestor_information_content") {
                        *value = ancestor_information_content.to_string();
                    }
                    if let Some(value) = output_map.get_mut("phenodigm_score") {
                        *value = (ancestor_information_content * jaccard_similarity)
                            .sqrt()
                            .to_string();
                    }
                    if let Some(value) = output_map.get_mut("cosine_similarity") {
                        *value = cosine_similarity.to_string();
                    }
                    if let Some(value) = output_map.get_mut("ancestor_id") {
                        *value = ancestor_id.into_iter().collect::<Vec<String>>().join(", ");
                    }

                    if minimum_jaccard_threshold.map_or(true, |t| jaccard_similarity > t)
                        && minimum_resnik_threshold
                            .map_or(true, |t| ancestor_information_content > t)
                    {
                        // Write the line to the TSV file
                        let mut output_bytes: Vec<u8> = output_map
                            .values()
                            .map(|value| {
                                let s = value;
                                s.to_string()
                            })
                            .collect::<Vec<String>>()
                            .join("\t")
                            .as_bytes()
                            .to_vec();

                        output_bytes.extend_from_slice(b"\n");
                        let mut writer_2 = writer.lock().unwrap();
                        writer_2.write_all(&output_bytes).unwrap();
                    }

                    pb.inc(1);
                }
            });
        drop(writer);
        if let Some(output_columns_vector) = &self.pairwise_similarity_attributes {
            let _ = rearrange_columns_and_rewrite(outfile, output_columns_vector.to_owned());
        } else {
            let _ = rearrange_columns_and_rewrite(
                outfile,
                column_names.iter().map(|s| s.to_owned()).collect(),
            );
        }
        pb.finish_with_message("done");
    }

    pub fn termset_comparison(
        &self,
        subject_terms: &HashSet<TermID>,
        object_terms: &HashSet<TermID>,
    ) -> Result<f64, String> {
        Ok(calculate_average_termset_information_content(
            self,
            subject_terms,
            object_terms,
        ))
    }

    pub fn termset_pairwise_similarity(
        &self,
        subject_terms: &HashSet<TermID>,
        object_terms: &HashSet<TermID>,
        _outfile: &Option<&str>,
    ) -> TermsetPairwiseSimilarity {
        let metric = "ancestor_information_content";
        let all_by_all: SimilarityMap =
            self.all_by_all_pairwise_similarity(subject_terms, object_terms, &None, &None);

        let mut all_by_all_object_perspective: SimilarityMap = HashMap::new();

        for (key1, value1) in all_by_all.iter() {
            for (key2, value2) in value1.iter() {
                if !all_by_all_object_perspective.contains_key(key2.as_str()) {
                    all_by_all_object_perspective.insert(key2.to_owned(), HashMap::new());
                }
                all_by_all_object_perspective
                    .get_mut(key2.as_str())
                    .unwrap()
                    .insert(key1.to_owned(), value2.to_owned());
            }
        }
        let db_path = RESOURCE_PATH.lock().unwrap();
        let all_terms = subject_terms
            .union(object_terms)
            .cloned()
            .collect::<Vec<String>>();
        let term_label_map = get_labels(db_path.clone().unwrap().as_str(), all_terms).unwrap();

        let subject_termset: Vec<BTreeMap<String, BTreeMap<String, String>>> =
            get_termset_vector(subject_terms, &term_label_map);
        let object_termset: Vec<BTreeMap<String, BTreeMap<String, String>>> =
            get_termset_vector(object_terms, &term_label_map);
        let average_termset_information_content = &self
            .termset_comparison(subject_terms, object_terms)
            .unwrap();

        let (subject_best_matches, subject_best_matches_similarity_map) =
            get_best_matches(&subject_termset, &all_by_all, &term_label_map, metric);
        let (object_best_matches, object_best_matches_similarity_map) = get_best_matches(
            &object_termset,
            &all_by_all_object_perspective,
            &term_label_map,
            metric,
        );
        let best_score = get_best_score(&subject_best_matches, &object_best_matches);

        TermsetPairwiseSimilarity::new(
            subject_termset,
            subject_best_matches,
            subject_best_matches_similarity_map,
            object_termset,
            object_best_matches,
            object_best_matches_similarity_map,
            *average_termset_information_content,
            best_score,
            metric.to_string(),
        )
    }
}

#[pyclass]
pub struct Semsimian {
    ss: RustSemsimian,
}

#[pymethods]
impl Semsimian {
    #[new]
    fn new(
        spo: Option<Vec<(TermID, Predicate, TermID)>>,
        predicates: Option<Vec<String>>,
        pairwise_similarity_attributes: Option<Vec<String>>,
        resource_path: Option<&str>,
    ) -> PyResult<Self> {
        //Check if OS is Windows and if so do this.
        #[cfg(target_os = "windows")]
        let processed_path = resource_path.map(|path| {
            let path_buf = std::path::PathBuf::from(path);
            let drive = path_buf.parent().unwrap().to_str().unwrap();
            let remaining_path = path_buf.file_stem().unwrap().to_str().unwrap();
            if drive.is_empty() {
            format!("{}", remaining_path)
            } else {
                path.to_string()
            }
        });
        #[cfg(target_os = "windows")]
        let resource_path = processed_path.as_ref().map(|s| s.as_str());

        let ss = RustSemsimian::new(
            spo,
            predicates,
            pairwise_similarity_attributes,
            resource_path,
        );
        Ok(Semsimian { ss })
    }

    fn jaccard_similarity(&mut self, term1: TermID, term2: TermID) -> PyResult<f64> {
        self.ss.update_closure_and_ic_map();
        Ok(self.ss.jaccard_similarity(&term1, &term2))
    }

    fn cosine_similarity(
        &mut self,
        term1: TermID,
        term2: TermID,
        embeddings_file: &str,
    ) -> PyResult<f64> {
        self.ss.load_embeddings(embeddings_file);
        Ok(self
            .ss
            .cosine_similarity(&term1, &term2, &self.ss.embeddings))
    }

    fn resnik_similarity(
        &mut self,
        term1: TermID,
        term2: TermID,
    ) -> PyResult<(HashSet<String>, f64)> {
        self.ss.update_closure_and_ic_map();
        Ok(self.ss.resnik_similarity(&term1, &term2))
    }

    fn all_by_all_pairwise_similarity(
        &mut self,
        subject_terms: HashSet<TermID>,
        object_terms: HashSet<TermID>,
        minimum_jaccard_threshold: Option<f64>,
        minimum_resnik_threshold: Option<f64>,
    ) -> SimilarityMap {
        // first make sure we have the closure and ic map for the given predicates
        self.ss.update_closure_and_ic_map();

        self.ss.all_by_all_pairwise_similarity(
            &subject_terms,
            &object_terms,
            &minimum_jaccard_threshold,
            &minimum_resnik_threshold,
        )
    }

    fn all_by_all_pairwise_similarity_quick(
        &mut self,
        subject_terms: HashSet<TermID>,
        object_terms: HashSet<TermID>,
        minimum_jaccard_threshold: Option<f64>,
        minimum_resnik_threshold: Option<f64>,
        embeddings_file: Option<&str>,
        outfile: Option<&str>,
    ) -> PyResult<()> {
        // first make sure we have the closure and ic map for the given predicates
        self.ss.update_closure_and_ic_map();
        if let Some(file) = embeddings_file {
            self.ss.load_embeddings(file);
        }

        self.ss.all_by_all_pairwise_similarity_with_output(
            &subject_terms,
            &object_terms,
            &minimum_jaccard_threshold,
            &minimum_resnik_threshold,
            &outfile,
        );
        Ok(())
    }

    fn termset_pairwise_similarity(
        &mut self,
        subject_terms: HashSet<TermID>,
        object_terms: HashSet<TermID>,
        outfile: Option<&str>,
        py: Python,
    ) -> PyResult<PyObject> {
        self.ss.update_closure_and_ic_map();

        let tsps = self
            .ss
            .termset_pairwise_similarity(&subject_terms, &object_terms, &outfile);
        Ok(tsps.into_py(py))
    }

    fn get_spo(&self) -> PyResult<Vec<(TermID, Predicate, TermID)>> {
        Ok(self.ss.spo.to_vec())
    }

    fn termset_comparison(
        &mut self,
        subject_terms: HashSet<TermID>,
        object_terms: HashSet<TermID>,
    ) -> PyResult<f64> {
        self.ss.update_closure_and_ic_map();

        match self.ss.termset_comparison(&subject_terms, &object_terms) {
            Ok(score) => Ok(score),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(err)),
        }
    }
}

impl fmt::Debug for RustSemsimian {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RustSemsimian {{ spo: {:?}, ic_map: {:?}, closure_map: {:?} }}",
            self.spo, self.ic_map, self.closure_map
        )
    }
}

#[pymodule]
fn semsimian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Semsimian>()?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::test_utils::test_constants::BFO_SPO;
    use crate::{test_utils::test_constants::SPO_FRUITS, RustSemsimian};
    use std::{
        collections::HashSet,
        io::{BufRead, BufReader},
    };

    #[test]
    fn test_object_creation() {
        let spo_cloned = Some(SPO_FRUITS.clone());
        let predicates: Option<Vec<Predicate>> = Some(vec!["related_to".to_string()]);
        let ss = RustSemsimian::new(spo_cloned, None, None, None);
        assert_eq!(predicates, ss.predicates);
    }

    #[test]
    fn test_object_creation_using_resource() {
        let predicates: Option<Vec<Predicate>> = Some(vec!["rdfs:subClassOf".to_string()]);
        let db = Some("tests/data/go-nucleus.db");
        let expected_length: usize = 1302;
        let ss = RustSemsimian::new(None, predicates, None, db);
        // dbg!(ss.spo.len());
        assert_eq!(ss.spo.len(), expected_length)
    }

    #[test]
    fn test_jaccard_similarity() {
        let spo_cloned = Some(SPO_FRUITS.clone());
        let predicates: Option<Vec<Predicate>> = Some(vec!["related_to".to_string()]);
        let no_predicates: Option<Vec<Predicate>> = None;
        let mut ss_with_predicates = RustSemsimian::new(spo_cloned.clone(), predicates, None, None);
        let mut ss_without_predicates = RustSemsimian::new(spo_cloned, no_predicates, None, None);
        ss_without_predicates.update_closure_and_ic_map();
        ss_with_predicates.update_closure_and_ic_map();
        println!(
            "Closure table for ss without predicates  {:?}",
            ss_without_predicates.closure_map
        );
        println!(
            "Closure table for ss with predicates {:?}",
            ss_with_predicates.closure_map
        );
        //Closure table: {"+related_to": {"apple": {"banana", "apple"}, "banana": {"orange", "banana"}, "pear": {"kiwi", "pear"}, "orange": {"orange", "pear"}}}
        let apple = "apple".to_string();
        let banana = "banana".to_string();
        let sim = ss_with_predicates.jaccard_similarity(&apple, &banana);
        let sim2 = ss_without_predicates.jaccard_similarity(&apple, &banana);

        assert_eq!(sim, 1.0 / 3.0);
        assert_eq!(sim2, 1.0 / 3.0);
    }

    #[test]
    fn test_get_closure_and_ic_map() {
        let spo_cloned = Some(SPO_FRUITS.clone());
        let test_predicates: Option<Vec<Predicate>> = Some(vec!["related_to".to_string()]);
        let mut semsimian = RustSemsimian::new(spo_cloned, test_predicates, None, None);
        println!("semsimian after initialization: {semsimian:?}");
        semsimian.update_closure_and_ic_map();
        assert!(!semsimian.closure_map.is_empty());
        assert!(!semsimian.ic_map.is_empty());
    }

    #[test]
    fn test_resnik_similarity() {
        let spo_cloned = Some(SPO_FRUITS.clone());
        let predicates: Option<Vec<String>> = Some(vec!["related_to".to_string()]);
        let mut rs = RustSemsimian::new(spo_cloned, predicates.clone(), None, None);
        rs.update_closure_and_ic_map();
        println!("Closure_map from semsimian {:?}", rs.closure_map);
        let (_, sim) = rs.resnik_similarity("apple", "banana");
        println!("DO THE print{sim}");
        assert_eq!(sim, 1.3219280948873622);
    }

    #[test]
    fn test_all_by_all_pairwise_similarity_with_empty_inputs() {
        let rss = RustSemsimian::new(
            Some(vec![(
                "apple".to_string(),
                "is_a".to_string(),
                "fruit".to_string(),
            )]),
            None,
            None,
            None,
        );

        let subject_terms: HashSet<TermID> = HashSet::new();
        let object_terms: HashSet<TermID> = HashSet::new();

        let result = rss.all_by_all_pairwise_similarity(
            &subject_terms,
            &object_terms,
            &Some(0.0),
            &Some(0.0),
        );

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_all_by_all_pairwise_similarity_with_nonempty_inputs() {
        let mut rss = RustSemsimian::new(
            Some(vec![
                ("apple".to_string(), "is_a".to_string(), "fruit".to_string()),
                ("apple".to_string(), "is_a".to_string(), "food".to_string()),
                ("apple".to_string(), "is_a".to_string(), "item".to_string()),
                ("fruit".to_string(), "is_a".to_string(), "food".to_string()),
                ("fruit".to_string(), "is_a".to_string(), "item".to_string()),
                ("food".to_string(), "is_a".to_string(), "item".to_string()),
            ]),
            Some(vec!["is_a".to_string()]),
            None,
            None,
        );

        let apple = "apple".to_string();
        let fruit = "fruit".to_string();
        let food = "food".to_string();

        let mut subject_terms: HashSet<String> = HashSet::new();
        subject_terms.insert(apple.clone());
        subject_terms.insert(fruit.clone());

        let mut object_terms: HashSet<TermID> = HashSet::new();
        object_terms.insert(fruit.clone());
        object_terms.insert(food.clone());

        rss.update_closure_and_ic_map();
        let result = rss.all_by_all_pairwise_similarity(
            &subject_terms,
            &object_terms,
            &Some(0.0),
            &Some(0.0),
        );

        assert_eq!(result.len(), 2);
        assert!(result.contains_key(&apple));
        // assert!(result.contains_key(&fruit));

        // Apple
        let apple_similarities = result.get(&apple).unwrap();
        // println!("{apple_similarities:?}");
        assert_eq!(apple_similarities.len(), 1);
        assert!(apple_similarities.contains_key(&fruit));
        assert!(!apple_similarities.contains_key(&food)); // Since resnik <= threshold

        // Apple, fruit tests
        let apple_fruit_jaccard = rss.jaccard_similarity(&apple, &fruit);
        let (apple_fruit_mica, apple_fruit_resnik) = rss.resnik_similarity(&apple, &fruit);
        let (
            apple_fruit_jaccard_from_similarity,
            apple_fruit_resnik_from_similarity,
            apple_fruit_phenodigm_from_similarity,
            _, // Cosine similarity
            apple_fruit_mica_from_similarity,
        ) = apple_similarities.get(&fruit).unwrap();

        assert_eq!(*apple_fruit_resnik_from_similarity, apple_fruit_resnik);
        assert_eq!(*apple_fruit_jaccard_from_similarity, apple_fruit_jaccard);
        assert_eq!(
            *apple_fruit_phenodigm_from_similarity,
            (apple_fruit_jaccard * apple_fruit_resnik).sqrt()
        );
        // println!("{apple_similarities:?}");
        // println!("{apple_fruit_mica:?}");

        assert_eq!(*apple_fruit_mica_from_similarity, apple_fruit_mica);

        //Apple, food tests
        let apple_food_jaccard = rss.jaccard_similarity(&apple, &food);
        let (apple_food_mcra, apple_food_resnik) = rss.resnik_similarity(&apple, &food);

        assert_eq!(0.0, apple_food_resnik);
        assert_eq!(0.3333333333333333, apple_food_jaccard);
        assert_eq!(HashSet::from(["item".to_string()]), apple_food_mcra);

        // Fruit
        let fruit_similarities = result.get(&fruit).unwrap();
        let fruit_fruit_jaccard = rss.jaccard_similarity(&fruit, &fruit);
        let (fruit_fruit_mica, fruit_fruit_resnik) = rss.resnik_similarity(&fruit, &fruit);
        let (
            fruit_fruit_jaccard_from_similarity,
            fruit_fruit_resnik_from_similarity,
            fruit_fruit_phenodigm_from_similarity,
            _, // cosine similarity
            fruit_fruit_mica_from_similarity,
        ) = fruit_similarities.get(&fruit).unwrap();

        // println!("{fruit_similarities:?}");
        // println!("{fruit_fruit_mica:?}");

        assert_eq!(fruit_similarities.len(), 1);
        assert!(fruit_similarities.contains_key(&fruit));
        assert!(!fruit_similarities.contains_key(&food)); // Since Resnik <= threshold

        // Fruit, fruit tests
        assert_eq!(*fruit_fruit_resnik_from_similarity, fruit_fruit_resnik);
        assert_eq!(*fruit_fruit_jaccard_from_similarity, fruit_fruit_jaccard);
        assert_eq!(
            *fruit_fruit_phenodigm_from_similarity,
            (fruit_fruit_resnik * fruit_fruit_jaccard).sqrt()
        );
        assert_eq!(*fruit_fruit_mica_from_similarity, fruit_fruit_mica);

        // Fruit, food tests
        let fruit_food_jaccard = rss.jaccard_similarity(&fruit, &food);
        let (fruit_food_mica, fruit_food_resnik) = rss.resnik_similarity(&fruit, &food);
        assert_eq!(0.0, fruit_food_resnik);
        assert_eq!(0.5, fruit_food_jaccard);
        assert_eq!(HashSet::from(["item".to_string()]), fruit_food_mica);
        assert!(!result.contains_key(&food)); // Since Resnik <= threshold
        println!("all_by_all_pairwise_similarity result: {result:?}");
    }

    #[test]
    fn test_all_by_all_pairwise_similarity_with_output() {
        let output_columns = crate::test_utils::test_constants::OUTPUT_COLUMNS_VECTOR.clone();
        let mut rss = RustSemsimian::new(
            Some(SPO_FRUITS.clone()),
            Some(vec!["related_to".to_string()]),
            Some(output_columns),
            None,
        );
        let banana = "banana".to_string();
        let apple = "apple".to_string();
        let pear = "pear".to_string();
        let outfile = Some("tests/data/output/pairwise_similarity_test_output.tsv");
        let embeddings_file = Some("tests/data/test_embeddings.tsv");

        let mut subject_terms: HashSet<String> = HashSet::new();
        subject_terms.insert(banana);
        subject_terms.insert(apple.clone());

        let mut object_terms: HashSet<TermID> = HashSet::new();
        object_terms.insert(apple);
        object_terms.insert(pear);

        rss.update_closure_and_ic_map();
        rss.load_embeddings(embeddings_file.unwrap());
        rss.all_by_all_pairwise_similarity_with_output(
            &subject_terms,
            &object_terms,
            &Some(0.0),
            &Some(0.0),
            &outfile,
        );

        // Read the outfile and count the number of lines
        let file = File::open(outfile.unwrap()).unwrap();
        let reader = BufReader::new(file);

        let line_count = reader.lines().count();
        // Assert that the line count is 3 (including the header)
        assert_eq!(line_count, 3);

        // Clean up the temporary file
        std::fs::remove_file(outfile.unwrap()).expect("Failed to remove file");
    }

    #[test]
    fn test_resnik_using_bfo() {
        let spo = Some(BFO_SPO.clone());

        let predicates: Option<Vec<Predicate>> = Some(vec![
            "rdfs:subClassOf".to_string(),
            "BFO:0000050".to_string(),
        ]);

        let mut rss = RustSemsimian::new(spo, predicates, None, None);

        rss.update_closure_and_ic_map();
        // println!("IC_map from semsimian {:?}", rss.ic_map);
        let (_, sim) = rss.resnik_similarity("BFO:0000040", "BFO:0000002");
        println!("DO THE print {sim}");
        assert_eq!(sim, 0.4854268271702417);
    }

    #[test]
    fn test_cosine_using_bfo() {
        let spo = Some(BFO_SPO.clone());
        let mut rss = RustSemsimian::new(spo, None, None, None);
        let embeddings_file = Some("tests/data/bfo_embeddings.tsv");

        rss.load_embeddings(embeddings_file.unwrap());

        let cosine_similarity =
            rss.cosine_similarity("BFO:0000040", "BFO:0000002", &rss.embeddings);
        println!("DO THE print {cosine_similarity}");
        assert_eq!(cosine_similarity, 0.09582515104047208);
    }

    #[test]
    fn test_termset_pairwise_similarity() {
        let db = Some("tests/data/go-nucleus.db");
        // Call the function with the test parameters
        let predicates: Option<Vec<Predicate>> = Some(vec![
            "rdfs:subClassOf".to_string(),
            "BFO:0000050".to_string(),
        ]);
        let subject_terms = HashSet::from(["GO:0005634".to_string(), "GO:0016020".to_string()]);
        let object_terms = HashSet::from(["GO:0031965".to_string(), "GO:0005773".to_string()]);
        let outfile = Some("tests/data/output/termset_similarity_output.tsv");
        let mut rss = RustSemsimian::new(None, predicates, None, db);
        rss.update_closure_and_ic_map();
        let tsps = rss.termset_pairwise_similarity(&subject_terms, &object_terms, &outfile);
        assert_eq!(tsps.average_score, 5.4154243283740175);
        assert_eq!(tsps.best_score, 5.8496657269155685);
    }

    #[test]
    fn test_termset_comparison() {
        let spo = Some(BFO_SPO.clone());
        let predicates: Option<Vec<Predicate>> = Some(vec![
            "rdfs:subClassOf".to_string(),
            "BFO:0000050".to_string(),
        ]);
        let mut rss = RustSemsimian::new(spo, predicates, None, None);

        rss.update_closure_and_ic_map();

        let entity1: HashSet<TermID> =
            HashSet::from(["BFO:0000020".to_string(), "BFO:0000002".to_string()]);
        let entity2: HashSet<TermID> =
            HashSet::from(["BFO:0000030".to_string(), "BFO:0000005".to_string()]);

        let result = rss.termset_comparison(&entity1, &entity2); //Result<f64, String>
        let expected_result = 0.36407012037768127;

        assert_eq!(result.unwrap(), expected_result);
    }

    #[test]
    fn test_termset_comparison_with_db() {
        let db = Some("tests/data/go-nucleus.db");
        let predicates: Option<Vec<Predicate>> = Some(vec![
            "rdfs:subClassOf".to_string(),
            "BFO:0000050".to_string(),
        ]);
        let mut rss = RustSemsimian::new(None, predicates, None, db);

        rss.update_closure_and_ic_map();

        let entity1: HashSet<TermID> =
            HashSet::from(["GO:0005634".to_string(), "GO:0016020".to_string()]);
        let entity2: HashSet<TermID> =
            HashSet::from(["GO:0031965".to_string(), "GO:0005773".to_string()]);

        let result = rss.termset_comparison(&entity1, &entity2); //Result<f64, String>
        let expected_result = 5.4154243283740175;
        assert_eq!(result.unwrap(), expected_result);
    }
}
