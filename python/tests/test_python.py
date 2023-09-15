import os
import sys
import time
import unittest
from semsimian import Semsimian
from pathlib import Path


class testSemsimianWithPython(unittest.TestCase):
    def setUp(self):
        spo = [
            ("apple", "related_to", "apple"),
            ("apple", "related_to", "banana"),
            ("banana", "related_to", "banana"),
            ("banana", "related_to", "orange"),
            ("orange", "related_to", "orange"),
            ("orange", "related_to", "pear"),
            ("pear", "related_to", "pear"),
            ("pear", "related_to", "kiwi"),
        ]
        predicates = ["related_to"]

        self.semsimian = Semsimian(spo, predicates)
        self.db = str(Path(__file__).parents[2] / "tests/data/go-nucleus.db")

    def test_jaccard_similarity(self):
        term1 = "apple"
        term2 = "banana"
        result = self.semsimian.jaccard_similarity(term1, term2)
        self.assertEqual(result, 1.0 / 3.0)

    def test_resnik_similarity(self):
        term1 = "apple"
        term2 = "banana"
        result = self.semsimian.resnik_similarity(term1, term2)
        self.assertEqual(result, ({"banana"}, 1.3219280948873622))

    def test_all_by_all_pairwise_similarity(self):
        subject_terms = {"apple", "banana", "orange"}
        object_terms = {"orange", "pear", "kiwi"}
        orange_mica = {"orange", "pear"}
        result = self.semsimian.all_by_all_pairwise_similarity(
            subject_terms, object_terms, 0.0, 0.0
        )
        self.assertEqual(result["orange"]["orange"][4], orange_mica)
        result2 = self.semsimian.all_by_all_pairwise_similarity(
            subject_terms=subject_terms,
            object_terms=object_terms,
        )
        self.assertEqual(result2["orange"]["orange"][4], orange_mica)

    def test_termset_comparison(self):
        subject_terms = {"apple", "banana", "orange"}
        object_terms = {"orange", "pear", "kiwi"}
        expected_score = 0.8812853965915748
        score = self.semsimian.termset_comparison(subject_terms, object_terms)
        self.assertEqual(expected_score, score)

    def test_termset_comparison_with_test_file(self):
        subject_terms = {"GO:0005634", "GO:0016020"}
        object_terms = {"GO:0031965", "GO:0005773"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=self.db,
        )
        expected_score = 5.4154243283740175
        score = semsimian.termset_comparison(subject_terms, object_terms)
        self.assertEqual(expected_score, score)

    def test_termset_pairwise_similarity(self):
        subject_terms = {"GO:0005634", "GO:0016020"}
        object_terms = {"GO:0031965", "GO:0005773"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=self.db,
        )
        tsps = semsimian.termset_pairwise_similarity(subject_terms, object_terms)
        self.assertEqual(tsps["average_score"], 5.4154243283740175)
        self.assertEqual(tsps["best_score"], 5.8496657269155685)

    @unittest.skipIf(
        sys.platform == "win32",
        "GitHub Windows errors because interval_1=0.0 it\
                   somehow rounds down in spite of code to avoid it",
    )
    def test_building_closure_ic_map_once(self):
        subject_terms = {"GO:0005634", "GO:0016020"}
        object_terms = {"GO:0031965", "GO:0005773"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=self.db,
        )
        load_start = time.time()
        _ = semsimian.termset_pairwise_similarity(subject_terms, object_terms)
        interval_1 = time.time() - load_start
        print(f"Warmup time: {interval_1} sec")
        second_compare_time = time.time()
        _ = semsimian.termset_pairwise_similarity(subject_terms, object_terms)
        interval_2 = time.time() - second_compare_time
        print(f"Second compare time: {interval_2} sec")
        self.assertTrue(interval_1 - interval_2 >= 0)

    def test_association_search(self):
        subject_prefixes = ["GO:"]
        object_terms = {"GO:0019222"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_nucleus"}
        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=self.db,
        )
        limit = 10
        result = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            True,
            False,
            None,
            subject_prefixes,
            limit,
        )
        self.assertEqual(len(result), limit)

    def test_association_quick_search(self):
        subject_prefixes = ["GO:"]
        object_terms = {"GO:0019222"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_nucleus"}
        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=self.db,
        )
        limit = 10
        result = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            True,
            True,
            None,
            subject_prefixes,
            limit,
        )
        self.assertEqual(len(result), limit)

    @unittest.skip("Too long and local db file.")
    def test_association_search_phenio(self):
        subject_prefixes = ["MGI:"]
        object_terms = {"MP:0003143"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_phenotype"}
        db_path = os.path.expanduser("~/.data/oaklib/phenio.db")
        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=db_path,
        )
        limit = 10
        result = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            True,
            False,
            None,
            subject_prefixes,
            limit,
        )
        self.assertEqual(len(result), limit)

    @unittest.skip("Too long and local db file.")
    def test_association_quick_search_phenio(self):
        subject_prefixes = ["MGI:"]
        object_terms = {"MP:0003143"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_phenotype"}
        db_path = os.path.expanduser("~/.data/oaklib/phenio.db")
        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=db_path,
        )
        limit = 10
        result = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            True,
            True,
            None,
            subject_prefixes,
            limit,
        )
        self.assertEqual(len(result), limit)

    @unittest.skip("Too long and local db file.")
    def test_association_search_caching(self):
        subject_prefixes = ["MGI:"]
        object_terms = {"MP:0003143"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_phenotype"}
        db_path = os.path.expanduser("~/.data/oaklib/phenio.db")
        include_similarity_object = True
        quick_search = False
        subject_terms = None
        limit = 10

        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=db_path,
        )
        load_start = time.time()
        search_1 = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            include_similarity_object,
            quick_search,
            subject_terms,
            subject_prefixes,
            limit,
        )
        interval_1 = time.time() - load_start
        print(f"Warmup time: {interval_1} sec")
        second_compare_time = time.time()

        search_2 = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            include_similarity_object,
            quick_search,
            subject_terms,
            subject_prefixes,
            limit,
        )
        interval_2 = time.time() - second_compare_time
        print(f"Second compare time: {interval_2} sec")
        self.assertTrue(interval_1 - interval_2 >= 0)
        self.assertEqual(len(search_1), len(search_2))


if __name__ == "__main__":
    unittest.main()
