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
        search_type = "full"
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
            search_type,
            None,
            subject_prefixes,
            limit,
        )
        self.assertEqual(len(result), limit)

    def test_association_flat_search(self):
        subject_prefixes = ["GO:"]
        object_terms = {"GO:0019222"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_nucleus"}
        search_type = "flat"
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
            search_type,
            None,
            subject_prefixes,
            limit,
        )
        self.assertEqual(len(result), limit)

    def test_association_hybrid_search(self):
        subject_prefixes = ["GO:"]
        object_terms = {"GO:0019222"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_nucleus"}
        search_type = "hybrid"
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
            search_type,
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
        search_type = "full"
        limit = 10
        result = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            True,
            False,
            search_type,
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
        search_type = "flat"
        limit = 10
        result = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            True,
            True,
            search_type,
            None,
            subject_prefixes,
            limit,
        )
        self.assertEqual(len(result), limit)

    # @unittest.skip("Too long and local db file.")
    def test_association_search_caching(self):
        subject_prefixes = ["MGI:"]
        # object_terms = {"MP:0003143"}
        # Test Ehlers-Danlos syndrome
        object_terms = {
                                    'HP:0100699',
                                    'HP:0001388',
                                    'HP:0001382',
                                    'HP:0001065',
                                    'HP:0001373',
                                    'HP:0000977',
                                    'HP:0002758',
                                    'HP:0000974',
                                    'HP:0001634',
                                    'HP:0000023',
                                    'HP:0000140',
                                    'HP:0000144',
                                    'HP:0000164',
                                    'HP:0000168',
                                    'HP:0000174',
                                    'HP:0000212',
                                    'HP:0000230',
                                    'HP:0000286',
                                    'HP:0000508',
                                    'HP:0000563',
                                    'HP:0000691',
                                    'HP:0000716',
                                    'HP:0000762',
                                    'HP:0000963',
                                    'HP:0000987',
                                    'HP:0001063',
                                    'HP:0001097',
                                    'HP:0001376',
                                    'HP:0001482',
                                    'HP:0001537',
                                    'HP:0001760',
                                    'HP:0001763',
                                    'HP:0002017',
                                    'HP:0002019',
                                    'HP:0002020',
                                    'HP:0002024',
                                    'HP:0002076',
                                    'HP:0002104',
                                    'HP:0002321',
                                    'HP:0002360',
                                    'HP:0002579',
                                    'HP:0002645',
                                    'HP:0002650',
                                    'HP:0002797',
                                    'HP:0002827',
                                    'HP:0002829',
                                    'HP:0003019',
                                    'HP:0003042',
                                    'HP:0003326',
                                    'HP:0003401',
                                    'HP:0004970',
                                    'HP:0005293',
                                    'HP:0005294',
                                    'HP:0005692',
                                    'HP:0010318',
                                    'HP:0011675',
                                    'HP:0012378',
                                    'HP:0012732',
                                    'HP:0100550',
                                    'HP:0100645',
                                    'HP:0100823'
                                }
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_phenotype"}
        db_path = os.path.expanduser("~/.data/oaklib/phenio.db")
        include_similarity_object = True
        quick_search = "full"
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

    def test_termset_pairwise_similarity_weighted_negated_method_exists(self):
        # assert method exists
        self.assertTrue(
            hasattr(self.semsimian, "termset_pairwise_similarity_weighted_negated"))
        pass



if __name__ == "__main__":
    unittest.main()

