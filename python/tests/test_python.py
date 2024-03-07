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
        self.aic_metric = "ancestor_information_content"
        self.phenodigm_metric = "phenodigm_score"
        self.jaccard_metric = "jaccard_similarity"

        self.semsimian = Semsimian(spo, predicates)
        self.db = str(Path(__file__).parents[2] / "tests/data/go-nucleus.db")
        self.marfan_syndrome_profile = {
            "HP:0100775",
            "HP:0003179",
            "HP:0001083",
            "HP:0000501",
            "HP:0002705",
            "HP:0004382",
            "HP:0004326",
            "HP:0002816",
            "HP:0004298",
            "HP:0002996",
            "HP:0002808",
            "HP:0002751",
            "HP:0002647",
            "HP:0002636",
            "HP:0002616",
            "HP:0002435",
            "HP:0002360",
            "HP:0007800",
            "HP:0032934",
            "HP:0012432",
            "HP:0007720",
            "HP:0002107",
            "HP:0002105",
            "HP:0007676",
            "HP:0000939",
            "HP:0000938",
            "HP:0002097",
            "HP:0012369",
            "HP:0000767",
            "HP:0000678",
            "HP:0012019",
            "HP:0010807",
            "HP:0000577",
            "HP:0000565",
            "HP:0000545",
            "HP:0000541",
            "HP:0000494",
            "HP:0000486",
            "HP:0006687",
            "HP:0007018",
            "HP:0000278",
            "HP:0000276",
            "HP:0000275",
            "HP:0000272",
            "HP:0000268",
            "HP:0000218",
            "HP:0000189",
            "HP:0000175",
            "HP:0000098",
            "HP:0000023",
            "HP:0001635",
            "HP:0001763",
            "HP:0005294",
            "HP:0003758",
            "HP:0003326",
            "HP:0003302",
            "HP:0003202",
            "HP:0003199",
            "HP:0005059",
            "HP:0003088",
            "HP:0025586",
            "HP:0005136",
            "HP:0001761",
            "HP:0001704",
            "HP:0001765",
            "HP:0001659",
            "HP:0001653",
            "HP:0001634",
            "HP:0001533",
            "HP:0001519",
            "HP:0008132",
            "HP:0001382",
            "HP:0001371",
            "HP:0001252",
            "HP:0001166",
            "HP:0001132",
            "HP:0000347",
            "HP:0001065",
            "HP:0000490",
            "HP:0000505",
            "HP:0000518",
            "HP:0000768",
            "HP:0004970",
            "HP:0004933",
            "HP:0004927",
            "HP:0002108",
            "HP:0004872",
            "HP:0012499",
            "HP:0002650",
        }
        self.ehlers_danlos_profile = {
            "HP:0100699",
            "HP:0001388",
            "HP:0001382",
            "HP:0001065",
            "HP:0001373",
            "HP:0000977",
            "HP:0002758",
            "HP:0000974",
            "HP:0001634",
            "HP:0000023",
            "HP:0000140",
            "HP:0000144",
            "HP:0000164",
            "HP:0000168",
            "HP:0000174",
            "HP:0000212",
            "HP:0000230",
            "HP:0000286",
            "HP:0000508",
            "HP:0000563",
            "HP:0000691",
            "HP:0000716",
            "HP:0000762",
            "HP:0000963",
            "HP:0000987",
            "HP:0001063",
            "HP:0001097",
            "HP:0001376",
            "HP:0001482",
            "HP:0001537",
            "HP:0001760",
            "HP:0001763",
            "HP:0002017",
            "HP:0002019",
            "HP:0002020",
            "HP:0002024",
            "HP:0002076",
            "HP:0002104",
            "HP:0002321",
            "HP:0002360",
            "HP:0002579",
            "HP:0002645",
            "HP:0002650",
            "HP:0002797",
            "HP:0002827",
            "HP:0002829",
            "HP:0003019",
            "HP:0003042",
            "HP:0003326",
            "HP:0003401",
            "HP:0004970",
            "HP:0005293",
            "HP:0005294",
            "HP:0005692",
            "HP:0010318",
            "HP:0011675",
            "HP:0012378",
            "HP:0012732",
            "HP:0100550",
            "HP:0100645",
            "HP:0100823",
        }

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

    def test_termset_comparison_aic(self):
        subject_terms = {"apple", "banana", "orange"}
        object_terms = {"orange", "pear", "kiwi"}
        expected_aic_score = 0.8812853965915748
        expected_pheno_score = 0.6045201840255207
        expected_jaccard_score = 0.4444444444444444
        score_aic = self.semsimian.termset_comparison(
            subject_terms, object_terms, self.aic_metric
        )
        score_pheno = self.semsimian.termset_comparison(
            subject_terms, object_terms, self.phenodigm_metric
        )
        score_jaccard = self.semsimian.termset_comparison(
            subject_terms, object_terms, self.jaccard_metric
        )
        self.assertEqual(expected_aic_score, score_aic)
        self.assertEqual(expected_pheno_score, score_pheno)
        self.assertEqual(expected_jaccard_score, score_jaccard)

    def test_termset_comparison_aic_with_test_file(self):
        subject_terms = {"GO:0005634", "GO:0016020"}
        object_terms = {"GO:0031965", "GO:0005773"}
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=self.db,
        )
        expected_aic_score = 5.4154243283740175
        expected_pheno_score = 1.8610697515464185
        expected_jaccard_score = 0.6878019323671498
        score_aic = semsimian.termset_comparison(
            subject_terms, object_terms, self.aic_metric
        )
        score_pheno = semsimian.termset_comparison(
            subject_terms, object_terms, self.phenodigm_metric
        )
        score_jaccard = semsimian.termset_comparison(
            subject_terms, object_terms, self.jaccard_metric
        )
        self.assertEqual(expected_aic_score, score_aic)
        self.assertEqual(expected_pheno_score, score_pheno)
        self.assertEqual(expected_jaccard_score, score_jaccard)

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
        tsps_aic = semsimian.termset_pairwise_similarity(
            subject_terms, object_terms, self.aic_metric
        )
        tsps_pheno = semsimian.termset_pairwise_similarity(
            subject_terms, object_terms, self.phenodigm_metric
        )
        tsps_jaccard = semsimian.termset_pairwise_similarity(
            subject_terms, object_terms, self.jaccard_metric
        )
        self.assertEqual(tsps_aic["average_score"], 5.4154243283740175)
        self.assertEqual(tsps_aic["best_score"], 5.8496657269155685)
        self.assertEqual(tsps_pheno["average_score"], 1.8610697515464185)
        self.assertEqual(tsps_pheno["best_score"], 2.06411807897654)
        self.assertEqual(tsps_jaccard["average_score"], 0.6878019323671498)
        self.assertEqual(tsps_jaccard["best_score"], 0.8333333333333334)

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
        _ = semsimian.termset_pairwise_similarity(
            subject_terms, object_terms, self.aic_metric
        )
        interval_1 = time.time() - load_start
        print(f"Warmup time: {interval_1} sec")
        second_compare_time = time.time()
        _ = semsimian.termset_pairwise_similarity(
            subject_terms, object_terms, self.aic_metric
        )
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
            None,
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
            self.aic_metric,
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
            None,
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
            search_type,
            None,
            subject_prefixes,
            self.aic_metric,
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
            search_type,
            None,
            subject_prefixes,
            None,
            limit,
        )
        self.assertEqual(len(result), limit)

    @unittest.skip("Too long and local db file.")
    def test_association_search_caching(self):
        subject_prefixes = ["MGI:"]
        # object_terms = {"MP:0003143"}
        object_terms = self.marfan_syndrome_profile
        predicates = ["rdfs:subClassOf", "BFO:0000050", "UPHENO:0000001"]
        assoc_predicate = {"biolink:has_phenotype"}
        db_path = os.path.expanduser("~/.data/oaklib/phenio.db")
        include_similarity_object = False
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
            self.aic_metric,
            limit,
        )
        interval_1 = time.time() - load_start
        print(f"Round 1 search time: {interval_1} sec")
        second_compare_time = time.time()

        search_2 = semsimian.associations_search(
            assoc_predicate,
            object_terms,
            include_similarity_object,
            quick_search,
            subject_terms,
            subject_prefixes,
            self.aic_metric,
            limit,
        )
        interval_2 = time.time() - second_compare_time
        print(f"Round 2 search time: time: {interval_2} sec")
        self.assertTrue(interval_1 - interval_2 >= 0)
        self.assertEqual(len(search_1), len(search_2))

    def test_termset_pairwise_similarity_weighted_negated_method_exists(self):
        # assert method exists
        self.assertTrue(
            hasattr(self.semsimian, "termset_pairwise_similarity_weighted_negated")
        )
        pass

    def test_semsimian_cache_creation(self):
        predicates = ["rdfs:subClassOf", "BFO:0000050"]
        assoc_predicate = {"biolink:has_nucleus"}

        # assoc_predicate = {"biolink:has_phenotype"}
        # db_path = os.path.expanduser("~/.data/oaklib/phenio.db")

        semsimian = Semsimian(
            spo=None,
            predicates=predicates,
            pairwise_similarity_attributes=None,
            resource_path=self.db,
        )
        search_type = "flat"
        semsimian.pregenerate_cache(assoc_predicate, search_type)
        # semsimian.get_prefix_association_cache() returns a dictionary
        cache = semsimian.get_prefix_association_cache()

        # Assert that the dictionary is not empty
        self.assertTrue(bool(cache), "The prefix association cache is empty")


if __name__ == "__main__":
    unittest.main()
