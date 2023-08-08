import unittest
from semsimian import Semsimian


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


if __name__ == "__main__":
    unittest.main()
