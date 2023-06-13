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

        self.semsimian = Semsimian(spo)

    def test_jaccard_similarity(self):
        term1 = "apple"
        term2 = "banana"
        predicates = {"related_to"}
        result = self.semsimian.jaccard_similarity(term1, term2, predicates)
        self.assertEqual(result, 1.0 / 3.0)

    def test_resnik_similarity(self):
        term1 = "apple"
        term2 = "banana"
        predicates = {"related_to"}
        result = self.semsimian.resnik_similarity(term1, term2, predicates)
        self.assertEqual(result, ({"banana"}, 1.3219280948873622))

    def test_all_by_all_pairwise_similarity(self):
        subject_terms = {"apple", "banana", "orange"}
        object_terms = {"orange", "pear", "kiwi"}
        predicates = {"related_to"}
        orange_mica = {"orange", "pear"}
        result = self.semsimian.all_by_all_pairwise_similarity(
            subject_terms, object_terms, 0.0, 0.0, predicates
        )
        self.assertEqual(result["orange"]["orange"][3], orange_mica)


if __name__ == "__main__":
    unittest.main()
