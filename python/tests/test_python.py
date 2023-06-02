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
     predicates  = {"related_to"}
     result = self.semsimian.jaccard_similarity(term1, term2, predicates)
     self.assertEqual(result, 1.0/3.0)
                       
  def test_resnik_similarity(self):
     term1 = "apple"
     term2 = "banana"
     predicates  = {"related_to"}
     result = self.semsimian.resnik_similarity(term1, term2, predicates)
     self.assertEqual(result, 0.0)

if __name__ == "__main__":
    unittest.main()
  
