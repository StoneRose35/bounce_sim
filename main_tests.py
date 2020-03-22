import unittest
import main


class MainTests(unittest.TestCase):
    def testCollides(self):
        a = main.Agent()
        w = main.Wall()
        w.p1 = (0, 100)
        w.p2 = (100, 100)

        self.assertFalse(w.collides(a))

        a.position = (50, 97)

        self.assertTrue(w.collides(a))

        a.position = (120, 97)

        self.assertFalse(w.collides(a))

    def testBounce(self):
        a = main.Agent()
        w = main.Wall()
        w.p1 = (0, 100)
        w.p2 = (100, 100)

        d_new = w.bounce_direction(a)
        self.assertTrue(d_new >= 0.0 and d_new <= 360.0)



if __name__ == '__main__':
    unittest.main()
