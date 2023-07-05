import unittest
import numpy as np
from networks import SongUNet, DhariwalUNet

class TestModels(unittest.TestCase):
    def test_SongUNet(self):
        model = SongUNet(img_resolution=224, channels=3, label_dim=128)
        input_data = np.random.normal(size=(2, 224, 224, 3))
        noise_labels = np.random.normal(size=(2,))
        class_labels = np.random.normal(size=(2, 128))
        output = model(input_data, noise_labels, class_labels)
        self.assertEqual(output.shape, (2, 224, 224, 3))

    def test_DhariwalUNet(self):
        model = DhariwalUNet(img_resolution=224, channels=3, label_dim=128)
        input_data = np.random.normal(size=(2, 224, 224, 3))
        noise_labels = np.random.normal(size=(2,))
        class_labels = np.random.normal(size=(2, 128))
        output = model(input_data, noise_labels, class_labels)
        self.assertEqual(output.shape, (2, 224, 224, 3))

if __name__ == '__main__':
    unittest.main()
