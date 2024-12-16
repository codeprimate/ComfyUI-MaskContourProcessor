import unittest
import numpy as np
import torch
from nodes.nodes import MaskContourProcessor

class TestMaskContourProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = MaskContourProcessor()
        
        # Create some common test masks
        self.empty_mask = np.zeros((64, 64), dtype=np.float32)
        self.full_mask = np.ones((64, 64), dtype=np.float32)
        self.circle_mask = np.zeros((64, 64), dtype=np.float32)
        center = (32, 32)
        radius = 20
        y, x = np.ogrid[:64, :64]
        self.circle_mask[(x - center[0])**2 + (y - center[1])**2 <= radius**2] = 1

    def test_calculate_mask_centroid(self):
        """Test centroid calculation for various masks."""
        # Test empty mask
        center_x, center_y = self.processor.calculate_mask_centroid(self.empty_mask)
        self.assertEqual((center_x, center_y), (32, 32))  # Should return center of image

        # Test full mask
        center_x, center_y = self.processor.calculate_mask_centroid(self.full_mask)
        self.assertEqual((center_x, center_y), (31.5, 31.5))  # Should be center of image

        # Test circle mask
        center_x, center_y = self.processor.calculate_mask_centroid(self.circle_mask)
        self.assertAlmostEqual(center_x, 32, delta=1)
        self.assertAlmostEqual(center_y, 32, delta=1)

    def test_detect_edge_points(self):
        """Test edge point detection."""
        center = (32, 32)
        edge_points = self.processor.detect_edge_points(self.circle_mask, center)
        
        # Check that we got some edge points
        self.assertTrue(len(edge_points) > 0)
        
        # Check that all points are on the edge (should have at least one
        # neighboring pixel with different value)
        for x, y in edge_points:
            has_different_neighbor = False
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < 64 and 0 <= ny < 64 and 
                    self.circle_mask[ny, nx] != self.circle_mask[y, x]):
                    has_different_neighbor = True
                    break
            self.assertTrue(has_different_neighbor)

    def test_redistribute_points(self):
        """Test point redistribution."""
        # Create a simple square of points
        square_points = [(0,0), (0,10), (10,10), (10,0)]
        
        # Test redistribution to different numbers of points
        for target_count in [4, 8, 16]:
            new_points = self.processor.redistribute_points(square_points, target_count)
            self.assertEqual(len(new_points), target_count)

    def test_process_mask(self):
        """Test the main processing function."""
        # Convert numpy mask to tensor
        mask_tensor = torch.from_numpy(self.circle_mask)
        
        # Process the mask
        result = self.processor.process_mask(
            mask_tensor,
            line_length=1.0,
            line_count=20,
            line_width=0.01
        )
        
        # Check return type and shape
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertEqual(result[0].shape, mask_tensor.shape)
        
        # Check that the output contains at least the original mask
        # (output should be >= input everywhere)
        self.assertTrue(torch.all(result[0] >= mask_tensor))

if __name__ == '__main__':
    unittest.main()
