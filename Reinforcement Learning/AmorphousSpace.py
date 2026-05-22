import os

# Get the absolute path to the directory containing this script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the file
file_path = os.path.join(dir_path, 'spheres.txt')

import numpy as np
from gymnasium import spaces

class AmorphousSpace(spaces.Space):
    """Custom space class for representing a 3D amorphous observation space with spheres."""

    def __init__(self):
        """
        Initialize the amorphous space with 3D spheres.

        Parameters
        ----------
        - spheres : list
            A list of dictionaries representing spherical regions in 3D space.
            Each dictionary contains:
          - 'center' : np.array
                The center of the sphere (3D point).
          - 'radius' : float
                The radius of the sphere.
        """
        spheres = []
        with open(file_path, 'r') as f:
            for line in f:
                x, y, z, r = line.strip().split(',')
                spheres.append({'center': np.array([float(x), float(y), float(z)]), 'radius': float(r)})

        self.spheres = spheres
        self.low = np.array([sphere['center'][0] - sphere['radius'] for sphere in spheres])
        self.high = np.array([sphere['center'][0] + sphere['radius'] for sphere in spheres])
        super(AmorphousSpace, self).__init__((3,))

    def sample(self):
        """Sample a random point from the amorphous space."""
        sphere = self.spheres[np.random.randint(len(self.spheres))]

        # Generate random point within sphere using spherical coordinates
        theta = np.random.uniform(low=0, high=2*np.pi)
        phi = np.random.uniform(low=0, high=np.pi)
        distance = np.random.uniform(low=0, high=sphere['radius'])

        x = sphere['center'][0] + distance * np.sin(phi) * np.cos(theta)
        y = sphere['center'][1] + distance * np.sin(phi) * np.sin(theta)
        z = sphere['center'][2] + distance * np.cos(phi)
        return np.array([x, y, z])

    def contains(self, x):
        """Check if a 3D point is within the bounds of the amorphous space."""
        for sphere in self.spheres:
            if np.linalg.norm(x - sphere['center']) <= sphere['radius']:
                return True
        return False

    def clip(self, x):
        """Clip a 3D point to the bounds of the amorphous space."""
        if self.contains(x):
            return x
        else:
            # Find the nearest point on the boundary of the space
            min_distance = float('inf')
            nearest_point = None
            for sphere in self.spheres:
                distance = np.linalg.norm(x - sphere['center'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = sphere['radius'] * (x - sphere['center']) / distance + sphere['center']
            return nearest_point
