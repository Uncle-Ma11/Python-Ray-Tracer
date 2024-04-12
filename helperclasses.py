import math

import glm
import numpy as np
from PIL import Image


# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry


class Ray:
    def __init__(self, o: glm.vec3, d: glm.vec3):
        self.origin = o
        self.direction = d

    def getDistance(self, point: glm.vec3):
        return glm.length(point - self.origin)

    def getPoint(self, t: float):
        return self.origin + self.direction * t


class Texture:
    def __init__(self, filename):
        # Load the image file as a texture
        self.image = Image.open(filename)
        self.width, self.height = self.image.size
        self.pixels = self.image.load()

    def sample(self, u, v):
        # Convert the UV texture coordinates to pixel coordinates
        x = int(u * self.width) % self.width
        y = int((1 - v) * self.height) % self.height  # Image origin is top left, so we flip v
        return glm.vec3(*self.pixels[x, y][:3]) / 255.0  # Convert color to [0, 1] range


class Material:
    def __init__(self, name: str, specular: glm.vec3, diffuse: glm.vec3, hardness: float, ID: int, reflective: bool,
                 refractive: bool, IOR: float, texture_file=None
                 ):
        self.name = name
        self.specular = specular
        self.diffuse = diffuse
        self.hardness = hardness
        self.ID = ID
        self.reflective = reflective
        self.refractive = refractive
        self.IOR = IOR
        self.texture = Texture(texture_file) if texture_file else None

    def get_color(self, u, v):
        return self.texture.sample(u, v)

    @staticmethod
    def default():
        name = "default"
        specular = diffuse = glm.vec3(0, 0, 0)
        hardness = ID = -1
        return Material(name, specular, diffuse, hardness, ID, False, False, 1.0)


class Light:
    def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, power: float):
        self.type = ltype
        self.name = name
        self.colour = colour
        self.vector = vector
        self.power = power


class AreaLight(Light):
    def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, power: float, shape: str,
                 normal: glm.vec3, height: float = None, width: float = None, radius: float = None):
        super().__init__(ltype, name, colour, vector, power)
        self.shape = shape
        self.normal = glm.normalize(normal)
        self.height = height
        self.width = width
        self.radius = radius

    def sample(self):
        sample_point = self.sample_circle() if self.shape == "circle" else self.sample_rectangle()
        # return Light(self.type, self.name, self.colour, sample_point, self.power)
        return sample_point

    def sample_circle(self):
        up, right = self.compute_local_coordinates()

        r = self.radius * math.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi

        local_point = r * (math.cos(theta) * right + math.sin(theta) * up)

        # Translate the local point to world space
        world_point = self.vector + local_point

        return world_point

    def sample_rectangle(self):
        up, right = self.compute_local_coordinates()

        u_offset = (np.random.random() - 0.5) * self.width
        v_offset = (np.random.random() - 0.5) * self.height

        # Calculate the local point within the bounds of the rectangle
        local_point = u_offset * right + v_offset * up

        # Translate the local point to world space
        world_point = self.vector + local_point

        return world_point

    def compute_local_coordinates(self):
        up = glm.vec3(0, 1, 0) if abs(self.normal.x) < abs(self.normal.z) else glm.vec3(1, 0, 0)
        right = glm.cross(self.normal, up)
        up = glm.cross(right, self.normal)
        right = glm.normalize(right)
        up = glm.normalize(up)

        return up, right


class Intersection:

    def __init__(self, time: float, normal: glm.vec3, position: glm.vec3, material: Material):
        self.time = time
        self.normal = normal
        self.position = position
        self.mat = material

    @staticmethod
    def default():
        time = float("inf")
        normal = glm.vec3(0, 0, 0)
        position = glm.vec3(0, 0, 0)
        mat = Material.default()
        return Intersection(time, normal, position, mat)
