import math
from tqdm import tqdm

import glm
import numpy as np

import geometry as geom
import helperclasses as hc

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

shadow_epsilon = 10 ** (-3)
refract_epsilon = 10 ** (-4)


class Scene:

    def __init__(self,
                 width: int,
                 height: int,
                 jitter: bool,
                 samples: int,
                 position: glm.vec3,
                 lookat: glm.vec3,
                 up: glm.vec3,
                 fov: float,
                 ambient: glm.vec3,
                 lights: list[hc.Light],
                 materials: list[hc.Material],
                 objects: list[geom.Geometry],
                 max_bounces: int
                 ):
        self.width = width  # width of image
        self.height = height  # height of image
        self.aspect = width / height  # aspect ratio
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.position = position  # camera position in 3D
        self.lookat = lookat  # camera look at vector
        self.up = up  # camera up position
        self.fov = fov  # camera field of view
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.materials = materials  # all materials of objects in the scene
        self.objects = objects  # all objects in the scene
        self.max_bounces = max_bounces

    def is_high_variance(self, image, i, j, threshold=0.1):
        min_x = max(i - 1, 0)
        max_x = min(i + 1, self.width - 1)
        min_y = max(j - 1, 0)
        max_y = min(j + 1, self.height - 1)

        current_color = glm.vec3(image[i, j])

        # Check surrounding pixels
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if x == i and y == j:
                    continue  # Skip the current pixel

                neighbor_color = glm.vec3(image[x, y])
                color_diff = glm.abs(neighbor_color - current_color)

                # If the color difference exceeds the threshold,
                # we consider it high variance.
                if color_diff.x > threshold or color_diff.y > threshold or color_diff.z > threshold:
                    return True
        return False

    def generate_eye_ray(self, i, j, left, right, top, bottom, u, v, d, w):
        jitter_x = np.random.uniform() if self.jitter else 0.5
        jitter_y = np.random.uniform() if self.jitter else 0.5

        u_coefficient = left + (right - left) * (i + jitter_x) / self.width
        v_coefficient = bottom + (top - bottom) * (j + jitter_y) / self.height

        ray_dir = u_coefficient * u + v_coefficient * v - d * w
        return hc.Ray(self.position, glm.normalize(ray_dir))

    def find_intersection(self, ray):
        intersection = hc.Intersection.default()
        for obj in self.objects:
            intersection = obj.intersect(ray, intersection)
        return intersection

    def shadow_test(self, intersection, light_position, light_dir):
        shadow_ray = hc.Ray(intersection.position + shadow_epsilon * intersection.normal, light_dir)
        shadow_intersection = hc.Intersection.default()
        is_in_shadow = False
        for obj_shadow in self.objects:
            shadow_intersection = obj_shadow.intersect(shadow_ray, shadow_intersection)
            if shadow_intersection.time < glm.length(light_position - shadow_ray.origin):
                is_in_shadow = True
                break
        return is_in_shadow

    def compute_ambient(self, intersection):
        return self.ambient * intersection.mat.diffuse

    def compute_diffuse(self, intersection, light_dir, light_color, light_power):
        diff = max(glm.dot(intersection.normal, light_dir), 0.0)
        return light_color * light_power * intersection.mat.diffuse * diff

    def compute_specular(self, intersection, light_dir, light_colour, light_power):
        view_dir = glm.normalize(self.position - intersection.position)
        half_dir = glm.normalize(light_dir + view_dir)
        spec = max(glm.dot(intersection.normal, half_dir), 0.0) ** intersection.mat.hardness
        return light_colour * light_power * intersection.mat.specular * spec

    def compute_direct_lighting(self, intersection, light_samples=16):
        # ambient = self.compute_ambient(intersection)
        diffuse = glm.vec3(0, 0, 0)
        specular = glm.vec3(0, 0, 0)
        for light in self.lights:
            light_positions = []
            light_directions = []
            if isinstance(light, hc.AreaLight):
                for i in range(light_samples):
                    light_position = light.sample()
                    light_dir = glm.normalize(light_position - intersection.position)
                    light_positions.append(light_position)
                    light_directions.append(light_dir)
            else:
                light_positions.append(light.vector)
                light_directions.append(glm.normalize(light.vector - intersection.position))
            light_color = light.colour
            light_power = light.power
            
            for i in range(len(light_positions)):
                is_in_shadow = self.shadow_test(intersection, light_positions[i], light_directions[i])
                if is_in_shadow:
                    continue
                diffuse += self.compute_diffuse(intersection, light_directions[i], light_color, light_power)
                specular += self.compute_specular(intersection, light_directions[i], light_color, light_power)
            diffuse /= len(light_positions)
            specular /= len(light_positions)
        return diffuse + specular

    def add_local_shading(self, cur_ray, cur_intersection, cur_depth, is_inside):
        if cur_depth == self.max_bounces:
            return glm.vec3(0, 0, 0)

        color = self.compute_direct_lighting(cur_intersection)

        if cur_intersection.mat.reflective or cur_intersection.mat.refractive:
            cos_theta = glm.dot(-cur_ray.direction, cur_intersection.normal)
            fresnel_reflectance = self.fresnel_schlick(cos_theta, cur_intersection.mat.IOR)
            reflect_color = glm.vec3(0, 0, 0)

            if cur_intersection.mat.reflective:
                reflect_ray = self.create_reflected_ray(cur_ray, cur_intersection)
                reflect_intersection = self.find_intersection(cur_ray)
                if reflect_intersection.time != float("inf"):
                    reflect_color += self.add_local_shading(reflect_ray, reflect_intersection, cur_depth + 1, is_inside)
                    color += reflect_color * fresnel_reflectance

            if cur_intersection.mat.refractive:
                refract_ray = self.create_refracted_ray(cur_ray, cur_intersection, is_inside)
                if refract_ray is not None:
                    refract_intersection = self.find_intersection(refract_ray)
                    if refract_intersection.time != float("inf"):
                        refract_color = self.add_local_shading(refract_ray, refract_intersection, cur_depth + 1,
                                                               not is_inside)
                        color += refract_color * (1 - fresnel_reflectance)
                else:
                    color += reflect_color * (1 - fresnel_reflectance)

        return color

    def create_reflected_ray(self, ray, intersection):
        normalized_direction = glm.normalize(ray.direction)
        normalized_normal = glm.normalize(intersection.normal)

        reflection_direction = normalized_direction - 2 * glm.dot(normalized_direction,
                                                                  normalized_normal) * normalized_normal

        reflection_ray_origin = intersection.position + shadow_epsilon * normalized_normal
        reflection_ray = hc.Ray(reflection_ray_origin, reflection_direction)

        return reflection_ray

    def create_refracted_ray(self, cur_ray, cur_intersection, is_inside):
        eta = cur_intersection.mat.IOR if is_inside else 1.0 / cur_intersection.mat.IOR

        N = cur_intersection.normal if not is_inside else -cur_intersection.normal
        I = glm.normalize(cur_ray.direction)
        cos_theta_i = -glm.dot(N, I)
        sin2_theta_t = eta ** 2 * (1.0 - cos_theta_i ** 2)

        # Check for total internal reflection
        if sin2_theta_t > 1.0:
            return None

        # Calculate the refracted direction vector
        cos_theta_t = glm.sqrt(1.0 - sin2_theta_t)
        refracted_direction = eta * I + (eta * cos_theta_i - cos_theta_t) * N

        # Offset the intersection point by a small amount to avoid precision issues
        refracted_origin = cur_intersection.position + refract_epsilon * refracted_direction

        return hc.Ray(refracted_origin, glm.normalize(refracted_direction))

    def fresnel_schlick(self, cos, IOR):
        R0 = ((1 - IOR) / (1 + IOR)) ** 2
        return R0 + (1 - R0) * (1 - cos) ** 5

    def sample_color(self, i, j, left, right, top, bottom, u, v, d, w):
        colour = glm.vec3(0, 0, 0)
        ray = self.generate_eye_ray(i, j, left, right, top, bottom, u, v, d, w)

        intersection = self.find_intersection(ray)

        if intersection.time != float("inf"):
            colour += self.compute_ambient(intersection)
            colour += self.add_local_shading(ray, intersection, 0, False)

        return colour

    def render(self):

        image = np.zeros((self.width, self.height, 3))

        cam_dir = self.position - self.lookat
        d = 1.0
        top = d * math.tan(0.5 * math.pi * self.fov / 180)
        right = self.aspect * top
        bottom = -top
        left = -right

        w = glm.normalize(cam_dir)
        u = glm.cross(self.up, w)
        u = glm.normalize(u)
        v = glm.cross(w, u)

        for i in tqdm(range(self.width)):
            for j in range(self.height):
                colour_accumulator = glm.vec3(0, 0, 0)

                for s in range(self.samples):
                    colour_accumulator += self.sample_color(i, j, left, right, top, bottom, u, v, d, w)
                average_colour = colour_accumulator / self.samples
                image[i, j, 0] = max(0.0, min(1.0, average_colour.x))
                image[i, j, 1] = max(0.0, min(1.0, average_colour.y))
                image[i, j, 2] = max(0.0, min(1.0, average_colour.z))
        if self.jitter:
            for i in range(self.width):
                for j in range(self.height):
                    # If the pixel is near an edge or high variance area
                    if self.is_high_variance(image, i, j):
                        adaptive_color = glm.vec3(0, 0, 0)
                        additional_samples = 8  # A higher number of samples

                        for s in range(additional_samples):
                            # ... perform additional sampling ...
                            # Accumulate color contributions from the additional samples
                            adaptive_color += self.sample_color(i, j, left, right, top, bottom, u, v, d, w)
                        # Combine the colors from the initial and adaptive sampling
                        image[i, j] += adaptive_color
                        image[i, j] /= (self.samples + additional_samples)
        return image
