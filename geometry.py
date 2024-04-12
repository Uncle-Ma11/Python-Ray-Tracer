import math

import numpy as np

import helperclasses as hc
import glm
import igl

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

epsilon = 10 ** (-4)


class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect


class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        ori_to_center = self.center - ray.origin
        a = glm.dot(ray.direction, ray.direction)
        b = glm.dot(2 * ori_to_center, ray.direction)
        c = glm.dot(ori_to_center, ori_to_center) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return intersect

        t0 = -(-b + math.sqrt(discriminant)) / (2 * a)
        t1 = -(-b - math.sqrt(discriminant)) / (2 * a)

        if t0 > epsilon and t1 > epsilon:
            t = min(t0, t1)
        elif t0 > epsilon:
            t = t0
        elif t1 > epsilon:
            t = t1
        else:
            return intersect

        if t >= intersect.time:
            return intersect

        intersect_point = ray.getPoint(t)
        normal = glm.normalize(intersect_point - self.center)
        if self.materials[0].texture:
            u, v = self.compute_spherical_uv(intersect_point)
            self.materials[0].diffuse = self.materials[0].get_color(u, v)
        return hc.Intersection(t, normal, intersect_point, self.materials[0])

    def compute_spherical_uv(self, intersect_point):
        # Get the normalized vector from the sphere center to the intersect point
        p = glm.normalize(intersect_point - self.center)
        # Spherical coordinates
        u = 0.5 + (np.arctan2(p.z, p.x) / (2 * np.pi))
        v = 0.5 - (np.arcsin(p.y) / np.pi)
        return u, v


class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        test = glm.dot(ray.direction, self.normal)
        if abs(test) < epsilon:
            return intersect
        t = glm.dot((self.point - ray.origin), self.normal) / test
        if t > epsilon:
            intersect_point = ray.getPoint(t)
            int_x = math.floor(intersect_point.x)
            int_z = math.floor(intersect_point.z)
            if (int_x + int_z) % 2 == 0:
                mat = self.materials[0]
            else:
                mat = self.materials[1]
            return hc.Intersection(t, self.normal, intersect_point, mat)
        else:
            return intersect


class Quadric(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], Q: glm.mat4):
        super().__init__(name, gtype, materials)
        self.Q = Q

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        origin = glm.vec4(ray.origin, 1.0)
        direction = glm.vec4(ray.direction, 0.0)

        a = glm.dot(direction, self.Q * direction)
        b = glm.dot(direction, self.Q * origin) + glm.dot(origin, self.Q * direction)
        c = glm.dot(origin, self.Q * origin)

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return intersect

        t0 = (-b + math.sqrt(discriminant)) / (2 * a)
        t1 = (-b - math.sqrt(discriminant)) / (2 * a)

        if t0 > epsilon and t1 > epsilon:
            t = min(t0, t1)
        elif t0 > epsilon:
            t = t0
        elif t1 > epsilon:
            t = t1
        else:
            return intersect

        if t >= intersect.time:
            return intersect

        intersect_point = ray.getPoint(t)
        intersect_point_homo = glm.vec4(intersect_point, 1.0)
        normal_homo = glm.vec4(2.0) * self.Q * intersect_point_homo
        normal = glm.normalize(glm.vec3(normal_homo))
        if glm.dot(normal, ray.direction) > 0:
            normal = - normal
        return hc.Intersection(t, normal, intersect_point, self.materials[0])


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        halfside = dimension / 2
        self.minpos = center - halfside
        self.maxpos = center + halfside

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        for i in range(3):
            if abs(ray.direction[i]) < 1e-6:
                # if ray.origin[i] < self.minpos[i] or ray.origin[i] > self.maxpos[i]:
                return intersect
        tx_min = (self.minpos.x - ray.origin.x) / ray.direction.x
        tx_max = (self.maxpos.x - ray.origin.x) / ray.direction.x
        ty_min = (self.minpos.y - ray.origin.y) / ray.direction.y
        ty_max = (self.maxpos.y - ray.origin.y) / ray.direction.y
        tz_min = (self.minpos.z - ray.origin.z) / ray.direction.z
        tz_max = (self.maxpos.z - ray.origin.z) / ray.direction.z

        tx_low = min(tx_min, tx_max)
        tx_high = max(tx_min, tx_max)
        ty_low = min(ty_min, ty_max)
        ty_high = max(ty_min, ty_max)
        tz_low = min(tz_min, tz_max)
        tz_high = max(tz_min, tz_max)

        t_min = max(tx_low, ty_low, tz_low)
        t_max = min(tx_high, ty_high, tz_high)

        if t_min >= intersect.time:
            return intersect

        if t_min < t_max and t_max >= 0:
            if t_min == tx_low:
                normal = glm.vec3(1, 0, 0)
            elif t_min == ty_low:
                normal = glm.vec3(0, 1, 0)
            else:
                normal = glm.vec3(0, 0, 1)
            intersect_point = ray.getPoint(t_min)
            return hc.Intersection(t_min, normal, intersect_point, self.materials[0])
        else:
            return intersect


class BVH:
    def __init__(self, faces=None):
        self.bbox = None
        self.left = None
        self.right = None
        self.faces = faces

    def is_leaf(self):
        return self.faces is not None


class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
        for n in norms:
            self.norms.append(glm.vec3(n[0], n[1], n[2]))
        self.root = self.construct_bvh(self.faces)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return self.intersect_helper(ray, intersect, self.root)
        # for f in self.faces:
        #     intersect = self.triangle_intersect(ray, intersect, f)
        # return intersect

    def intersect_helper(self, ray: hc.Ray, intersect: hc.Intersection, node: BVH):
        intersection = node.bbox.intersect(ray, hc.Intersection.default())
        if intersection.time == float("inf"):
            return intersect
        if node.is_leaf():
            tri_intersection = hc.Intersection.default()
            for f in node.faces:
                tri_intersection = self.triangle_intersect(ray, tri_intersection, f)
            return tri_intersection
        intersection_left = self.intersect_helper(ray, intersect, node.left)
        intersection_right = self.intersect_helper(ray, intersect, node.right)
        return intersection_left if intersection_left.time < intersection_right.time else intersection_right

    def triangle_intersect(self, ray: hc.Ray, intersect: hc.Intersection, face):
        v0 = self.verts[face[0]]
        v1 = self.verts[face[1]]
        v2 = self.verts[face[2]]

        e1 = v1 - v0
        e2 = v2 - v0
        s = ray.origin - v0
        s1 = glm.cross(ray.direction, e2)
        s2 = glm.cross(s, e1)

        det = glm.dot(s1, e1)
        if abs(det) < epsilon:
            return intersect
        factor = 1 / det
        b1 = factor * glm.dot(s1, s)
        b2 = factor * glm.dot(s2, ray.direction)

        if b1 < 0.0 or b1 > 1.0:
            return intersect

        if b2 < 0.0 or b1 + b2 > 1.0:
            return intersect

        t = factor * glm.dot(s2, e2)
        if t >= intersect.time:
            return intersect
        if t > epsilon:
            normal = glm.normalize(glm.cross(e1, e2))
            position = ray.getPoint(t)
            return hc.Intersection(t, normal, position, self.materials[0])
        else:
            return intersect

    def compute_bounding_box(self, faces):
        # Calculate the axis-aligned bounding box for a set of faces
        min_point = glm.vec3(float('inf'), float('inf'), float('inf'))
        max_point = glm.vec3(float('-inf'), float('-inf'), float('-inf'))

        for face in faces:
            for idx in face:
                vert = self.verts[idx]
                min_point = glm.min(min_point, vert)
                max_point = glm.max(max_point, vert)

        return min_point, max_point

    def split_faces(self, faces):
        min_corner, max_corner = self.compute_bounding_box(faces)
        bbox_size = max_corner - min_corner

        # Determine the longest axis (0: x-axis, 1: y-axis, 2: z-axis)
        if bbox_size.x > bbox_size.y and bbox_size.x > bbox_size.z:
            split_axis = 0  # x-axis
        elif bbox_size.y > bbox_size.z:
            split_axis = 1  # y-axis
        else:
            split_axis = 2  # z-axis

        # Calculate centroids of faces and find the median along the longest axis
        centroids = [(self.verts[face[0]] + self.verts[face[1]] + self.verts[face[2]]) / 3.0 for face in faces]
        sorted_faces = sorted(zip(centroids, faces), key=lambda x: x[0][split_axis])

        median_idx = len(sorted_faces) // 2

        # Split faces based on the median
        left_faces = [face for centroid, face in sorted_faces[:median_idx]]
        right_faces = [face for centroid, face in sorted_faces[median_idx:]]

        return left_faces, right_faces

    def merge_boxes(self, box1, box2):
        min_1 = box1.minpos
        max_1 = box1.maxpos
        min_2 = box2.minpos
        max_2 = box2.maxpos

        # Calculate the union of the two bounding boxes
        merged_min = glm.min(min_1, min_2)
        merged_max = glm.max(max_1, max_2)

        merged_box = AABB("bounding box", "box", self.materials, glm.vec3(0, 0, 0), glm.vec3(0, 0, 0))
        merged_box.minpos = merged_min
        merged_box.maxpos = merged_max

        return merged_box

    def construct_bvh(self, faces):
        if len(faces) <= 10:
            min_point, max_point = self.compute_bounding_box(faces)
            box = AABB("bounding box", "box", self.materials, glm.vec3(0, 0, 0), glm.vec3(0, 0, 0))
            box.minpos = min_point
            box.maxpos = max_point
            bvh_leaf = BVH(faces=faces)
            bvh_leaf.bbox = box
            return bvh_leaf

        left_faces, right_faces = self.split_faces(faces)
        left_node = self.construct_bvh(left_faces)
        right_node = self.construct_bvh(right_faces)
        bounding_box = self.merge_boxes(left_node.bbox, right_node.bbox)
        cur_node = BVH()
        cur_node.bbox = bounding_box
        cur_node.left = left_node
        cur_node.right = right_node
        return cur_node


class Hierarchy(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], t: glm.vec3, r: glm.vec3, s: glm.vec3):
        super().__init__(name, gtype, materials)
        self.t = t
        self.M = glm.mat4(1.0)
        self.Minv = glm.mat4(1.0)
        self.make_matrices(t, r, s)
        self.children: list[Geometry] = []

    def make_matrices(self, t: glm.vec3, r: glm.vec3, s: glm.vec3):
        self.M = glm.mat4(1.0)
        self.M = glm.translate(self.M, t)
        self.M = glm.rotate(self.M, glm.radians(r.x), glm.vec3(1, 0, 0))
        self.M = glm.rotate(self.M, glm.radians(r.y), glm.vec3(0, 1, 0))
        self.M = glm.rotate(self.M, glm.radians(r.z), glm.vec3(0, 0, 1))
        self.M = glm.scale(self.M, s)
        self.Minv = glm.inverse(self.M)
        self.t = t

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        transformed_ray = hc.Ray(
            o=(self.Minv * glm.vec4(ray.origin, 1.0)).xyz,
            d=(self.Minv * glm.vec4(ray.direction, 0.0)).xyz
        )

        intersection = hc.Intersection.default()

        for child in self.children:
            child_intersection = child.intersect(transformed_ray, hc.Intersection.default())
            if child_intersection.time < intersection.time:
                intersection = child_intersection
        if intersection.time == float('inf'):
            return intersect
        intersection.position = glm.vec3(self.M * glm.vec4(intersection.position, 1.0))
        intersection.normal = glm.normalize(glm.vec3(glm.transpose(self.Minv) * glm.vec4(intersection.normal, 0.0)))
        return intersection
