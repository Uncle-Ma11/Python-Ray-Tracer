import copy
import json
import helperclasses as hc
import geometry as geom
import scene
import glm


# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry


def populateVec(array: list):
    return glm.vec3(array[0], array[1], array[2])


def load_scene(infile):
    print("Parsing file:", infile)
    f = open(infile)
    data = json.load(f)

    # Loading camera
    cam_pos = populateVec(data["camera"]["position"])
    cam_lookat = populateVec(data["camera"]["lookAt"])
    cam_up = populateVec(data["camera"]["up"])
    cam_fov = data["camera"]["fov"]
    depth = data["depth"]

    # Loading resolution
    try:
        width = data["resolution"][0]
        height = data["resolution"][1]
    except KeyError:
        print("No resolution found, defaulting to 1080x720.")
        width = 1080
        height = 720

    # Loading ambient light
    try:
        ambient = populateVec(data["ambient"])
    except KeyError:
        print("No ambient light defined, defaulting to [0, 0, 0]")
        ambient = populateVec([0, 0, 0])

    # Loading Anti-Aliasing options
    try:
        jitter = data["AA"]["jitter"]
        samples = data["AA"]["samples"]
    except KeyError:
        print("No Anti-Aliasing options found, setting to defualt")
        jitter = False
        samples = 1

    # Loading scene lights
    lights = []
    try:
        for light in data["lights"]:
            l_type = light["type"]
            l_name = light["name"]
            l_colour = populateVec(light["colour"])

            if l_type == "point":
                l_vector = populateVec(light["position"])
                l_power = light["power"]
                light = hc.Light(l_type, l_name, l_colour, l_vector, l_power)

            elif l_type == "directional":
                l_vector = populateVec(light["direction"])
                l_power = 1.0
                light = hc.Light(l_type, l_name, l_colour, l_vector, l_power)
            elif l_type == "area":
                l_vector = populateVec(light["position"])
                l_power = light["power"]
                l_shape = light["shape"]
                l_normal = light["normal"]
                if l_shape == "circle":
                    l_radius = light["radius"]
                    light = hc.AreaLight(l_type, l_name, l_colour, l_vector, l_power, l_shape, l_normal,
                                         radius=l_radius)
                else:
                    l_height = light["height"]
                    l_width = light["width"]
                    light = hc.AreaLight(l_type, l_name, l_colour, l_vector, l_power, l_shape, l_normal,
                                         height=l_height, width=l_width)
            else:
                print("Unkown light type", l_type, ", skipping initialization")
                continue
            lights.append(light)
    except KeyError:
        print("here")
        lights = []

    # Loading materials
    materials = []
    for material in data["materials"]:
        mat_diffuse = populateVec(material["diffuse"])
        mat_specular = populateVec(material["specular"])
        mat_hardness = material["hardness"]
        mat_name = material["name"]
        mat_id = material["ID"]
        mat_reflective = material["reflective"]
        mat_refractive = material["refractive"]
        mat_IOR = material["IOR"]
        if "texture" in material:
            mat_texture = material["texture"]
            mat = hc.Material(mat_name, mat_specular, mat_diffuse, mat_hardness, mat_id, mat_reflective, mat_refractive,
                              mat_IOR, mat_texture)
        else:
            mat = hc.Material(mat_name, mat_specular, mat_diffuse, mat_hardness, mat_id, mat_reflective, mat_refractive,
                              mat_IOR)
        materials.append(mat)

    # Loading geometry
    objects = []

    # Extra stuff for hierarchies
    rootNames = []
    roots = []
    for geometry in data["objects"]:
        # Elements common to all objects: name, type, position, material(s)
        g_name = geometry["name"]
        g_type = geometry["type"]
        g_pos = populateVec(geometry["position"])
        g_mats = associate_material(materials, geometry["materials"])

        if add_basic_shape(g_name, g_type, g_pos, g_mats, geometry, objects):
            # Non-hierarchies are straightforward
            continue
        elif g_type == "node":
            g_ref = geometry["ref"]
            g_r = populateVec(geometry["rotation"])
            g_s = populateVec(geometry["scale"])

            if g_ref == "":
                # Brand-new hierarchy
                rootNames.append(g_name)
                node = geom.Hierarchy(g_name, g_type, g_mats, g_pos, g_r, g_s)
                traverse_children(node, geometry["children"], materials)
                roots.append(node)
                objects.append(node)
            else:
                # Hierarchy that depends on a previously defined one
                rid = -1
                for i in range(len(rootNames)):
                    # Find hierarchy that this references
                    if g_ref == rootNames[i]:
                        rid = i
                        break
                if rid != -1:
                    node = copy.deepcopy(roots[rid])
                    node.name = g_name
                    node.materials = g_mats
                    node.make_matrices(g_pos, g_r, g_s)
                    objects.append(node)
                else:
                    print("Node reference", g_ref, "not found, skipping creation")

        else:
            print("Unkown object type", g_type, ", skipping initialization")
            continue

    print("Parsing complete")
    return scene.Scene(width, height, jitter, samples,  # General settings
                       cam_pos, cam_lookat, cam_up, cam_fov,  # Camera settings
                       ambient, lights,  # Light settings
                       materials, objects, depth)  # General settings


def add_basic_shape(g_name: str, g_type: str, g_pos: glm.vec3, g_mats: list[hc.Material], geometry,
                    objects: list[geom.Geometry]):
    # Function for adding non-hierarchies to a list, since there's nothing extra to do with them
    # Returns True if a shape was added, False otherwise
    if g_type == "sphere":
        g_radius = geometry["radius"]
        objects.append(geom.Sphere(g_name, g_type, g_mats, g_pos, g_radius))
    elif g_type == "plane":
        g_normal = populateVec(geometry["normal"])
        objects.append(geom.Plane(g_name, g_type, g_mats, g_pos, g_normal))
    elif g_type == "box":
        try:
            g_size = populateVec(geometry["size"])
            objects.append(geom.AABB(g_name, g_type, g_mats, g_pos, g_size))
        except KeyError:
            # Boxes can also be directly declared with a min and max position
            box = geom.AABB(g_name, g_type, g_mats, g_pos, glm.vec3(0, 0, 0))
            box.minpos = populateVec(geometry["min"])
            box.maxpos = populateVec(geometry["max"])
            objects.append(box)
    elif g_type == "mesh":
        g_path = geometry["filepath"]
        g_scale = geometry["scale"]
        objects.append(geom.Mesh(g_name, g_type, g_mats, g_pos, g_scale, g_path))
    elif g_type == "quadric":
        A = geometry["A"]
        B = geometry["B"]
        C = geometry["C"]
        D = geometry["D"]
        E = geometry["E"]
        F = geometry["F"]
        G = geometry["G"]
        H = geometry["H"]
        I = geometry["I"]
        J = geometry["J"]
        g_Q = glm.mat4(A, D, E, G,
                       D, B, F, H,
                       E, F, C, I,
                       G, H, I, J)
        objects.append(geom.Quadric(g_name, g_type, g_mats, g_Q))
    else:
        return False
    return True


def traverse_children(node: geom.Hierarchy, children, materials: list[hc.Material]):
    for geometry in children:
        # Obtain info common to all shapes like in the main body of the parser
        g_name = geometry["name"]
        g_type = geometry["type"]
        try:
            g_pos = populateVec(geometry["position"])
        except KeyError:
            g_pos = glm.vec3(0, 0, 0)
        g_mats = associate_material(materials, geometry["materials"])

        if add_basic_shape(g_name, g_type, g_pos, g_mats, geometry, node.children):
            # Nothing fancy to do for non-hierarchies
            continue
        elif g_type == "node":
            # Hierarchy within a hierarchy, recurse
            g_r = populateVec(geometry["rotation"])
            g_s = populateVec(geometry["scale"])
            inner = geom.Hierarchy(g_name, g_type, g_mats, g_pos, g_r, g_s)
            node.children.append(inner)
            traverse_children(inner, geometry["children"], materials)
        else:
            print("Unkown child object type", g_type, ", skipping initialization")


def associate_material(mats: list[hc.Material], ids: list[int]):
    new_list = []
    for i in ids:
        for mat in mats:
            if i == mat.ID:
                new_list.append(mat)
    return new_list
