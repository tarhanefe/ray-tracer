from __future__ import annotations
from typing import TYPE_CHECKING
from vec3 import Vec3
from dataclasses import dataclass
from ray import Ray
from abc import ABC, abstractmethod
from math import sqrt, inf

if TYPE_CHECKING:
    from material import Material

# Axis-aligned bounding box as (min_x, min_y, min_z, max_x, max_y, max_z).
Box = tuple[float, float, float, float, float, float]

@dataclass
class HitRecord:
    t: float
    p: Vec3
    normal: Vec3
    material: Material

class Hittable(ABC):
    @abstractmethod
    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord | None:
        pass

    def bounding_box(self) -> Box | None:
        """Box enclosing the object, or None if it has no finite bounds."""
        return None

@dataclass
class Sphere(Hittable):
    center: Vec3
    radius: float
    material: Material

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        # Plain float math: this runs for every ray/sphere pair, so avoid creating Vec3s on a miss.
        origin, direction, center = ray.origin, ray.direction, self.center
        dx, dy, dz = direction.x, direction.y, direction.z
        ocx = origin.x - center.x
        ocy = origin.y - center.y
        ocz = origin.z - center.z

        a = dx * dx + dy * dy + dz * dz
        half_b = ocx * dx + ocy * dy + ocz * dz
        c = ocx * ocx + ocy * ocy + ocz * ocz - self.radius * self.radius

        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return None

        sqrt_discriminant = sqrt(discriminant)

        # Try the nearer intersection first.
        root = (-half_b - sqrt_discriminant) / a
        if not (t_min < root < t_max):
            # The ray may start inside the sphere; try the farther intersection.
            root = (-half_b + sqrt_discriminant) / a
            if not (t_min < root < t_max):
                return None

        p = ray.at(root)
        normal = (p - center) / self.radius

        return HitRecord(t=root, p=p, normal=normal, material=self.material)

    def bounding_box(self) -> Box:
        c, r = self.center, abs(self.radius)
        return (c.x - r, c.y - r, c.z - r, c.x + r, c.y + r, c.z + r)

def _surrounding_box(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes), min(b[2] for b in boxes),
        max(b[3] for b in boxes), max(b[4] for b in boxes), max(b[5] for b in boxes),
    )

class BVHNode:
    """Bounding volume hierarchy: skips every object whose box the ray misses."""

    LEAF_SIZE = 4

    def __init__(self, objects: list[Hittable], boxes: list[Box]):
        self.box = _surrounding_box(boxes)

        if len(objects) <= self.LEAF_SIZE:
            self.objects = objects
            self.left = self.right = None
            return

        # Split along the axis where the object centers are most spread out.
        centers = [((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2) for b in boxes]
        axis = max(range(3), key=lambda k: max(c[k] for c in centers) - min(c[k] for c in centers))
        order = sorted(range(len(objects)), key=lambda i: centers[i][axis])
        mid = len(order) // 2

        self.objects = None
        self.left = BVHNode([objects[i] for i in order[:mid]], [boxes[i] for i in order[:mid]])
        self.right = BVHNode([objects[i] for i in order[mid:]], [boxes[i] for i in order[mid:]])

    def hit(self, ray: Ray, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min: float, t_max: float) -> HitRecord | None:
        # Slab test: does the ray pass through this node's box within [t_min, t_max]?
        x0, y0, z0, x1, y1, z1 = self.box
        lo, hi = t_min, t_max
        for slab_min, slab_max, o, inv_d in ((x0, x1, ox, inv_dx), (y0, y1, oy, inv_dy), (z0, z1, oz, inv_dz)):
            t0 = (slab_min - o) * inv_d
            t1 = (slab_max - o) * inv_d
            if t0 > t1:
                t0, t1 = t1, t0
            if t0 > lo:
                lo = t0
            if t1 < hi:
                hi = t1
            if hi <= lo:
                return None

        if self.objects is not None:
            hit_record = None
            for obj in self.objects:
                temp_record = obj.hit(ray, t_min, t_max)
                if temp_record:
                    t_max = temp_record.t
                    hit_record = temp_record
            return hit_record

        left_record = self.left.hit(ray, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, t_max)
        if left_record:
            t_max = left_record.t
        right_record = self.right.hit(ray, ox, oy, oz, inv_dx, inv_dy, inv_dz, t_min, t_max)
        return right_record or left_record

class HittableList(Hittable):
    def __init__(self):
        self.objects: list[Hittable] = []
        self._bvh: BVHNode | None = None

    def add(self, obj: Hittable):
        self.objects.append(obj)
        self._bvh = None

    def __getstate__(self):
        # Send only the objects to worker processes; each worker rebuilds the BVH on its first hit.
        return {"objects": self.objects, "_bvh": None}

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        if self._bvh is None:
            boxes = [obj.bounding_box() for obj in self.objects]
            if not self.objects or any(box is None for box in boxes):
                return self._hit_each(ray, t_min, t_max)
            self._bvh = BVHNode(self.objects, boxes)

        origin, direction = ray.origin, ray.direction
        return self._bvh.hit(
            ray,
            origin.x, origin.y, origin.z,
            1.0 / direction.x if direction.x else inf,
            1.0 / direction.y if direction.y else inf,
            1.0 / direction.z if direction.z else inf,
            t_min, t_max,
        )

    def _hit_each(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        hit_record: HitRecord | None = None
        closest_so_far = t_max

        for obj in self.objects:
            temp_record = obj.hit(ray, t_min, closest_so_far)
            if temp_record:
                closest_so_far = temp_record.t
                hit_record = temp_record

        return hit_record

    def bounding_box(self) -> Box | None:
        boxes = [obj.bounding_box() for obj in self.objects]
        if not boxes or any(box is None for box in boxes):
            return None
        return _surrounding_box(boxes)
