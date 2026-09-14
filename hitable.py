from __future__ import annotations
from typing import TYPE_CHECKING
from vec3 import Vec3
from dataclasses import dataclass
from ray import Ray
from abc import ABC, abstractmethod
from math import sqrt

if TYPE_CHECKING:
    from material import Material

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

class HittableList(Hittable):
    def __init__(self):
        self.objects: list[Hittable] = []

    def add(self, obj: Hittable):
        self.objects.append(obj)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        hit_record: HitRecord | None = None
        closest_so_far = t_max

        for obj in self.objects:
            temp_record = obj.hit(ray, t_min, closest_so_far)
            if temp_record:
                closest_so_far = temp_record.t
                hit_record = temp_record

        return hit_record
