from abc import ABC, abstractmethod
from math import sqrt
from dataclasses import dataclass
from ray import Ray
from vec3 import Vec3
from hitable import HitRecord
import random

class Material(ABC):
    @abstractmethod
    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray] | None:
        pass

def random_in_unit_sphere() -> Vec3:
    while True:
        p = 2.0 * Vec3(random.random(), random.random(), random.random()) - Vec3(1.0, 1.0, 1.0)
        if p.length_squared() < 1.0:
            return p

@dataclass
class Lambertian(Material):
    albedo: Vec3

    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray]:
        target = record.p + record.normal + random_in_unit_sphere()

        scattered_ray = Ray(
            origin=record.p,
            direction=target - record.p,
        )

        return self.albedo, scattered_ray

def reflect(v: Vec3, n: Vec3) -> Vec3:
    return v - 2.0 * v.dot(n) * n

def refract(v: Vec3, normal: Vec3, eta: float) -> Vec3 | None:
    """
    v: incoming ray direction
    normal: surface normal pointing against the incoming ray
    eta: refractive-index ratio, n1 / n2
    """
    unit_v = v.unit()

    cos_theta = min((-unit_v).dot(normal), 1.0)
    perpendicular = eta * (unit_v + cos_theta * normal)

    parallel_length_squared = 1.0 - perpendicular.length_squared()

    # No refracted ray is possible.
    if parallel_length_squared < 0.0:
        return None

    parallel = -sqrt(parallel_length_squared) * normal

    return perpendicular + parallel

def reflectance(cosine: float, refraction_ratio: float) -> float:
    r0 = ((1.0 - refraction_ratio) / (1.0 + refraction_ratio)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cosine) ** 5

@dataclass
class Metal(Material):
    albedo: Vec3
    fuzz: float = 0.0

    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray] | None:
        reflected = reflect(
            incoming_ray.direction.unit(),
            record.normal,
        )

        scattered_ray = Ray(record.p, reflected + self.fuzz * random_in_unit_sphere())

        # A valid reflection must leave the surface.
        if scattered_ray.direction.dot(record.normal) <= 0:
            return None

        return self.albedo, scattered_ray

@dataclass
class Dielectric(Material):
    refraction_index: float  # Glass is usually 1.5

    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray]:
        # Glass is clear: it does not tint or absorb light here.
        attenuation = Vec3(1.0, 1.0, 1.0)

        unit_direction = incoming_ray.direction.unit()

        # Is the ray entering the object or leaving it?
        if unit_direction.dot(record.normal) < 0:
            # Ray enters: air -> glass
            normal = record.normal
            eta = 1.0 / self.refraction_index
        else:
            # Ray exits: glass -> air
            normal = -record.normal
            eta = self.refraction_index

        cos_theta = min((-unit_direction).dot(normal), 1.0)
        sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

        # Total internal reflection, or probabilistic normal reflection.
        must_reflect = eta * sin_theta > 1.0

        if must_reflect or reflectance(cos_theta, eta) > random.random():
            direction = reflect(unit_direction, normal)
        else:
            direction = refract(unit_direction, normal, eta)

        scattered_ray = Ray(record.p, direction)

        return attenuation, scattered_ray