import numpy as np
import matplotlib.pyplot as plt



class DirectionalLight:
    def __init__(self, direction, color):
        self.dir = direction
        self.color = color

class Camera:
    def __init__(self, origin, focal_length, resolution):
        self.origin = origin
        self.focal_length = focal_length
        self.h, self.w = resolution
        self.aspect_ratio = resolution[1]/resolution[0]
    def get_ray(self, i, j):
        return Ray(origin=self.origin, dir=np.array([-(i/self.h - 0.5 - (0.5/self.h)), self.aspect_ratio * (0.5 - j / self.w) + (0.5/self.w), self.focal_length] - self.origin))

class Ray:
    def __init__(self, origin, dir):
        self.origin = origin
        self.dir = dir

class Sphere:
    def __init__(self, center, radius, kd):
        self.center = center
        self.radius = radius
        self.kd = kd
    def get_normal(self, points):
        normals = points - self.center
        return normals / np.linalg.norm(normals)


    
    def get_intersection_point(self, ray):
        a = np.linalg.norm(ray.dir) ** 2
        b = 2 * np.sum((ray.origin - self.center) * ray.dir)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2

        disc = b**2 - 4 * a * c
        
        if disc < 0:
            return -1
        
        disc = np.sqrt(disc)
        tmin = (-b - disc) / (2 * a)
        tmax = (-b + disc) / (2 * a)

        if tmin > 0:
            return ray.origin + ray.dir * tmin
        elif tmax > 0:
            return ray.origin + ray.dir * tmax
        return -1

def shade(light, surface, intersection_point):
    if isinstance(intersection_point, int):
        return np.zeros(3)
    normal = surface.get_normal(intersection_point)
    return max(0, -light.dir.dot(normal)) * np.array([1., 0., 0])


if __name__ == '__main__':
    h, w = 480, 960
    light = DirectionalLight(np.array([10., 0., 0.]), np.ones(3))
    img = np.zeros((h, w, 3))
    sphere = Sphere(center=np.array([0., 0., 1.1]), radius=1, kd=0)
    camera = Camera(origin=np.array([0, 0, 0]), focal_length=0.1, resolution=(h, w))

    for i in range(h):
        for j in range(w):
            img[i, j] = shade(light, sphere, sphere.get_intersection_point(camera.get_ray(i, j)))

    plt.savefig("test.png")
    # plt.show()
