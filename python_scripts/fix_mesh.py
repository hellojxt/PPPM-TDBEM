#!/usr/bin/env python

"""
Remesh the input mesh to remove degeneracies and improve triangle quality.
"""

import argparse
import numpy as np
from numpy.linalg import norm
import pymesh


def fix_mesh(mesh, detail):
    print("Fixing mesh...")
    target_len = float(detail)
    count = 0
    old_vertices_num = mesh.num_vertices
    old_faces_num = mesh.num_faces
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len / 2,
                                               preserve_feature=True)
        mesh, __ = pymesh.split_long_edges(mesh, target_len)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        count += 1
        if count > 20:
            break
    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)
    print("Completed in {} iterations.".format(count))
    print("Number of vertices: {} -> {}".format(old_vertices_num,
                                                mesh.num_vertices))
    print("Number of faces: {} -> {}".format(old_faces_num,
                                             mesh.num_faces))
    return mesh


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        "--detail", help="precision of detail to preserve", default=1e-2)
    parser.add_argument("in_mesh", help="input mesh")
    parser.add_argument("out_mesh", help="output mesh")
    return parser.parse_args()


def main():
    args = parse_args()
    mesh = pymesh.meshio.load_mesh(args.in_mesh)
    mesh = fix_mesh(mesh, detail=args.detail)
    pymesh.meshio.save_mesh(args.out_mesh, mesh)


if __name__ == "__main__":
    main()
