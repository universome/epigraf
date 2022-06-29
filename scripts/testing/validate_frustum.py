"""
Validates that the viewing frustum lies inside the [-1, 1]^3 cube
"""

import argparse
from src.training.rendering import validate_frustum

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius', type=float, help='Camera distance from the origin')
    parser.add_argument('-n', '--near', type=float, help='Position of the near plane')
    parser.add_argument('-f', '--far', type=float, help='Position of the far plane')
    parser.add_argument('--fov', type=float, help='Field of view (in degrees)')
    parser.add_argument('--scale', type=float, default=1.0, help='The additional scaling of the [-1, 1] cube (if any)')
    parser.add_argument('--step', type=float, default=1e-2, help='Step size when sampling the points')
    parser.add_argument('--device', type=str, default='cpu', help='`cpu` or `cuda` device?')
    args = parser.parse_args()

    is_valid = validate_frustum(
        radius=args.radius,
        near=args.near,
        far=args.far,
        fov=args.fov,
        scale=args.scale,
        step=args.step,
        device=args.device,
        verbose=True,
    )
    print('Valid?', is_valid)
