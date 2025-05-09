# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import os

import numpy as np
import torch
import tyro
import utm
from datasets import ColmapDataset
from datasets.colmap_dataset import ColmapScene
from osgeo import gdal, gdal_array, osr
from pyproj import Transformer
from scipy.interpolate import griddata

from fvdb import GaussianSplat3d


def lonlat2UTM(lon, lat):
    """
    Find utm zone closest to provided lon, lat in EPSG:4326
    Args:
        lon: longitude in decimal degrees
        lat: latitude in decimal degrees

    Returns:
        UTM Zone in well known text, proj4 string, and zone number
    """
    utm_dat = utm.from_latlon(lat, lon)
    zone_number = utm_dat[2]
    src = osr.SpatialReference()

    src.SetUTM(utm_dat[2])
    srs_wkt = src.ExportToWkt()
    srs_proj4 = src.ExportToProj4()

    return srs_wkt, srs_proj4, zone_number


def get_minmax(points, percentile=98):
    """
    Find minmax of provided points using a percentile across each dimension
    Args:
        points: locations in [npoints,ndim]
        percentile: percentile to filter by

    Returns:
        bounds in sequence of min, max across each dimension
    """
    ndim = points.shape[1]
    bounds = []
    for idx in range(ndim):
        lower_bound, upper_bound = np.percentile(points[:, 0], [100 - percentile, percentile])
        bounds.extend([lower_bound, upper_bound])

    return bounds


# def get_scene_minmax(points, percentile=98):
#     """
#     Find minmax of provided points using a percentile
#     Args:
#         points: 3d locations in [npoints,3]
#         percentile: percentile to filter by

#     Returns:
#         bounds in minx, maxx, miny, may
#     """
#     lower_boundx = np.percentile(points[:, 0], 100 - percentile)
#     upper_boundx = np.percentile(points[:, 0], percentile)

#     lower_boundy = np.percentile(points[:, 1], 100 - percentile)
#     upper_boundy = np.percentile(points[:, 1], percentile)

#     return lower_boundx, upper_boundx, lower_boundy, upper_boundy


# def get_scene_heights(points, percentile=99):
#     """
#     Find the min and max in z dimension using percentile filter
#     Args:
#         points: 3d locations in [npoints,3]
#         percentile: percentile to filter by

#     Returns:
#         bounds in minz, maxz
#     """
#     lower_boundz = np.percentile(points[:, 2], 100 - percentile)
#     upper_boundz = np.percentile(points[:, 2], percentile)

#     return lower_boundz, upper_boundz


def get_geot_with_npixx(minx, maxx, miny, maxy, npix):
    """
    Calculate the geotransform with set number of pixel resolution
    Args:
        minx: min value in x dimension
        maxx: mas value in x dimension
        miny: min value in y dimension
        maxy: max value in y dimension
        npix: number of desired pixels in x dimension

    Returns:
        geotransform
    """
    # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
    # GT(1) w-e pixel resolution / pixel width.
    # GT(2) row rotation (typically zero).
    # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
    # GT(4) column rotation (typically zero).
    # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).

    gsdx = (maxx - minx) / float(npix)  # assume square pixel
    npixy = (maxy - miny) / gsdx
    npixy = round(npixy)
    gsdy = (maxy - miny) / float(npixy)

    geot = [minx, gsdx, 0, maxy, 0, -gsdy]

    return geot, round(npixy)


def rotate_geot(geot, rot_deg):
    """
    Calculate the geotransform with set number of pixel resolution
    Args:
        geot: geotransform (6 parameter list of float)
        rot_deg: how much to rotate transform by in degrees

    Returns:
        geotransform
    """
    r = np.deg2rad(rot_deg)
    geot = [geot[0], np.cos(r) * geot[1], -np.sin(r) * geot[1], geot[3], np.sin(r) * geot[5], np.cos(r) * geot[5]]
    return geot


def write_geotiff(ortho, geot, srs_wkt, output_fname):
    """
    Write a geotiff out to disk.
    Args:
        ortho: image pixels of ortho
        geot: geotransform of ortho pixels,
        srs_wkt: well known text of coordinate reference system
        output_fname: filename of ortho to save to

    """
    type_code = gdal_array.NumericTypeCodeToGDALTypeCode(ortho.dtype)

    io_driver = gdal.GetDriverByName("GTiff")

    if len(ortho.shape) == 2:
        nb = 1
        ortho = ortho[:, :, np.newaxis]
    else:
        nb = ortho.shape[2]

    ortho_ds = io_driver.Create(output_fname, ortho.shape[1], ortho.shape[0], nb, type_code, options=["COMPRESS=LZW"])
    ortho_ds.SetGeoTransform(geot)
    ortho_ds.SetProjection(srs_wkt)

    for bc in range(nb):

        ortho_band = ortho_ds.GetRasterBand(bc + 1)
        ortho_band.SetNoDataValue(0)
        ortho_band.WriteArray(np.squeeze(ortho[:, :, bc]))
        ortho_band.FlushCache()
        ortho_band = None

    ortho_ds = None


@torch.no_grad()
def main(data_path: str, checkpoint_path: str, ortho_fn: str, npixx: int, percent_clip: int, device: str = "cuda"):
    """
    Script to create an orthomosaic from provided checkpoint. Assumes that Colmap run is in earth centered earth fixed coordinates

    Args:
        data_path: path to colmap run
        checkpoint_path: path to fvdb checkpoint
        ortho_fn: output path filename for ortho that is created
        npixx: desired number of pixels in x for output ortho
        percent_clip: percentile to use for clipping outlier splats [0-100]
        device: which device to use for ortho generation
    """

    EPSG_ECEF = "EPSG:4978"  # ecef from colmap model aligner

    trainset = ColmapDataset(
        dataset_path=data_path,
        image_downsample_factor=1,
        normalization_type="ecef2enu",
        test_every=10000,
        split="train",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    all_points = checkpoint["splats"]["means"].cpu().detach().numpy()
    # all_points = trainset.parser.points #can also use colmap sparse point cloud but less accurate?

    minx, maxx, miny, maxy, minz, maxz = get_minmax(all_points, percentile=percent_clip)

    geot_enu, npixy = get_geot_with_npixx(minx, maxx, miny, maxy, npixx)

    width, height = maxx - minx, maxy - miny
    scene_center = np.array([minx + width / 2.0, miny + height / 2.0])

    xv = np.array(
        [
            [minx, maxy],  # top left of image
            [minx, miny],  # bot left
            [maxx, miny],  # bot right
            [maxx, maxy],  # top right
            [scene_center[0], scene_center[1]],
        ]
    )

    good_inds = np.logical_and.reduce(
        [
            all_points[:, 0] > minx,
            all_points[:, 0] < maxx,
            all_points[:, 1] > miny,
            all_points[:, 1] < maxy,
            all_points[:, 2] > minz,
            all_points[:, 2] < maxz,
        ]
    )
    pts_in_grid = all_points[good_inds, :]

    # interpolate out z locations from ortho grid in enu
    interp = griddata(
        (pts_in_grid[:, 0].flatten(), pts_in_grid[:, 1].flatten()), pts_in_grid[:, 2].flatten(), xv, method="nearest"
    )
    image_corners_xyz = np.column_stack([xv, interp[..., np.newaxis]])

    invT = np.linalg.inv(trainset.colmap_scene.normalization_transform)
    image_corners_xyz_ecef = ColmapScene._transform_point_cloud(invT, image_corners_xyz)

    tform_ecef2lonlat = Transformer.from_crs(EPSG_ECEF, "EPSG:4326", always_xy=True)
    pt_lonlat = tform_ecef2lonlat.transform(
        image_corners_xyz_ecef[0, 0], image_corners_xyz_ecef[0, 1], image_corners_xyz_ecef[0, 2]
    )
    _, proj4_utm, zone_number = lonlat2UTM(pt_lonlat[0], pt_lonlat[1])
    tform_ecef2utm = Transformer.from_crs(EPSG_ECEF, proj4_utm, always_xy=True)

    image_corners_xyz_utm = tform_ecef2utm.transform(
        image_corners_xyz_ecef[:, 0], image_corners_xyz_ecef[:, 1], image_corners_xyz_ecef[:, 2]
    )
    image_corners_xyz_utm = np.column_stack(
        [image_corners_xyz_utm[0], image_corners_xyz_utm[1], image_corners_xyz_utm[2]]
    )

    tform_utm2lonlat = Transformer.from_crs(proj4_utm, "EPSG:4326", always_xy=True)
    gc_pnt = tform_utm2lonlat.transform(image_corners_xyz_utm[0][0], image_corners_xyz_utm[0][1])
    gcp = [np.deg2rad(gc_pnt[0]), np.deg2rad(gc_pnt[1])]
    central_meridian_degrees = (zone_number - 1.0) * 6.0 - 180.0 + 3.0
    cma = np.deg2rad(central_meridian_degrees)

    utm2enu_angle = np.arctan(np.tan(gcp[0] - cma) * np.sin(gcp[1]))
    utm2enu_angle_deg = np.rad2deg(utm2enu_angle)

    # update enu geot to utm
    geot_utm = geot_enu
    # update upper left to utm coordinate
    geot_utm[0] = image_corners_xyz_utm[0][0]
    geot_utm[3] = image_corners_xyz_utm[0][1]
    # apply grid convergence rotation
    geot_utm = rotate_geot(geot_utm, -utm2enu_angle_deg)

    # orthographic camera model
    # set flying height - place high above scene to avoid large camera angle
    cam_height = maxz * 100

    fx = npixx / width
    cx = npixx - fx * maxx

    fy = npixy / height
    cy = npixy - fy * maxy

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    c2w_manual = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    c2w_manual[:2, 3] = scene_center
    c2w_manual[2, 3] = cam_height

    w2c = np.linalg.inv(c2w_manual)

    world_to_cam_mats = torch.from_numpy(w2c[np.newaxis, ...]).float().to(device)
    intrinsics_mats = torch.from_numpy(K[np.newaxis, ...]).float().to(device)

    model = GaussianSplat3d.from_state_dict(checkpoint["splats"])

    torch.cuda.synchronize()

    colors, _ = model.render_images(
        world_to_camera_matrices=world_to_cam_mats,
        projection_matrices=intrinsics_mats,
        image_width=npixy,
        image_height=npixx,
        near=0.01,
        far=1e10,
        projection_type="orthographic",
    )
    colors = colors[0, ..., :3].cpu().numpy()

    torch.cuda.synchronize()

    # write image
    img = (colors * 255).astype(np.uint8)

    if os.path.exists(ortho_fn):
        os.remove(ortho_fn)

    # TODO not sure this is the correct way to adjust residual translation
    geot_utm[0] = geot_utm[0] + scene_center[0]
    geot_utm[3] = geot_utm[3] - scene_center[1]

    print("writing ortho to: ", ortho_fn)
    write_geotiff(img, geot_utm, proj4_utm, ortho_fn)


if __name__ == "__main__":
    tyro.cli(main)
