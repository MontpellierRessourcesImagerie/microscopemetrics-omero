import json
from itertools import product
from random import choice
from string import ascii_letters
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from omero import grid
from omero.constants import metadata
from omero.gateway import (
    BlitzGateway,
    CommentAnnotationWrapper,
    DatasetWrapper,
    FileAnnotationWrapper,
    ImageWrapper,
    MapAnnotationWrapper,
    ProjectWrapper,
    RoiWrapper,
    TagAnnotationWrapper,
)
from omero.model import (
    DatasetI,
    EllipseI,
    ImageI,
    LengthI,
    LineI,
    MaskI,
    OriginalFileI,
    PointI,
    PolygonI,
    ProjectI,
    RectangleI,
    RoiI,
    enums,
    DatasetImageLinkI,
    ProjectDatasetLinkI,
)
from omero.rtypes import rdouble, rint, rlong, rstring
from pandas import DataFrame

from microscopemetrics.data_schema import core_schema as mm_schema

DTYPES_NP_TO_OMERO = {
    "int8": enums.PixelsTypeint8,
    "int16": enums.PixelsTypeint16,
    "uint16": enums.PixelsTypeuint16,
    "int32": enums.PixelsTypeint32,
    "float_": enums.PixelsTypefloat,
    "float8": enums.PixelsTypefloat,
    "float16": enums.PixelsTypefloat,
    "float32": enums.PixelsTypefloat,
    "float64": enums.PixelsTypedouble,
    "complex_": enums.PixelsTypecomplex,
    "complex64": enums.PixelsTypecomplex,
}

DTYPES_OMERO_TO_NP = {
    enums.PixelsTypeint8: "int8",
    enums.PixelsTypeuint8: "uint8",
    enums.PixelsTypeint16: "int16",
    enums.PixelsTypeuint16: "uint16",
    enums.PixelsTypeint32: "int32",
    enums.PixelsTypeuint32: "uint32",
    enums.PixelsTypefloat: "float32",
    enums.PixelsTypedouble: "double",
}


COLUMN_TYPES = {
    "string": grid.StringColumn,
    "long": grid.LongColumn,
    "bool": grid.BoolColumn,
    "double": grid.DoubleColumn,
    "long_array": grid.LongArrayColumn,
    "float_array": grid.FloatArrayColumn,
    "double_array": grid.DoubleArrayColumn,
    "image": grid.ImageColumn,
    "dataset": grid.DatasetColumn,
    "plate": grid.PlateColumn,
    "well": grid.WellColumn,
    "roi": grid.RoiColumn,
    "mask": grid.MaskColumn,
    "file": grid.FileColumn,
}


def get_object_from_url(url: str) -> List[Tuple[str, int]]:
    """Get the ID from an OMERO URL. For example:
    https://omero.server.fr/webclient/?show=image-12345 or
    https://omero.mri.cnrs.fr/webclient/?show=image-1556622|image-1556623 for multiple objects"""
    tail = url.split("/")[-1].split("=")[-1]  # TODO: make all of this a regex
    if "|" in tail:
        return [(x.split("-")[0], int(x.split("-")[-1])) for x in tail.split("|")]
    else:
        return [(tail.split("-")[0], int(tail.split("-")[-1]))]


def get_url_from_object(obj: Union[ImageWrapper, DatasetWrapper, ProjectWrapper]) -> str:
    """Get the URL from an OMERO object"""
    return obj._conn.c.host + "/webclient/?show=" + obj.OMERO_TYPE + "-" + str(obj.getId())


def _label_channels(image, labels):
    if len(labels) != image.getSizeC():
        raise ValueError('The length of the channel labels is not of the same size as the size of the c dimension')
    for label, channel in zip(labels, image.getChannels(noRE=True)):
        logical_channel = channel.getLogicalChannel()
        logical_channel.setName(label)
        logical_channel.save()


def _get_image_shape(image):
    try:
        image_shape = (image.getSizeZ(),
                       image.getSizeC(),
                       image.getSizeT(),
                       image.getSizeY(),
                       image.getSizeX())
    except Exception as e:
        raise e

    return image_shape


def _get_pixel_size(image, order='ZXY'):
    pixels = image.getPrimaryPixels()

    order = order.upper()
    if order not in ['ZXY', 'ZYX', 'XYZ', 'XZY', 'YXZ', 'YZX']:
        raise ValueError('The provided order for the axis is not valid')
    pixel_sizes = ()
    for a in order:
        pixel_sizes += (getattr(pixels, f'getPhysicalSize{a}')().getValue(), )
    return pixel_sizes


def _get_pixel_size_units(image):
    pixels = image.getPrimaryPixels()

    return (
        pixels.getPhysicalSizeX().getUnit().name,
        pixels.getPhysicalSizeY().getUnit().name,
        pixels.getPhysicalSizeZ().getUnit().name,
    )


def get_image_intensities(
        image: ImageWrapper,
        z_range=None,
        c_range: range = None,
        t_range: range = None,
        y_range: range = None,
        x_range: range = None
):
    """Returns a numpy array containing the intensity values of the image
    Returns an array with dimensions arranged as zctyx
    """
    image_shape = _get_image_shape(image)

    # TODO: verify that image fits in ice message size. Otherwise get in tiles

    # Decide if we are going to call getPlanes or getTiles
    if not x_range and not y_range:
        whole_planes = True
    else:
        whole_planes = False

    ranges = list(range(5))
    for dim, r in enumerate([z_range, c_range, t_range, y_range, x_range]):
        # Verify that requested ranges are within the available data
        if r is None:  # Range is not specified
            ranges[dim] = range(image_shape[dim])
        else:  # Range is specified
            if type(r) is int:
                ranges[dim] = range(r, r + 1)
            elif type(r) is not tuple:
                raise TypeError("Range is not provided as a tuple.")
            else:  # range is a tuple
                if len(r) == 1:
                    ranges[dim] = range(r[0])
                elif len(r) == 2:
                    ranges[dim] = range(r[0], r[1])
                elif len(r) == 3:
                    ranges[dim] = range(r[0], r[1], r[2])
                else:
                    raise IndexError("Range values must contain 1 to three values")
            if not 1 <= ranges[dim].stop <= image_shape[dim]:
                raise IndexError("Specified range is outside of the image dimensions")

    output_shape = (
        len(ranges[0]),
        len(ranges[1]),
        len(ranges[2]),
        len(ranges[3]),
        len(ranges[4]),
    )
    nr_planes = output_shape[0] * output_shape[1] * output_shape[2]
    zct_list = list(product(ranges[0], ranges[1], ranges[2]))

    pixels = image.getPrimaryPixels()
    data_type = DTYPES_OMERO_TO_NP[pixels.getPixelsType().getValue()]

    # intensities = np.zeros(output_shape, dtype=data_type)

    intensities = np.zeros(
        shape=(nr_planes, output_shape[3], output_shape[4]), dtype=data_type
    )
    if whole_planes:
        np.stack(list(pixels.getPlanes(zctList=zct_list)), out=intensities)
    else:
        # Tile is formatted (X, Y, Width, Heigth)
        tile_region = (ranges[4].start, ranges[3].start, len(ranges[4]), len(ranges[3]))
        zct_tile_list = [(z, c, t, tile_region) for z, c, t in zct_list]
        np.stack(list(pixels.getTiles(zctTileList=zct_tile_list)), out=intensities)

    intensities = np.reshape(intensities, newshape=output_shape)

    return intensities


def get_tagged_images_in_dataset(dataset, tag_id):
    images = []
    for image in dataset.listChildren():
        for ann in image.listAnnotations():
            if type(ann) == TagAnnotationWrapper and ann.getId() == tag_id:
                images.append(image)
    return images


def _create_image_copy(conn,
                       source_image_id,
                       channel_list=None,
                       image_name=None,
                       image_description=None,
                       size_x=None, size_y=None, size_z=None, size_t=None):
    """Creates a copy of an existing OMERO image using all the metadata but not the pixels values.
    The parameter values will override the ones of the original image"""
    pixels_service = conn.getPixelsService()

    if channel_list is None:
        source_image = conn.getObject('Image', source_image_id)
        channel_list = list(range(source_image.getSizeC()))

    image_id = pixels_service.copyAndResizeImage(imageId=source_image_id,
                                                 sizeX=rint(size_x),
                                                 sizeY=rint(size_y),
                                                 sizeZ=rint(size_z),
                                                 sizeT=rint(size_t),
                                                 channelList=channel_list,
                                                 methodology=image_name,
                                                 copyStats=False)

    new_image = conn.getObject("Image", image_id)

    if image_description is not None:  # Description is not provided as an override option in the OMERO interface
        new_image.setDescription(image_description)
        new_image.save()

    return new_image


def _create_image(conn, image_name, size_x, size_y, size_z, size_t, size_c, data_type, channel_labels=None, image_description=None):
    """Creates an OMERO empty image from scratch"""
    pixels_service = conn.getPixelsService()
    query_service = conn.getQueryService()

    if data_type not in DTYPES_NP_TO_OMERO:  # try to look up any not named above
        pixel_type = data_type
    else:
        pixel_type = DTYPES_NP_TO_OMERO[data_type]

    pixels_type = query_service.findByQuery(
        f"from PixelsType as p where p.value='{pixel_type}'", None
    )
    if pixels_type is None:
        raise ValueError(
            "Cannot create an image in omero from numpy array "
            "with dtype: %s" % data_type)

    image_id = pixels_service.createImage(sizeX=size_x,
                                          sizeY=size_y,
                                          sizeZ=size_z,
                                          sizeT=size_t,
                                          channelList=list(range(size_c)),
                                          pixelsType=pixels_type,
                                          name=image_name,
                                          description=image_description)

    new_image = conn.getObject("Image", image_id.getValue())

    if channel_labels is not None:
        _label_channels(new_image, channel_labels)

    return new_image


def _create_image_whole(conn, data, image_name, image_description=None, dataset=None, channel_list=None, source_image_id=None):

    zct_generator = (data[z, c, t, :, :] for z, c, t in product(range(data.shape[0]),
                                                                range(data.shape[1]),
                                                                range(data.shape[2])))

    return conn.createImageFromNumpySeq(zctPlanes=zct_generator,
                                        imageName=image_name,
                                        sizeZ=data.shape[0],
                                        sizeC=data.shape[1],
                                        sizeT=data.shape[2],
                                        description=image_description,
                                        dataset=dataset,
                                        channelList=channel_list,
                                        sourceImageId=source_image_id)


def create_image_from_numpy_array(conn: BlitzGateway,
                                  data: np.ndarray,
                                  image_name: str,
                                  image_description: str = None,
                                  channel_labels: Union[list, tuple] = None,
                                  dataset: DatasetWrapper = None,
                                  source_image_id: int = None,
                                  channels_list: list[int] = None,
                                  force_whole_planes: bool = False):
    """
    Creates a new image in OMERO from a n dimensional numpy array.
    :param channel_labels: A list of channel labels
    :param force_whole_planes:
    :param channels_list:
    :param conn: The conn object to OMERO
    :param data: the ndarray. Must be a 5D array with dimensions in the order zctyx
    :param image_name: The name of the image
    :param image_description: The description of the image
    :param dataset: The dataset where the image will be created
    :param source_image_id: The id of the image to copy metadata from
    :return: The new image
    """

    zct_list = list(product(range(data.shape[0]), range(data.shape[1]), range(data.shape[2])))
    zct_generator = (data[z, c, t, :, :] for z, c, t in zct_list)

    # Verify if the image must be tiled
    max_plane_size = conn.getMaxPlaneSize()
    if force_whole_planes or (data.shape[-1] < max_plane_size[-1] and data.shape[-2] < max_plane_size[-2]):
        # Image is small enough to fill it with full planes
        new_image = conn.createImageFromNumpySeq(zctPlanes=zct_generator,
                                                 imageName=image_name,
                                                 sizeZ=data.shape[0],
                                                 sizeC=data.shape[1],
                                                 sizeT=data.shape[2],
                                                 description=image_description,
                                                 dataset=dataset,
                                                 sourceImageId=source_image_id,
                                                 channelList=channels_list)

    else:
        zct_tile_list = _get_tile_list(zct_list, data.shape, max_plane_size)

        if source_image_id is not None:
            new_image = _create_image_copy(conn, source_image_id,
                                           image_name=image_name,
                                           image_description=image_description,
                                           size_x=data.shape[-1],
                                           size_y=data.shape[-2],
                                           size_z=data.shape[0],
                                           size_t=data.shape[2],
                                           channel_list=channels_list)

        else:
            new_image = _create_image(conn,
                                      image_name=image_name,
                                      size_x=data.shape[-1],
                                      size_y=data.shape[-2],
                                      size_z=data.shape[0],
                                      size_t=data.shape[2],
                                      size_c=data.shape[1],
                                      data_type=data.dtype.name,
                                      image_description=image_description)

        raw_pixel_store = conn.c.sf.createRawPixelsStore()
        pixels_id = new_image.getPrimaryPixels().getId()
        raw_pixel_store.setPixelsId(pixels_id, True)

        for tile_coord in zct_tile_list:
            tile_data = data[tile_coord[0],
                             tile_coord[1],
                             tile_coord[2],
                             tile_coord[3][1]:tile_coord[3][1] + tile_coord[3][3],
                             tile_coord[3][0]:tile_coord[3][0] + tile_coord[3][2]]
            tile_data = tile_data.byteswap()
            bin_tile_data = tile_data.tostring()

            raw_pixel_store.setTile(bin_tile_data,
                                    tile_coord[0],
                                    tile_coord[1],
                                    tile_coord[2],
                                    tile_coord[3][0],
                                    tile_coord[3][1],
                                    tile_coord[3][2],
                                    tile_coord[3][3],
                                    conn.SERVICE_OPTS
                                    )

        if dataset is not None:
            _link_image_to_dataset(conn, new_image, dataset)

    if channel_labels is not None:
        _label_channels(new_image, channel_labels)

    return new_image


def _get_tile_list(zct_list, data_shape, tile_size):
    zct_tile_list = []
    for p in zct_list:
        for tile_offset_y in range(0, data_shape[-2], tile_size[1]):
            for tile_offset_x in range(0, data_shape[-1], tile_size[0]):
                tile_width = tile_size[0]
                tile_height = tile_size[1]
                if tile_width + tile_offset_x > data_shape[-1]:
                    tile_width = data_shape[-1] - tile_offset_x
                if tile_height + tile_offset_y > data_shape[-2]:
                    tile_height = data_shape[-2] - tile_offset_y

                tile_xywh = (tile_offset_x, tile_offset_y, tile_width, tile_height)
                zct_tile_list.append((*p, tile_xywh))

    return zct_tile_list


def create_roi(conn, image, shapes, name, description):
    # create an ROI, link it to Image
    roi = RoiI()  # TODO: work with wrappers
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(image._obj)
    if name is not None:
        roi.setName(rstring(name))
    if description is not None:
        roi.setDescription(rstring(description))
    for shape in shapes:
        roi.addShape(shape)

    return RoiWrapper(conn, conn.getUpdateService().saveAndReturnObject(roi))

# TODO: move references to model out of here
def _rgba_to_int(rgba_color: mm_schema.Color):
    """Return the color as an Integer in RGBA encoding"""
    r = rgba_color.r << 24
    g = rgba_color.g << 16
    b = rgba_color.b << 8
    a = rgba_color.alpha
    rgba_int = sum([r, g, b, a])
    if rgba_int > (2**31 - 1):  # convert to signed 32-bit int
        rgba_int = rgba_int - 2**32

    return rgba_int


def _set_shape_properties(
    shape,
    label: str = None,
    fill_color: mm_schema.Color = None,
    stroke_color: mm_schema.Color = None,
    stroke_width: int = None,
):
    if label is not None:
        shape.setTextValue(rstring(label))
    if fill_color is not None:
        shape.setFillColor(rint(_rgba_to_int(fill_color)))
    if stroke_color is not None:
        shape.setStrokeColor(rint(_rgba_to_int(stroke_color)))
    if stroke_width is not None:
        shape.setStrokeWidth(LengthI(stroke_width, enums.UnitsLength.PIXEL))


def create_shape_point(mm_point: mm_schema.Point):
    point = PointI()
    point.x = rdouble(mm_point.x)
    point.y = rdouble(mm_point.y)
    if mm_point.z is not None:
        point.theZ = rint(mm_point.z)
    if mm_point.c is not None:
        point.theC = rint(mm_point.c)
    if mm_point.t is not None:
        point.theT = rint(mm_point.t)
    _set_shape_properties(
        shape=point,
        label=mm_point.label,
        stroke_color=mm_point.stroke_color,
        stroke_width=mm_point.stroke_width,
        fill_color=mm_point.fill_color,
    )
    return point


def create_shape_line(mm_line: mm_schema.Line):
    line = LineI()
    line.x1 = rdouble(mm_line.x1)
    line.x2 = rdouble(mm_line.x2)
    line.y1 = rdouble(mm_line.y1)
    line.y2 = rdouble(mm_line.y2)
    line.theZ = rint(mm_line.z)
    line.theT = rint(mm_line.t)
    if mm_line.c is not None:
        line.theC = rint(mm_line.c)
    _set_shape_properties(
        shape=line,
        label=mm_line.label,
        stroke_color=mm_line.stroke_color,
        stroke_width=mm_line.stroke_width
    )
    return line


def create_shape_rectangle(mm_rectangle: mm_schema.Rectangle):
    rect = RectangleI()
    rect.x = rdouble(mm_rectangle.x)
    rect.y = rdouble(mm_rectangle.y)
    rect.width = rdouble(mm_rectangle.w)
    rect.height = rdouble(mm_rectangle.h)
    rect.theZ = rint(mm_rectangle.z)
    rect.theT = rint(mm_rectangle.t)
    _set_shape_properties(
        shape=rect,
        label=mm_rectangle.label,
        fill_color=mm_rectangle.fill_color,
        stroke_color=mm_rectangle.stroke_color,
        stroke_width=mm_rectangle.stroke_width,
    )
    return rect


def create_shape_ellipse(mm_ellipse: mm_schema.Ellipse):
    ellipse = EllipseI()
    ellipse.setX(rdouble(mm_ellipse.x))
    ellipse.setY(rdouble(mm_ellipse.y))  # TODO: setters and getters everywhere
    ellipse.radiusX = rdouble(mm_ellipse.x_rad)
    ellipse.radiusY = rdouble(mm_ellipse.y_rad)
    ellipse.theZ = rint(mm_ellipse.z)
    ellipse.theT = rint(mm_ellipse.t)
    _set_shape_properties(
        ellipse,
        label=mm_ellipse.label,
        fill_color=mm_ellipse.fill_color,
        stroke_color=mm_ellipse.stroke_color,
        stroke_width=mm_ellipse.stroke_width,
    )
    return ellipse


def create_shape_polygon(mm_polygon: mm_schema.Polygon):
    polygon = PolygonI()
    points_str = "".join(
        ["".join([str(vertex.x), ",", str(vertex.y), ", "]) for vertex in mm_polygon.vertexes]
    )[:-2]
    polygon.points = rstring(points_str)
    polygon.theZ = rint(mm_polygon.z)
    polygon.theT = rint(mm_polygon.t)
    _set_shape_properties(
        polygon,
        label=mm_polygon.label,
        fill_color=mm_polygon.fill_color,
        stroke_color=mm_polygon.stroke_color,
        stroke_width=mm_polygon.stroke_width,
    )
    return polygon


def create_shape_mask(mm_mask: mm_schema.Mask):
    mask = MaskI()
    mask.setX(rdouble(mm_mask.x))
    mask.setY(rdouble(mm_mask.y))
    mask.setTheZ(rint(mm_mask.z))
    mask.setTheT(rint(mm_mask.t))
    mask.setWidth(rdouble(mm_mask.mask.x.values[0]))  # TODO: see how to get shape if not np.array
    mask.setHeight(rdouble(mm_mask.mask.y.values[0]))
    mask_packed = np.packbits(mm_mask.mask.data)  # TODO: raise error when not boolean array
    mask.setBytes(mask_packed.tobytes())  # TODO: review how to setBytes when not a np.array
    _set_shape_properties(
        mask,
        label=mm_mask.label,
        fill_color=mm_mask.fill_color,
    )
    return mask


def create_tag(conn: BlitzGateway, tag_text: str, tag_description: str, omero_object):
    tag_ann = TagAnnotationWrapper(conn)
    tag_ann.setValue(tag_text)
    tag_ann.setDescription(tag_description)
    tag_ann.save()

    _link_annotation(omero_object, tag_ann)

    return tag_ann

def _serialize_map_value(value):
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except ValueError:
        # TODO: log an error
        return json.dumps(value.__str__())


def _dict_to_map(dictionary):
    """Converts a dictionary into a list of key:value pairs to be fed as map annotation.
    If value is not a string we serialize it as a json string"""
    return [[str(k), _serialize_map_value(v)] for k, v in dictionary.items()]


def create_key_value(
    conn: BlitzGateway,
    annotation: dict,
    omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
    annotation_name=None,
    annotation_description=None,
    namespace=None,
):
    """Creates a map_annotation for OMERO. It can create a map annotation from a
    dictionary or from a list of 2 elements list.
    """
    if namespace is None:
        namespace = (
            metadata.NSCLIENTMAPANNOTATION
        )  # This makes the annotation editable in the client
    # Convert a dictionary into a map annotation
    annotation = _dict_to_map(annotation)

    map_ann = MapAnnotationWrapper(conn)
    if annotation_name is not None:
        map_ann.setName(annotation_name)
    if annotation_description is not None:
        map_ann.setDescription(annotation_description)

    map_ann.setNs(namespace)

    map_ann.setValue(annotation)
    map_ann.save()

    _link_annotation(omero_object, map_ann)

    return map_ann


def _create_column(data_type, kwargs):
    column_class = COLUMN_TYPES[data_type]

    return column_class(**kwargs)


def _create_columns(table: Union[DataFrame, list[dict[str, list]], dict[str, list]]) -> list[grid.Column]:
    # TODO: Verify implementation of empty table creation
    if isinstance(table, pd.DataFrame):
        column_names = table.columns.tolist()
        values = [table[c].values.tolist() for c in table.columns]
    elif isinstance(table, list):
        column_names = [next(iter(c)) for c in table]
        values = [table[cn] for cn in column_names]
    elif isinstance(table, dict):
        column_names = list(table.keys())
        values = [table[cn] for cn in column_names]
    else:
        raise TypeError("Table must be a pandas dataframe or a list of dictionaries")

    columns = []
    for cn, v in zip(column_names, values):
        v_type = type(v[0])
        if v_type == str:
            size = (
                len(max(v, key=len)) * 2
            )  # We assume here that the max size is double of what we really have...
            args = {"name": cn, "size": size, "values": v}
            columns.append(_create_column(data_type="string", kwargs=args))
        elif v_type == int:
            if cn.lower() in ["imageid", "image id", "image_id"]:
                args = {"name": cn, "values": v}
                columns.append(_create_column(data_type="image", kwargs=args))
            elif cn.lower() in ["datasetid", "dataset id", "dataset_id"]:
                args = {"name": cn, "values": v}
                columns.append(_create_column(data_type="dataset", kwargs=args))
            elif cn.lower() in ["plateid", "plate id", "plate_id"]:
                args = {"name": cn, "values": v}
                columns.append(_create_column(data_type="plate", kwargs=args))
            elif cn.lower() in ["wellid", "well id", "well_id"]:
                args = {"name": cn, "values": v}
                columns.append(_create_column(data_type="well", kwargs=args))
            elif cn.lower() in ["roiid", "roi id", "roi_id"]:
                args = {"name": cn, "values": v}
                columns.append(_create_column(data_type="roi", kwargs=args))
            elif cn.lower() in ["mask", "maskid", "mask id", "mask_id"]:
                args = {"name": cn, "values": v}
                columns.append(_create_column(data_type="mask", kwargs=args))
            elif cn.lower() in ["fileid", "file id", "file_id"]:
                args = {"name": cn, "values": v}
                columns.append(_create_column(data_type="file", kwargs=args))
            else:
                args = {"name": cn, "values": v}
                columns.append(_create_column(data_type="long", kwargs=args))
        elif v_type == float:
            args = {"name": cn, "values": v}
            columns.append(_create_column(data_type="double", kwargs=args))
        elif v_type == bool:
            args = {"name": cn, "values": v}
            columns.append(_create_column(data_type="string", kwargs=args))
        elif v_type in [ImageWrapper, ImageI]:
            args = {"name": cn, "values": [img.getId() for img in v]}
            columns.append(_create_column(data_type="image", kwargs=args))
        elif v_type in [RoiWrapper, RoiI]:
            args = {"name": cn, "values": [roi.getId() for roi in v]}
            columns.append(_create_column(data_type="roi", kwargs=args))
        elif isinstance(v_type, (list, tuple)):  # We are creating array columns
            raise NotImplementedError(f"Array columns are not implemented. Column {cn}")
        else:
            raise TypeError(f"Could not detect column datatype for column {cn}")

    return columns


def create_table(
    conn: BlitzGateway,
    table: Union[DataFrame, list[dict[str, list]], dict[str, list]],
    table_name: str,
    omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
    table_description: str,
    namespace: str,
):
    """Creates a table annotation from a pandas dataframe or a list of columns as dictionaries."""

    table_name = (
        f'{table_name}_{"".join([choice(ascii_letters) for _ in range(32)])}.h5'
    )
    columns = _create_columns(table)

    resources = conn.c.sf.sharedResources()
    repository_id = resources.repositories().descriptions[0].getId().getValue()
    table = resources.newTable(repository_id, table_name)
    table.initialize(columns)
    table.addData(columns)
    original_file = table.getOriginalFile()
    table.close()

    file_ann = FileAnnotationWrapper(conn)
    if namespace is not None:
        file_ann.setNs(namespace)
    file_ann.setDescription(table_description)
    file_ann.setFile(
        OriginalFileI(original_file.id.val, False)
    )  # TODO: try to get this with a wrapper
    file_ann.save()

    _link_annotation(omero_object, file_ann)

    return file_ann


def create_comment(conn: BlitzGateway,
                   comment_text: str,
                   omero_object: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                   namespace: str):
    if namespace is None:
        namespace = (
            metadata.NSCLIENTMAPANNOTATION
        )  # This makes the annotation editable in the client
    comment_ann = CommentAnnotationWrapper(conn)
    comment_ann.setValue(comment_text)
    comment_ann.setNs(namespace)
    comment_ann.save()

    _link_annotation(omero_object, comment_ann)

    return comment_ann


def _link_annotation(object_wrapper: Union[ImageWrapper, DatasetWrapper, ProjectWrapper],
                     annotation_wrapper: Union[
                         TagAnnotationWrapper,
                         MapAnnotationWrapper,
                         FileAnnotationWrapper,
                         CommentAnnotationWrapper]):
    object_wrapper.linkAnnotation(annotation_wrapper)


def _link_dataset_to_project(conn: BlitzGateway, dataset: DatasetWrapper, project: ProjectWrapper):
    link = ProjectDatasetLinkI()
    link.setParent(ProjectI(project.getId(), False))  # linking to a loaded project might raise exception
    link.setChild(DatasetI(dataset.getId(), False))
    conn.getUpdateService().saveObject(link)


def _link_image_to_dataset(conn: BlitzGateway, image: ImageWrapper, dataset: DatasetWrapper):
    link = DatasetImageLinkI()
    link.setParent(DatasetI(dataset.getId(), False))
    link.setChild(ImageI(image.getId(), False))
    conn.getUpdateService().saveObject(link)
