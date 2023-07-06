import json
from itertools import product
from random import choice
from string import ascii_letters
from typing import Union

import numpy as np
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
    ProjectAnnotationLinkI,
    DatasetImageLinkI,
    TagAnnotationI,
    TagAnnotationLinkI,
    ProjectDatasetLinkI,
)
from omero.rtypes import rdouble, rint, rlong, rstring
from pandas import DataFrame

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
    image, z_range=None, c_range=None, t_range=None, y_range=None, x_range=None
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


def create_image_copy(conn,
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


def create_image(conn, image_name, size_x, size_y, size_z, size_t, size_c, data_type, channel_labels=None, image_description=None):
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
                                  image_description: str=None,
                                  channel_labels: Union[list, tuple]=None,
                                  dataset: DatasetWrapper=None,
                                  source_image_id: int=None,
                                  channels_list: list[int]=None,
                                  force_whole_planes: bool=False):
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
            new_image = create_image_copy(conn, source_image_id,
                                          image_name=image_name,
                                          image_description=image_description,
                                          size_x=data.shape[-1],
                                          size_y=data.shape[-2],
                                          size_z=data.shape[0],
                                          size_t=data.shape[2],
                                          channel_list=channels_list)

        else:
            new_image = create_image(conn,
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
    type_to_func = {
        "point": _create_shape_point,
        "line": _create_shape_line,
        "rectangle": _create_shape_rectangle,
        "ellipse": _create_shape_ellipse,
        "polygon": _create_shape_polygon,
        "mask": _create_shape_mask,
    }
    shapes = [type_to_func[shape["type"]](**shape["args"]) for shape in shapes]

    # create an ROI, link it to Image
    roi = RoiI()  # TODO: work with wrappers
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(image._obj)
    if name is not None:
        roi.setName(rstring(name))
    if description is not None:
        roi.setDescription(rstring(name))
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return conn.getUpdateService().saveAndReturnObject(roi)


def _rgba_to_int(red, green, blue, alpha=255):
    """Return the color as an Integer in RGBA encoding"""
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    rgba_int = sum([r, g, b, a])
    if rgba_int > (2**31 - 1):  # convert to signed 32-bit int
        rgba_int = rgba_int - 2**32

    return rgba_int


def _set_shape_properties(
    shape,
    name=None,
    fill_color=(10, 10, 10, 10),
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    if name:
        shape.setTextValue(rstring(name))
    shape.setFillColor(rint(_rgba_to_int(*fill_color)))
    shape.setStrokeColor(rint(_rgba_to_int(*stroke_color)))
    shape.setStrokeWidth(LengthI(stroke_width, enums.UnitsLength.PIXEL))


def _create_shape_point(
    x_pos,
    y_pos,
    z_pos=None,
    c_pos=None,
    t_pos=None,
    name=None,
    stroke_color=(255, 255, 255, 255),
    fill_color=(10, 10, 10, 20),
    stroke_width=1,
):
    point = PointI()
    point.x = rdouble(x_pos)
    point.y = rdouble(y_pos)
    if z_pos is not None:
        point.theZ = rint(z_pos)
    if c_pos is not None:
        point.theC = rint(c_pos)
    if t_pos is not None:
        point.theT = rint(t_pos)
    _set_shape_properties(
        shape=point,
        name=name,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill_color=fill_color,
    )

    return point


def _create_shape_line(
    x1_pos,
    y1_pos,
    x2_pos,
    y2_pos,
    c_pos=None,
    z_pos=None,
    t_pos=None,
    name=None,
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    line = LineI()
    line.x1 = rdouble(x1_pos)
    line.x2 = rdouble(x2_pos)
    line.y1 = rdouble(y1_pos)
    line.y2 = rdouble(y2_pos)
    line.theZ = rint(z_pos)
    line.theT = rint(t_pos)
    if c_pos is not None:
        line.theC = rint(c_pos)
    _set_shape_properties(
        line, name=name, stroke_color=stroke_color, stroke_width=stroke_width
    )
    return line


def _create_shape_rectangle(
    x_pos,
    y_pos,
    width,
    height,
    z_pos,
    t_pos,
    rectangle_name=None,
    fill_color=(10, 10, 10, 255),
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    rect = RectangleI()
    rect.x = rdouble(x_pos)
    rect.y = rdouble(y_pos)
    rect.width = rdouble(width)
    rect.height = rdouble(height)
    rect.theZ = rint(z_pos)
    rect.theT = rint(t_pos)
    _set_shape_properties(
        shape=rect,
        name=rectangle_name,
        fill_color=fill_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    return rect


def _create_shape_ellipse(
    x_pos,
    y_pos,
    x_radius,
    y_radius,
    z_pos,
    t_pos,
    ellipse_name=None,
    fill_color=(10, 10, 10, 255),
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    ellipse = EllipseI()
    ellipse.setX(rdouble(x_pos))
    ellipse.setY(rdouble(y_pos))  # TODO: setters and getters everywhere
    ellipse.radiusX = rdouble(x_radius)
    ellipse.radiusY = rdouble(y_radius)
    ellipse.theZ = rint(z_pos)
    ellipse.theT = rint(t_pos)
    _set_shape_properties(
        ellipse,
        name=ellipse_name,
        fill_color=fill_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    return ellipse


def _create_shape_polygon(
    points_list,
    z_pos,
    t_pos,
    polygon_name=None,
    fill_color=(10, 10, 10, 255),
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    polygon = PolygonI()
    points_str = "".join(
        ["".join([str(x), ",", str(y), ", "]) for x, y in points_list]
    )[:-2]
    polygon.points = rstring(points_str)
    polygon.theZ = rint(z_pos)
    polygon.theT = rint(t_pos)
    _set_shape_properties(
        polygon,
        name=polygon_name,
        fill_color=fill_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    return polygon


def _create_shape_mask(
    mask_array, x_pos, y_pos, z_pos, t_pos, mask_name=None, fill_color=(10, 10, 10, 255)
):
    mask = MaskI()
    mask.setX(rdouble(x_pos))
    mask.setY(rdouble(y_pos))
    mask.setTheZ(rint(z_pos))
    mask.setTheT(rint(t_pos))
    mask.setWidth(rdouble(mask_array.shape[0]))
    mask.setHeight(rdouble(mask_array.shape[1]))
    mask.setFillColor(rint(_rgba_to_int(*fill_color)))
    if mask_name:
        mask.setTextValue(rstring(mask_name))
    mask_packed = np.packbits(mask_array)  # TODO: raise error when not boolean array
    mask.setBytes(mask_packed.tobytes())

    return mask


def create_tag(conn, tag_string, omero_object, description=None):
    tag_ann = TagAnnotationWrapper(conn)
    tag_ann.setValue(tag_string)
    if description is not None:
        tag_ann.setDescription(description)
    tag_ann.save()

    _link_annotation(omero_object, tag_ann)


def _serialize_map_value(value):
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except ValueError as e:
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


def _create_columns(table):
    # TODO: Verify implementation of empty table creation
    column_names = table.columns.tolist()
    values = [table[c].values.tolist() for c in table.columns]

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
    table: DataFrame,
    table_name,
    omero_object,
    table_description,
    namespace=None,
):
    """Creates a table annotation from a pandas dataframe"""

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


def create_comment(conn, comment_value, omero_object, namespace=None):
    if namespace is None:
        namespace = (
            metadata.NSCLIENTMAPANNOTATION
        )  # This makes the annotation editable in the client
    comment_ann = CommentAnnotationWrapper(conn)
    comment_ann.setValue(comment_value)
    comment_ann.setNs(namespace)
    comment_ann.save()

    _link_annotation(omero_object, comment_ann)


def _link_annotation(object_wrapper, annotation_wrapper):
    object_wrapper.linkAnnotation(annotation_wrapper)


def _link_dataset_to_project(conn, dataset, project):
    link = ProjectDatasetLinkI()
    link.setParent(ProjectI(project.getId(), False))  # linking to a loaded project might raise exception
    link.setChild(DatasetI(dataset.getId(), False))
    conn.getUpdateService().saveObject(link)


def _link_image_to_dataset(conn, image, dataset):
    link = DatasetImageLinkI()
    link.setParent(DatasetI(dataset.getId(), False))
    link.setChild(ImageI(image.getId(), False))
    conn.getUpdateService().saveObject(link)
