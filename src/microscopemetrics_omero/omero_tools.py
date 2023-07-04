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

def get_image_shape(image):
    try:
        image_shape = (image.getSizeZ(),
                       image.getSizeC(),
                       image.getSizeT(),
                       image.getSizeY(),
                       image.getSizeX())
    except Exception as e:
        raise e

    return image_shape


def get_pixel_size(image, order='ZXY'):
    pixels = image.getPrimaryPixels()

    order = order.upper()
    if order not in ['ZXY', 'ZYX', 'XYZ', 'XZY', 'YXZ', 'YZX']:
        raise ValueError('The provided order for the axis is not valid')
    pixel_sizes = ()
    for a in order:
        pixel_sizes += (getattr(pixels, f'getPhysicalSize{a}')().getValue(), )
    return pixel_sizes


def get_pixel_size_units(image):
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
    image_shape = get_image_shape(image)

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


def create_image(
    conn,
    image,
    source_image,
    image_name,
    description,
):
    zct_list = list(
        product(
            range(image.shape[0]),
            range(image.shape[1]),
            range(image.shape[2]),
        )
    )
    zct_generator = (image[z, c, t, :, :] for z, c, t in zct_list)

    new_image = conn.createImageFromNumpySeq(
        zctPlanes=zct_generator,
        imageName=image_name,
        sizeZ=image.shape[0],
        sizeC=image.shape[1],
        sizeT=image.shape[2],
        description=description,
        dataset=source_image.getParent(),
        sourceImageId=source_image.getId(),
    )

    # TODO: see with ome why description is not applied
    img_wrapper = conn.getObject("Image", new_image.getId())
    img_wrapper.setDescription(description)
    img_wrapper.save()

    return new_image


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
    else:
        try:
            return json.dumps(value)
        except ValueError as e:
            # TODO: log an error
            return json.dumps(value.__str__())


def _dict_to_map(dictionary):
    """Converts a dictionary into a list of key:value pairs to be fed as map annotation.
    If value is not a string we serialize it as a json string"""
    map_annotation = [[str(k), _serialize_map_value(v)] for k, v in dictionary.items()]
    return map_annotation


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
    for i, (cn, v) in enumerate(zip(column_names, values)):
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
