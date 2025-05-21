# Copyright (c) OpenMMLab. All rights reserved.
# 需要修改的文件位于以下目录中.mmdet/datasets
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Optional, Union

import mmcv
import numpy as np  # Ensure numpy is imported
import pycocotools.mask as maskUtils  # Ensure pycocotools.mask is imported
from mmengine.fileio import get, get_local_path, list_from_file

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class XMLDataset(BaseDetDataset):
    """XML dataset for detection.

    Args:
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        seg_subdir (str, optional): Subdir where segmentation masks are
            stored. Defaults to None.
        seg_map_suffix (str): Suffix for segmentation maps.
            Default: '.png'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 img_subdir: str = 'JPEGImages',
                 ann_subdir: str = 'Annotations',
                 seg_subdir: Optional[str] = None,  # For segmentation masks
                 seg_map_suffix: str = '.png',  # Suffix for seg maps
                 **kwargs) -> None:
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.seg_subdir = seg_subdir
        self.seg_map_suffix = seg_map_suffix
        super().__init__(**kwargs)

    @property
    def sub_data_root(self) -> str:
        """Return the sub data root."""
        return self.data_prefix.get('sub_data_root', '')

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)

        for img_id in img_ids:
            # These paths are relative to sub_data_root (e.g., 'JPEGImages/000032.jpg')
            file_name = osp.join(self.img_subdir, f'{img_id}.jpg')
            # xml_path from load_data_list will be relative to data_root,
            # by joining sub_data_root and ann_subdir.
            # This is consistent with how BaseDetDataset usually expects paths.
            xml_path = osp.join(self.sub_data_root, self.ann_subdir,
                                f'{img_id}.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = file_name  # Relative to sub_data_root
            raw_img_info['xml_path'] = xml_path  # Relative to data_root

            if self.seg_subdir is not None:
                # This path will also be relative to data_root
                seg_map_path_relative_to_data_root = osp.join(self.sub_data_root, self.seg_subdir,
                                                              f'{img_id}{self.seg_map_suffix}')
                raw_img_info['seg_map_path_relative_to_data_root'] = seg_map_path_relative_to_data_root

            parsed_data_info = self.parse_data_info(raw_img_info)
            if parsed_data_info is not None:  # Ensure parse_data_info might return None if image is bad
                data_list.append(parsed_data_info)
        return data_list

    @property
    def bbox_min_size(self) -> Optional[int]:  # Return type should be int or None
        """Return the minimum size of bounding boxes in the images."""
        if self.filter_cfg is not None:
            return self.filter_cfg.get('bbox_min_size', None)
        else:
            return None

    def parse_data_info(self, img_info: dict) -> Optional[dict]:  # Return Optional[dict]
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information. It includes:
                - img_id (str): Image id.
                - file_name (str): Image file name relative to `img_subdir` or `sub_data_root`.
                - xml_path (str): XML file name relative to `data_root`.
                - seg_map_path_relative_to_data_root (str, optional): Segmentation map
                  file name relative to `data_root`.

        Returns:
            Optional[dict]: Parsed annotation. Returns None if critical error occurs.
        """
        data_info = {}
        # Construct full image path
        # self.data_prefix['img_path'] might be set by user, otherwise use sub_data_root
        # This logic is a bit complex, let's assume BaseDetDataset handles self.data_prefix['img_path']
        # or one can simplify it for VOC like this:
        img_path = osp.join(self.data_root, self.sub_data_root, img_info['file_name'])
        data_info['img_path'] = img_path  # Absolute path for image
        data_info['img_id'] = img_info['img_id']
        # xml_path from load_data_list is already relative to data_root
        data_info['xml_path'] = osp.join(self.data_root, img_info['xml_path'])  # Absolute path for xml

        absolute_seg_map_path = None
        if 'seg_map_path_relative_to_data_root' in img_info:
            potential_abs_path = osp.join(self.data_root, img_info['seg_map_path_relative_to_data_root'])
            if osp.exists(potential_abs_path):
                absolute_seg_map_path = potential_abs_path
                # Store the relative path for consistency if other parts expect it,
                # but we'll use absolute_seg_map_path for loading here.
                data_info['seg_map_path'] = img_info['seg_map_path_relative_to_data_root']
            else:
                print(f"ParseDataInfo Warning: Segmentation map file not found at: {potential_abs_path} "
                      f"for img_id {img_info['img_id']}")

        # Deal with xml file
        try:
            with get_local_path(
                    data_info['xml_path'],  # Use absolute xml_path
                    backend_args=self.backend_args) as local_path:
                raw_ann_info = ET.parse(local_path)
        except FileNotFoundError:
            print(f"ParseDataInfo Error: XML file not found at {data_info['xml_path']} for img_id {img_info['img_id']}")
            return None  # Skip this image if XML is missing
        except ET.ParseError:
            print(
                f"ParseDataInfo Error: XML file corrupted or unparseable at {data_info['xml_path']} for img_id {img_info['img_id']}")
            return None

        root = raw_ann_info.getroot()
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            try:
                img_bytes = get(data_info['img_path'], backend_args=self.backend_args)  # Use absolute img_path
                img = mmcv.imfrombytes(img_bytes, backend='cv2')
                height, width = img.shape[:2]
                del img, img_bytes
            except Exception as e:
                print(
                    f"ParseDataInfo Error: Could not read image {data_info['img_path']} to get size for img_id {img_info['img_id']}: {e}")
                return None  # Skip if image can't be read

        data_info['height'] = height
        data_info['width'] = width

        # Parse basic instance info (bbox, label, ignore_flag)
        parsed_instances = self._parse_instance_info(
            raw_ann_info, minus_one=True)

        # If segmentation is expected, load seg_map and add 'mask' to instances
        # Check if the XML indicates segmentation data exists
        segmented_tag = root.find('segmented')
        has_segmentation_in_xml = segmented_tag is not None and segmented_tag.text == '1'

        if self.seg_subdir is not None and has_segmentation_in_xml:
            if absolute_seg_map_path:  # Check if path was resolved and file exists
                try:
                    full_seg_map_img = mmcv.imread(absolute_seg_map_path, flag='unchanged', backend='pillow')
                    if full_seg_map_img is None:
                        raise IOError(f"mmcv.imread failed to load {absolute_seg_map_path}")

                    # Ensure it's single channel (H, W)
                    if full_seg_map_img.ndim == 3 and full_seg_map_img.shape[2] == 1:
                        full_seg_map_img = full_seg_map_img[:, :, 0]
                    elif full_seg_map_img.ndim != 2:
                        print(
                            f"ParseDataInfo Warning: Segmentation map {absolute_seg_map_path} for img_id {img_info['img_id']} "
                            "is not single channel or has unexpected dimensions. Skipping mask processing for this image.")
                    else:  # It's a 2D array, proceed
                        for i, instance in enumerate(parsed_instances):
                            # XML object order corresponds to PNG instance pixel values 1, 2, 3...
                            instance_pixel_value = i + 1
                            binary_mask = (full_seg_map_img == instance_pixel_value)

                            if np.any(binary_mask):
                                rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
                                # COCO RLE 'counts' can be a string after decoding from bytes
                                if isinstance(rle['counts'], bytes):
                                    rle['counts'] = rle['counts'].decode('utf-8')
                                instance['mask'] = rle  # Add RLE mask to instance dict
                            else:
                                # If no pixels match, it's an empty mask for this instance.
                                # LoadAnnotations expects a valid RLE, so provide an empty one.
                                print(
                                    f"ParseDataInfo Info: No mask pixels found for instance {i + 1} (pixel_value={instance_pixel_value}) "
                                    f"in {absolute_seg_map_path} for img_id {img_info['img_id']}. "
                                    f"XML object: {self._metainfo['classes'][instance['bbox_label']] if 'bbox_label' in instance else 'Unknown'}.")
                                empty_binary_mask = np.zeros(full_seg_map_img.shape[:2], dtype=np.uint8)
                                rle = maskUtils.encode(np.asfortranarray(empty_binary_mask))
                                if isinstance(rle['counts'], bytes):
                                    rle['counts'] = rle['counts'].decode('utf-8')
                                instance['mask'] = rle
                except Exception as e:
                    print(f"ParseDataInfo Error: Failed to load or process segmentation map "
                          f"{absolute_seg_map_path} for img_id {img_info['img_id']}: {e}")
                    # Decide if you want to skip the image or proceed without masks
                    # For now, let's proceed without masks if seg map loading fails, LoadAnnotations might error later
                    # or we can clear parsed_instances if masks are critical
                    # To be safe, if masks are required and fail to load, maybe return None
                    # For simplicity now, if an error occurs, instances won't have 'mask' key for this image.

            elif has_segmentation_in_xml:  # seg_subdir was given, XML says segmented=1, but path was not resolved
                print(f"ParseDataInfo Warning: <segmented> is 1 for img_id {img_info['img_id']} "
                      "but segmentation map path could not be resolved or file does not exist. "
                      "Instances will not have masks.")

        data_info['instances'] = parsed_instances
        return data_info

    def _parse_instance_info(self,
                             raw_ann_info: ET.ElementTree,  # Type hint fix
                             minus_one: bool = True) -> List[dict]:
        """parse instance information.

        Args:
            raw_ann_info (ElementTree): ElementTree object.
            minus_one (bool): Whether to subtract 1 from the coordinates.
                Defaults to True.

        Returns:
            List[dict]: List of instances.
        """
        instances = []
        for obj in raw_ann_info.findall('object'):
            instance = {}
            name_node = obj.find('name')
            if name_node is None or name_node.text is None:  # Robustness
                print(f"Warning: Object without name found in XML, skipping.")
                continue
            name = name_node.text
            if name not in self.cat2label:  # Use self.cat2label
                print(f"Warning: Class '{name}' not in dataset metainfo classes, skipping object.")
                continue

            difficult_node = obj.find('difficult')
            difficult = 0 if difficult_node is None or difficult_node.text is None else int(difficult_node.text)

            bnd_box_node = obj.find('bndbox')
            if bnd_box_node is None:  # Robustness
                print(f"Warning: Object '{name}' without bndbox found in XML, skipping.")
                continue

            try:
                bbox = [
                    int(float(bnd_box_node.find('xmin').text)),
                    int(float(bnd_box_node.find('ymin').text)),
                    int(float(bnd_box_node.find('xmax').text)),
                    int(float(bnd_box_node.find('ymax').text))
                ]
            except (ValueError, AttributeError) as e:  # Robustness for missing/malformed bbox coords
                print(f"Warning: Malformed bbox for object '{name}' in XML, skipping object. Error: {e}")
                continue

            # VOC needs to subtract 1 from the coordinates
            if minus_one:
                bbox = [x - 1 for x in bbox]

            ignore = False
            # Ensure self.bbox_min_size is an int if not None
            min_bbox_size_val = self.bbox_min_size
            if min_bbox_size_val is not None and not self.test_mode:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < min_bbox_size_val or h < min_bbox_size_val:
                    ignore = True

            if difficult or ignore:
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[name]
            instances.append(instance)
        return instances

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        # Ensure data_list is not None (it could be if all parse_data_info returned None)
        if not self.data_list:
            return []

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if data_info is None:  # Skip if parse_data_info returned None
                continue
            width = data_info['width']
            height = data_info['height']
            # Also check if 'instances' key exists, as it might be missing if XML parsing failed badly
            if filter_empty_gt and len(data_info.get('instances', [])) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos