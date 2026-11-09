import os
import pathlib
import warnings

import h5py
import lxml.etree as ET

import openmc
from openmc._xml import get_elem_list, get_text, clean_indentation


class DataLibrary(list):
    """Collection of cross section data libraries.

    This class behaves like a list where each item is a dictionary summarizing
    cross section data from a single file. The dictionary has keys 'path',
    'type', and 'materials'.

    When the same material appears more than once for the same data type, the
    entry that comes first is used, both by :meth:`get_by_material` and by
    OpenMC when it reads the exported XML file. Use
    :meth:`register_file` with ``replace=True`` or :meth:`update` to override
    data that a library already provides.

    .. versionchanged:: 0.14.0
        This class now behaves like a list rather than requiring you to access
        the list of libraries through a special attribute.

    """

    def __init__(self, *args):
        super().__init__(*args)

    @property
    def libraries(self):
        # For backwards compatibility
        return self

    def get_by_material(self, name, data_type='neutron'):
        """Return the library dictionary containing a given material.

        If several entries contain the material, the first one is returned,
        which is also the one OpenMC uses.

        Parameters
        ----------
        name : str
            Name of material, e.g. 'Am241'
        data_type : str
            Name of data type, e.g. 'neutron', 'photon', 'photonuclear', 'wmp', or 'thermal'

            .. versionadded:: 0.12

        Returns
        -------
        library : dict or None
            Dictionary summarizing cross section data from a single file;
            the dictionary has keys 'path', 'type', and 'materials'.

        """
        for library in self:
            if name in library['materials'] and library['type'] == data_type:
                return library
        return None

    def remove_by_material(self, name, data_type='neutron'):
        """Remove the data of one or several materials.

        Every entry of the given data type that contains a material is
        removed; an entry that holds several materials loses only the ones
        given.

        Parameters
        ----------
        name : str or iterable of str
            Name of material, e.g. 'Am241', or several names

            .. versionchanged:: 0.16.1
                Several names can be given, and every matching entry is
                removed, not only the first.
        data_type : str or None
            Name of data type, e.g. 'neutron', 'photon', 'photonuclear',
            'wmp', or 'thermal'; None removes the materials from every type

        """
        names = {name} if isinstance(name, str) else set(name)
        kept = []
        for library in self:
            if library['type'] == 'depletion_chain' or (
                    data_type is not None and library['type'] != data_type):
                kept.append(library)
                continue
            remaining = [m for m in library['materials'] if m not in names]
            if len(remaining) == len(library['materials']):
                kept.append(library)
            elif remaining:
                kept.append({**library, 'materials': remaining})
        self[:] = kept

    def duplicates(self):
        """Materials listed more than once for the same data type.

        Returns
        -------
        dict
            Mapping of (data type, material) to the paths of the files that
            provide it, in order; the first path is the one that is used

        """
        seen = {}
        for library in self:
            for material in library['materials']:
                seen.setdefault((library['type'], material), []).append(
                    library['path'])
        return {key: paths for key, paths in seen.items() if len(paths) > 1}

    def _add(self, library, mode):
        """Add a library entry.

        For data the library already holds (same materials and data type),
        mode 'replace' puts the entry in place of the existing one, 'append'
        appends it with a warning, since the existing data keeps precedence,
        and 'missing' adds only the materials that are not present yet.
        """
        if library['type'] == 'depletion_chain':
            chains = [lib for lib in self if lib['type'] == 'depletion_chain']
            if chains and mode == 'replace':
                self[:] = [lib for lib in self
                           if lib['type'] != 'depletion_chain']
            elif chains and mode == 'missing':
                return
            elif chains:
                warnings.warn(
                    f"The library already has the depletion chain "
                    f"{chains[0]['path']}; it takes precedence over "
                    f"{library['path']}. Use replace=True to override it.",
                    stacklevel=3)
            self.append(library)
            return

        materials = set(library['materials'])
        overlap = [i for i, lib in enumerate(self)
                   if lib['type'] == library['type']
                   and not materials.isdisjoint(lib['materials'])]
        if not overlap:
            self.append(library)
        elif mode == 'missing':
            present = {m for i in overlap for m in self[i]['materials']}
            remaining = [m for m in library['materials'] if m not in present]
            if remaining:
                self.append({**library, 'materials': remaining})
        elif mode == 'append':
            names = sorted(materials.intersection(
                m for i in overlap for m in self[i]['materials']))
            warnings.warn(
                f"{library['type']} data for {', '.join(names)} is already in "
                f"the library; the entry from {self[overlap[0]]['path']} "
                f"takes precedence over {library['path']}. Use "
                "replace=True to override it.", stacklevel=3)
            self.append(library)
        else:
            # Put the new entry where the first replaced one was, so that it
            # keeps that precedence, and take its materials out of the others
            kept = []
            for i, lib in enumerate(self):
                if i == overlap[0]:
                    kept.append(library)
                if i in overlap:
                    remaining = [m for m in lib['materials']
                                 if m not in materials]
                    if remaining:
                        kept.append({**lib, 'materials': remaining})
                else:
                    kept.append(lib)
            self[:] = kept

    def register_file(self, filename, replace=False):
        """Register a file with the data library.

        Parameters
        ----------
        filename : str or Path
            Path to the file to be registered.
            If an ``xml`` file, treat as the depletion chain file without
            materials.
        replace : bool
            If True, the file replaces the data the library already holds
            for the same materials and data type, at the same position. If
            False (default), it is appended, and a warning is issued if it
            duplicates materials that are already present, since the first
            entry is the one that is used.

            .. versionadded:: 0.16.1

        """
        if not isinstance(filename, pathlib.Path):
            path = pathlib.Path(filename)
        else:
            path = filename

        if path.suffix == '.xml':
            filetype = 'depletion_chain'
            materials = []
        elif path.suffix == '.h5':
            with h5py.File(path, 'r') as h5file:
                filetype = h5file.attrs['filetype'].decode()[5:]
                materials = list(h5file)
        else:
            raise ValueError(
                f"File type {path.name} not supported by {self.__class__.__name__}")

        library = {'path': str(path), 'type': filetype, 'materials': materials}
        self._add(library, 'replace' if replace else 'append')

    def update(self, other, replace=True):
        """Add the entries of another library.

        Parameters
        ----------
        other : openmc.data.DataLibrary or str or os.PathLike
            Library, or path to its XML file, whose entries are added in
            order
        replace : bool
            If True (default), data from ``other`` replaces the data this
            library holds for the same materials and data type, at the same
            position, so that ``other`` takes precedence. If False, only the
            materials this library does not provide yet are added.

            .. versionadded:: 0.16.1

        """
        if not isinstance(other, DataLibrary):
            other = type(self).from_xml(other)
        mode = 'replace' if replace else 'missing'
        for library in other:
            self._add(dict(library), mode)

    def export_to_xml(self, path='cross_sections.xml', absolute=False):
        """Export cross section data library to an XML file.

        Parameters
        ----------
        path : str
            Path to file to write. Defaults to 'cross_sections.xml'.
        absolute : bool
            If False (default), files inside the directory of the XML file
            are written with paths relative to it, so that the directory can
            be moved or shared as a whole, and other files with absolute
            paths, so that the XML file alone can be moved. If True, all
            paths are absolute.

            .. versionadded:: 0.16.1

        """
        root = ET.Element('cross_sections')
        xml_dir = os.path.dirname(os.path.abspath(path))

        def file_path(library):
            p = os.path.abspath(library['path'])
            if absolute:
                return p
            try:
                rel = os.path.relpath(p, xml_dir)
            except ValueError:
                # e.g. another drive on Windows
                return p
            outside = rel == os.pardir or rel.startswith(os.pardir + os.sep)
            return p if outside else rel

        for library in self:
            if library['type'] == "depletion_chain":
                lib_element = ET.SubElement(root, "depletion_chain")
            else:
                lib_element = ET.SubElement(root, "library")
                lib_element.set('materials', ' '.join(library['materials']))
            lib_element.set('path', file_path(library))
            lib_element.set('type', library['type'])

        # Clean the indentation to be user-readable
        clean_indentation(root)

        # Write XML file
        tree = ET.ElementTree(root)
        tree.write(str(path), xml_declaration=True, encoding='utf-8',
                   method='xml')

    @classmethod
    def from_xml(cls, path=None):
        """Read cross section data library from an XML file.

        A warning is issued if a material is listed more than once for the
        same data type; the first entry is the one that is used.

        Parameters
        ----------
        path : str, optional
            Path to XML file to read. If not provided,
            openmc.config['cross_sections'] will be used.

        Returns
        -------
        data : openmc.data.DataLibrary
            Data library object initialized from the provided XML

        """

        data = cls()

        # If path is None, get the cross sections from the global configuration
        if path is None:
            path = openmc.config.get('cross_sections')

        # Check to make sure we picked up cross sections
        if path is None:
            raise ValueError("Either path or openmc.config['cross_sections'] "
                             "must be set")

        tree = ET.parse(path)
        root = tree.getroot()
        if root.find('directory') is not None:
            directory = root.find('directory').text
        else:
            directory = os.path.dirname(path)

        for lib_element in root.findall('library'):
            filename = os.path.join(directory, get_text(lib_element, "path"))
            filetype = get_text(lib_element, "type")
            materials = get_elem_list(lib_element, "materials", str) or []
            library = {'path': filename, 'type': filetype,
                       'materials': materials}
            data.libraries.append(library)

        # get depletion chain data
        dep_node = root.find("depletion_chain")
        if dep_node is not None:
            filename = os.path.join(directory, get_text(dep_node, "path"))
            library = {'path': filename, 'type': 'depletion_chain',
                       'materials': []}
            data.libraries.append(library)

        duplicates = data.duplicates()
        if duplicates:
            listed = ', '.join(f'{m} ({t})' for t, m in list(duplicates)[:5])
            more = len(duplicates) - 5
            warnings.warn(
                f"{path} lists {len(duplicates)} material(s) more than once "
                f"for the same data type: {listed}"
                f"{f' and {more} more' if more > 0 else ''}. The first entry "
                "of each is used.", stacklevel=2)

        return data
