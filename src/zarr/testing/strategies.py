import sys
from typing import Any, Literal

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numcodecs
import numcodecs.zarr3 as ncodecs
import numpy as np
from hypothesis import given, settings  # noqa: F401
from hypothesis.strategies import SearchStrategy

import zarr
from zarr import codecs as zcodecs
from zarr.abc.store import RangeByteRequest
from zarr.core.array import Array
from zarr.core.common import ZarrFormat
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StoreLike
from zarr.storage._common import _dereference_path

# Copied from Xarray
_attr_keys = st.text(st.characters(), min_size=1)
_attr_values = st.recursive(
    st.none() | st.booleans() | st.text(st.characters(), max_size=5),
    lambda children: st.lists(children) | st.dictionaries(_attr_keys, children),
    max_leaves=3,
)


def v3_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.boolean_dtypes()
        | npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        # | npst.byte_string_dtypes(endianness="=")
        # | npst.unicode_string_dtypes()
        # | npst.datetime64_dtypes()
        # | npst.timedelta64_dtypes()
    )


def v2_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.boolean_dtypes()
        | npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        | npst.byte_string_dtypes(endianness="=")
        | npst.unicode_string_dtypes(endianness="=")
        | npst.datetime64_dtypes(endianness="=")
        # | npst.timedelta64_dtypes()
    )


def safe_unicode_for_dtype(dtype: np.dtype[np.str_]) -> st.SearchStrategy[str]:
    """Generate UTF-8-safe text constrained to max_len of dtype."""
    # account for utf-32 encoding (i.e. 4 bytes/character)
    max_len = max(1, dtype.itemsize // 4)

    return st.text(
        alphabet=st.characters(
            blacklist_categories=["Cs"],  # Avoid *technically allowed* surrogates
            min_codepoint=32,
        ),
        min_size=1,
        max_size=max_len,
    )


# From https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#node-names
# 1. must not be the empty string ("")
# 2. must not include the character "/"
# 3. must not be a string composed only of period characters, e.g. "." or ".."
# 4. must not start with the reserved prefix "__"
zarr_key_chars = st.sampled_from(
    ".-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
)
node_names = st.text(zarr_key_chars, min_size=1).filter(
    lambda t: t not in (".", "..") and not t.startswith("__")
)
array_names = node_names
attrs = st.none() | st.dictionaries(_attr_keys, _attr_values)
keys = st.lists(node_names, min_size=1).map("/".join)
paths = st.just("/") | keys
# st.builds will only call a new store constructor for different keyword arguments
# i.e. stores.examples() will always return the same object per Store class.
# So we map a clear to reset the store.
stores = st.builds(MemoryStore, st.just({})).map(lambda x: sync(x.clear()))
zarr_formats: st.SearchStrategy[ZarrFormat] = st.sampled_from([3, 2])
array_shapes = npst.array_shapes(max_dims=4, min_side=0)


@st.composite  # type: ignore[misc]
def codecs(
    draw: st.DrawFn,
    *,
    zarr_formats: st.SearchStrategy[Literal[2, 3]] = zarr_formats,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
) -> Any:
    zarr_format = draw(zarr_formats)
    # we intentional don't parameterize over `level` or `clevel` to reduce the search space
    zarr_codecs = st.one_of(
        st.builds(zcodecs.ZstdCodec),
        st.builds(
            zcodecs.BloscCodec,
            shuffle=st.builds(
                zcodecs.BloscShuffle.from_int, num=st.integers(min_value=0, max_value=2)
            ),
        ),
        st.builds(zcodecs.GzipCodec),
        st.builds(zcodecs.Crc32cCodec),
    )
    num_codecs_v2 = st.one_of(
        st.builds(numcodecs.Zlib),
        st.builds(numcodecs.LZMA),
        st.builds(numcodecs.Zstd),
        st.builds(numcodecs.Zlib),
    )
    num_codecs_v3 = st.one_of(
        st.builds(ncodecs.Blosc),
        st.builds(ncodecs.LZMA),
        # st.builds(ncodecs.PCodec),
        # st.builds(ncodecs.ZFPY),
    )
    codec_kwargs = {"filters": draw(st.none() | st.just(()))}
    if zarr_format == 2:
        codec_kwargs["compressors"] = draw(num_codecs_v2 | st.none() | st.just(()))
    else:
        # Intentionally prioritize using a codec over no codec
        codec_kwargs["compressors"] = draw(zarr_codecs | num_codecs_v3 | st.none() | st.just(()))
    return codec_kwargs


@st.composite  # type: ignore[misc]
def numpy_arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
    zarr_formats: st.SearchStrategy[ZarrFormat] = zarr_formats,
) -> Any:
    """
    Generate numpy arrays that can be saved in the provided Zarr format.
    """
    zarr_format = draw(zarr_formats)
    dtype = draw(v3_dtypes() if zarr_format == 3 else v2_dtypes())
    if np.issubdtype(dtype, np.str_):
        safe_unicode_strings = safe_unicode_for_dtype(dtype)
        return draw(npst.arrays(dtype=dtype, shape=shapes, elements=safe_unicode_strings))

    return draw(npst.arrays(dtype=dtype, shape=shapes))


@st.composite  # type: ignore[misc]
def np_array_and_chunks(
    draw: st.DrawFn, *, arrays: st.SearchStrategy[np.ndarray] = numpy_arrays
) -> tuple[np.ndarray, tuple[int, ...]]:  # type: ignore[type-arg]
    """A hypothesis strategy to generate small sized random arrays.

    Returns: a tuple of the array and a suitable random chunking for it.
    """
    array = draw(arrays)
    # We want this strategy to shrink towards arrays with smaller number of chunks
    # 1. st.integers() shrinks towards smaller values. So we use that to generate number of chunks
    numchunks = draw(
        st.tuples(
            *[st.integers(min_value=0 if size == 0 else 1, max_value=size) for size in array.shape]
        )
    )
    # 2. and now generate the chunks tuple
    chunks = tuple(
        size // nchunks if nchunks > 0 else 0
        for size, nchunks in zip(array.shape, numchunks, strict=True)
    )
    return (array, chunks)


@st.composite  # type: ignore[misc]
def arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
    stores: st.SearchStrategy[StoreLike] = stores,
    paths: st.SearchStrategy[str | None] = paths,
    array_names: st.SearchStrategy = array_names,
    arrays: st.SearchStrategy | None = None,
    attrs: st.SearchStrategy = attrs,
    codecs: st.SearchStrategy = codecs,
    zarr_formats: st.SearchStrategy = zarr_formats,
) -> Array:
    store = draw(stores)
    path = draw(paths)
    name = draw(array_names)
    attributes = draw(attrs)
    zarr_format = draw(zarr_formats)
    if arrays is None:
        arrays = numpy_arrays(shapes=shapes, zarr_formats=st.just(zarr_format))
    nparray, chunks = draw(np_array_and_chunks(arrays=arrays))
    # test that None works too.
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(nparray.dtype)]))

    expected_attrs = {} if attributes is None else attributes

    array_path = _dereference_path(path, name)
    root = zarr.open_group(store, mode="w", zarr_format=zarr_format)
    codec_kwargs = draw(codecs(zarr_formats=st.just(zarr_format), dtypes=st.just(nparray.dtype)))
    a = root.create_array(
        array_path,
        shape=nparray.shape,
        chunks=chunks,
        dtype=nparray.dtype,
        attributes=attributes,
        fill_value=fill_value,
        **codec_kwargs,
    )

    assert isinstance(a, Array)
    if a.metadata.zarr_format == 3:
        assert a.fill_value is not None
    assert a.name is not None
    assert isinstance(root[array_path], Array)
    assert nparray.shape == a.shape
    assert chunks == a.chunks
    assert array_path == a.path, (path, name, array_path, a.name, a.path)
    assert a.basename == name, (a.basename, name)
    assert dict(a.attrs) == expected_attrs

    a[:] = nparray

    return a


def is_negative_slice(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.step is not None and idx.step < 0


@st.composite  # type: ignore[misc]
def basic_indices(draw: st.DrawFn, *, shape: tuple[int], **kwargs: Any) -> Any:
    """Basic indices without unsupported negative slices."""
    return draw(
        npst.basic_indices(shape=shape, **kwargs).filter(
            lambda idxr: (
                not (
                    is_negative_slice(idxr)
                    or (isinstance(idxr, tuple) and any(is_negative_slice(idx) for idx in idxr))
                )
            )
        )
    )


@st.composite  # type: ignore[misc]
def orthogonal_indices(
    draw: st.DrawFn, *, shape: tuple[int]
) -> tuple[tuple[np.ndarray[Any, Any], ...], tuple[np.ndarray[Any, Any], ...]]:
    """
    Strategy that returns
    (1) a tuple of integer arrays used for orthogonal indexing of Zarr arrays.
    (2) an tuple of integer arrays that can be used for equivalent indexing of numpy arrays
    """
    zindexer = []
    npindexer = []
    ndim = len(shape)
    for axis, size in enumerate(shape):
        val = draw(
            npst.integer_array_indices(
                shape=(size,), result_shape=npst.array_shapes(min_side=1, max_side=size, max_dims=1)
            )
            | basic_indices(min_dims=1, shape=(size,), allow_ellipsis=False)
            .map(lambda x: (x,) if not isinstance(x, tuple) else x)  # bare ints, slices
            .filter(lambda x: bool(x))  # skip empty tuple
        )
        (idxr,) = val
        if isinstance(idxr, int):
            idxr = np.array([idxr])
        zindexer.append(idxr)
        if isinstance(idxr, slice):
            idxr = np.arange(*idxr.indices(size))
        elif isinstance(idxr, (tuple, int)):
            idxr = np.array(idxr)
        newshape = [1] * ndim
        newshape[axis] = idxr.size
        npindexer.append(idxr.reshape(newshape))

    # casting the output of broadcast_arrays is needed for numpy 1.25
    return tuple(zindexer), tuple(np.broadcast_arrays(*npindexer))


def key_ranges(
    keys: SearchStrategy = node_names, max_size: int = sys.maxsize
) -> SearchStrategy[list[int]]:
    """
    Function to generate key_ranges strategy for get_partial_values()
    returns list strategy w/ form::

        [(key, (range_start, range_end)),
         (key, (range_start, range_end)),...]
    """

    def make_request(start: int, length: int) -> RangeByteRequest:
        return RangeByteRequest(start, end=min(start + length, max_size))

    byte_ranges = st.builds(
        make_request,
        start=st.integers(min_value=0, max_value=max_size),
        length=st.integers(min_value=0, max_value=max_size),
    )
    key_tuple = st.tuples(keys, byte_ranges)
    return st.lists(key_tuple, min_size=1, max_size=10)
